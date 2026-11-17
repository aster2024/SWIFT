"""Step-level SWIFT probe: a frozen LLM, a SWIFT gated linear head on its hidden states, aggregated per reasoning step.

Usage:
    python step/train_step_probe.py <model_dir> <data_dir> <out_dir> [--layers last4|all|33,34,35,36] [--seed 42] [--epochs 1] [--limit N]

<data_dir> is the output of prepare_prm800k.py (train.jsonl, dev.jsonl).

Model: hidden states of the selected layers are concatenated per token (default: the last 4 hidden states, as recommended
in the SWIFT README). A fused Linear(dim, 2) gives a gate g_t = sigmoid(z0) and a reward r_t = z1 per token (SWIFT's
LinearRewardModel). The score of step k is the gated mean over its token span: s_k = sum_t g_t r_t / sum_t g_t.
Training: step-level BCE (correct = 1, incorrect = 0), AdamW lr 1e-4, weight decay 1e-5 (SWIFT defaults), one update per
batch of at most 16384 tokens, 1 epoch. Features are computed on the fly; the LLM is never updated.
Threshold: after training, tau is chosen on the dev split from {0.05, ..., 0.95} by first-error F1.
Prediction: the first step with sigmoid(s_k) < tau is the first error; if none, the solution is predicted correct (-1).
Saves probe.safetensors and config.json.
"""
import os, sys, json, time, random, argparse
import torch, torch.nn as nn
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_features import build_example
from transformers import AutoTokenizer, AutoModelForCausalLM

LAYERS = None          # set in main() or by the evaluator from config.json
TOK_BUDGET = 16384


class StepProbe(nn.Module):
    """SWIFT's LinearRewardModel with per-step gated aggregation."""

    def __init__(self, dim):
        super().__init__()
        self.fused_layer = nn.Linear(dim, 2)

    def forward(self, feats, spans):
        """feats: (T, dim) token features of one sequence; spans: list of (start, end). Returns step logits (K,)."""
        z = self.fused_layer(feats)
        g, r = torch.sigmoid(z[:, 0]), z[:, 1]
        return torch.stack([(g[a:b] * r[a:b]).sum() / g[a:b].sum().clamp(min=1e-8) for a, b in spans])


def resolve_layers(spec, num_hidden_layers):
    n = num_hidden_layers + 1          # hidden_states includes the embedding output at index 0
    if spec == "last4":
        return list(range(n - 4, n))
    if spec == "all":
        return list(range(n))
    return [int(x) for x in spec.split(",")]


def batches(ex, budget, shuffle, seed):
    idx = list(range(len(ex)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    else:
        idx.sort(key=lambda i: len(ex[i]["ids"]))
    cur, cur_max = [], 0
    for i in idx:
        L = len(ex[i]["ids"])
        if cur and max(cur_max, L) * (len(cur) + 1) > budget:
            yield cur; cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur:
        yield cur


def forward_feats(model, tok, ex, b):
    T = max(len(ex[i]["ids"]) for i in b)
    ids = torch.full((len(b), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    for k, i in enumerate(b):
        n = len(ex[i]["ids"]); ids[k, :n] = torch.tensor(ex[i]["ids"]); att[k, :n] = 1
    with torch.inference_mode():
        out = model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
        hs = torch.cat([out.hidden_states[l] for l in LAYERS], -1)
    return [hs[k, : len(ex[i]["ids"])].float().clone() for k, i in enumerate(b)]


def first_error_pred(probs, tau):
    for k, p in enumerate(probs):
        if p < tau:
            return k
    return -1


def f1_first_error(preds, golds):
    """ProcessBench metric for one subset: harmonic mean of exact first-error accuracy on erroneous solutions
    and accuracy of predicting -1 on correct solutions."""
    err = [(p, g) for p, g in zip(preds, golds) if g != -1]
    cor = [p for p, g in zip(preds, golds) if g == -1]
    a = sum(p == g for p, g in err) / max(len(err), 1)
    b = sum(p == -1 for p in cor) / max(len(cor), 1)
    return (0.0 if a + b == 0 else 2 * a * b / (a + b)), a, b


def load(path, tok, limit=0):
    ex = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        ids, spans = build_example(x["problem"], x["steps"], tok)
        ex.append({"ids": ids, "spans": spans, "labels": x.get("labels"), "first_error": x.get("first_error")})
        if limit and len(ex) >= limit:
            break
    return ex


def main():
    global LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--layers", default="last4", help="last4 (default), all, or comma-separated hidden-state indices")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training samples (smoke tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LAYERS = resolve_layers(a.layers, model.config.num_hidden_layers)
    print("layers", LAYERS, flush=True)
    t0 = time.time()
    tr = load(os.path.join(a.data, "train.jsonl"), tok, a.limit)
    dv = load(os.path.join(a.data, "dev.jsonl"), tok, a.limit // 10 if a.limit else 0)
    print(f"tokenized in {time.time() - t0:.0f}s: train {len(tr)} samples, dev {len(dv)} samples", flush=True)
    dim = model.config.hidden_size * len(LAYERS)
    probe = StepProbe(dim).cuda()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        t1, n, run = time.time(), 0, 0.0
        for bi, b in enumerate(batches(tr, TOK_BUDGET, True, a.seed + ep)):
            feats = forward_feats(model, tok, tr, b)
            s = torch.cat([probe(f, tr[i]["spans"]) for f, i in zip(feats, b)])
            y = torch.tensor([v for i in b for v in tr[i]["labels"]], dtype=torch.float32, device="cuda")
            loss = bce(s, y); opt.zero_grad(); loss.backward(); opt.step()
            n += len(b); run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
            if bi % 200 == 0:
                print(f"epoch {ep} batch {bi} samples {n}/{len(tr)} loss {run:.4f} {time.time() - t1:.0f}s", flush=True)
    probs = [None] * len(dv)
    with torch.no_grad():
        for b in batches(dv, TOK_BUDGET, False, 0):
            for f, i in zip(forward_feats(model, tok, dv, b), b):
                probs[i] = torch.sigmoid(probe(f, dv[i]["spans"])).tolist()
    golds = [e["first_error"] for e in dv]
    best = None
    for tau in [round(0.05 * k, 2) for k in range(1, 20)]:
        f1, acc_e, acc_c = f1_first_error([first_error_pred(p, tau) for p in probs], golds)
        if best is None or f1 > best[1]:
            best = (tau, f1, acc_e, acc_c)
    print(f"dev: tau={best[0]} first-error F1 {100 * best[1]:.1f} (error acc {100 * best[2]:.1f}, correct acc {100 * best[3]:.1f})", flush=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in probe.state_dict().items()}, os.path.join(a.out, "probe.safetensors"))
    json.dump({"base_model": a.model.rstrip("/").split("/")[-1], "layers": LAYERS, "feature_dim": dim,
               "head": "SWIFT gated linear: fused_layer = Linear(feature_dim, 2); gate = sigmoid(z0), reward = z1",
               "step_score": "gated mean of token rewards within each step span", "threshold": best[0],
               "step_separator": "\\n\\n", "chat_template": "model chat template, enable_thinking=False; user = problem, assistant = steps joined by \\n\\n",
               "training": {"data": "PRM800K phase1+phase2 train, ProcessBench-overlapping problems removed", "epochs": a.epochs,
                            "lr": 1e-4, "weight_decay": 1e-5, "seed": a.seed, "token_budget_per_batch": TOK_BUDGET, "loss": "step-level BCE"},
               "dev": {"f1": best[1], "acc_err": best[2], "acc_correct": best[3], "n": len(dv)}},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
