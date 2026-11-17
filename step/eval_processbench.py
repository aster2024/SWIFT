"""Evaluate a trained step probe on ProcessBench (first-error identification).

Usage:
    python step/eval_processbench.py <model_dir> <probe_dir> <processbench_dir> <out.json>

The threshold is the one stored in <probe_dir>/config.json (selected on the PRM800K dev split); it is not tuned on ProcessBench.
Metric (official): for each subset, the harmonic mean of exact first-error accuracy on erroneous solutions and accuracy on
correct solutions; the reported score is the mean over gsm8k, math, olympiadbench and omnimath.
Also reported, as diagnostics only: a threshold sweep, and step-level AUC (probability that the first erroneous step
scores below a correct step), which does not depend on the threshold.
"""
import os, sys, json, bisect, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_features import build_example
import train_step_probe as T
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def score(probs_by_subset, golds_by_subset, tau):
    rows = {}
    for s in SUBSETS:
        f1, ae, ac = T.f1_first_error([T.first_error_pred(p, tau) for p in probs_by_subset[s]], golds_by_subset[s])
        rows[s] = {"f1": 100 * f1, "acc_err": 100 * ae, "acc_correct": 100 * ac, "n": len(golds_by_subset[s])}
    rows["avg_f1"] = sum(rows[s]["f1"] for s in SUBSETS) / len(SUBSETS)
    return rows


def step_auc(probs_by_subset, golds_by_subset):
    err, ok = [], []
    for s in SUBSETS:
        for p, g in zip(probs_by_subset[s], golds_by_subset[s]):
            if g == -1:
                ok += p
            else:
                ok += p[:g]; err.append(p[g])
    ok.sort()
    n = sum((len(ok) - bisect.bisect_right(ok, x)) + 0.5 * (bisect.bisect_right(ok, x) - bisect.bisect_left(ok, x)) for x in err)
    return n / (len(err) * len(ok))


def main():
    model_dir, probe_dir, pb_dir, out = sys.argv[1:5]
    cfg = json.load(open(os.path.join(probe_dir, "config.json")))
    tau = cfg["threshold"]; T.LAYERS = cfg["layers"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    probe = T.StepProbe(cfg["feature_dim"]).cuda()
    probe.load_state_dict(load_file(os.path.join(probe_dir, "probe.safetensors"))); probe.eval()
    probs, golds, ids = {}, {}, {}
    for s in SUBSETS:
        data = json.load(open(os.path.join(pb_dir, s + ".json")))
        ex = [dict(zip(("ids", "spans"), build_example(x["problem"], x["steps"], tok))) for x in data]
        pr = [None] * len(ex)
        with torch.no_grad():
            for b in T.batches(ex, T.TOK_BUDGET, False, 0):
                for f, i in zip(T.forward_feats(model, tok, ex, b), b):
                    pr[i] = torch.sigmoid(probe(f, ex[i]["spans"])).tolist()
        probs[s], golds[s], ids[s] = pr, [x["label"] for x in data], [x["id"] for x in data]
        print(s, "done", len(pr), flush=True)
    result = score(probs, golds, tau)
    sweep = {str(t): score(probs, golds, t)["avg_f1"] for t in [round(0.05 * k, 2) for k in range(1, 20)]}
    auc = step_auc(probs, golds)
    json.dump({"threshold": tau, "result": result, "step_auc": auc, "diagnostic_threshold_sweep_avg_f1": sweep,
               "per_item": {s: [{"id": i, "label": g, "step_probs": p} for i, g, p in zip(ids[s], golds[s], probs[s])] for s in SUBSETS}},
              open(out, "w"))
    print(f"tau={tau} (from dev)")
    for s in SUBSETS:
        r = result[s]
        print(f"  {s:14s} F1 {r['f1']:.1f} (error acc {r['acc_err']:.1f}, correct acc {r['acc_correct']:.1f}, n={r['n']})")
    print(f"  average F1 {result['avg_f1']:.1f}   step AUC {auc:.3f}")


if __name__ == "__main__":
    main()
