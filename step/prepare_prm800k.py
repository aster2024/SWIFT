"""Convert PRM800K into step-level first-error data and remove problems that appear in ProcessBench.

Usage:
    python step/prepare_prm800k.py <prm800k_dir> <processbench_dir> <out_dir>

<prm800k_dir> holds phase1_train.jsonl and phase2_train.jsonl (https://github.com/openai/prm800k).
<processbench_dir> holds gsm8k.json, math.json, olympiadbench.json, omnimath.json (https://huggingface.co/datasets/Qwen/ProcessBench).

Rules:
- Skip quality-control and initial-screening questions, and records with finish_reason == "bad_problem".
- Drop every record whose normalized problem text equals a ProcessBench problem (all four subsets).
- Follow the completion actually used at each step (chosen_completion, else human_completion; for the final
  step with no choice, the single rated completion). Ratings +1/0 -> correct (1), -1 -> incorrect (0).
  Stop at the first -1 (inclusive) or at the first unrated step.
- Each rated-but-unchosen completion with rating -1 yields an extra sample: (prefix so far) + that step, labeled 0.
  In phase 1 the chosen path almost never contains a -1, so these are the main source of negatives there.
- Dev split: 1/20 of problems by a hash of the normalized problem; all samples of a problem stay on one side.
Outputs train.jsonl, dev.jsonl ({problem, steps, labels, first_error, source}) and stats.json.
"""
import sys, os, json, re, hashlib, unicodedata, collections


def norm(s):
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", s).lower())


def main():
    src, pb_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    pb = set()
    for f in ("gsm8k", "math", "olympiadbench", "omnimath"):
        pb |= {norm(x["problem"]) for x in json.load(open(os.path.join(pb_dir, f + ".json")))}
    print("ProcessBench problems (normalized):", len(pb))
    st, rows = collections.Counter(), []
    for fn in ("phase1_train.jsonl", "phase2_train.jsonl"):
        tag = fn.split("_")[0]
        for line in open(os.path.join(src, fn), encoding="utf-8"):
            r = json.loads(line); st["read"] += 1
            if r.get("is_quality_control_question") or r.get("is_initial_screening_question"):
                st["skip_qc_or_screening"] += 1; continue
            if r["label"].get("finish_reason") == "bad_problem":
                st["skip_bad_problem"] += 1; continue
            q = r["question"]["problem"]
            if norm(q) in pb:
                st["drop_processbench_overlap"] += 1; continue
            steps, labels, extra = [], [], []
            for s in r["label"]["steps"]:
                comps = s.get("completions") or []
                c = s.get("chosen_completion")
                comp = comps[c] if (c is not None and c < len(comps)) else (s.get("human_completion") or None)
                if comp is None:
                    rated = [x for x in comps if x.get("rating") is not None]
                    comp = rated[0] if len(rated) == 1 else None
                for alt in comps:
                    if alt is not comp and alt.get("rating") == -1 and alt.get("text"):
                        extra.append({"problem": q, "steps": steps + [alt["text"]], "labels": labels + [0],
                                      "source": tag + "_alt", "first_error": len(steps)})
                if comp is None or comp.get("rating") is None:
                    break
                steps.append(comp["text"]); labels.append(1 if comp["rating"] >= 0 else 0)
                if comp["rating"] == -1:
                    break
            rows.extend(extra); st["negatives_from_unchosen"] += len(extra)
            if not steps:
                st["no_usable_steps"] += 1; continue
            st["kept"] += 1
            rows.append({"problem": q, "steps": steps, "labels": labels, "source": tag,
                         "first_error": labels.index(0) if 0 in labels else -1})
    dev_q = {norm(x["problem"]) for x in rows if int(hashlib.sha256(norm(x["problem"]).encode()).hexdigest(), 16) % 20 == 0}
    dev = [x for x in rows if norm(x["problem"]) in dev_q]
    train = [x for x in rows if norm(x["problem"]) not in dev_q]
    for name, part in (("train", train), ("dev", dev)):
        with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats = dict(st)
    stats.update(train=len(train), dev=len(dev), train_problems=len({norm(x["problem"]) for x in train}), dev_problems=len(dev_q),
                 train_steps=sum(len(x["steps"]) for x in train), train_negative_steps=sum(x["labels"].count(0) for x in train))
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
