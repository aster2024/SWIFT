#!/usr/bin/env python
import argparse
import json
import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import load_data


def aggregate_external_reward(sample):
    """
    Aggregate external reward for a candidate sample using orm mode.
    Use sample['score'] (a single number).
    """
    if "score" not in sample:
        warnings.warn("Candidate missing 'score' field for external reward (orm).")
        return None
    return sample["score"]


def aggregate_our_reward(sample):
    """
    Aggregate our computed reward for a candidate sample using orm mode.
    Use sample['reward'] (a single number).
    """
    if "reward" not in sample:
        warnings.warn("Candidate missing 'reward' field for our computed reward (orm).")
        return None
    return sample["reward"]


def group_by_prompt(candidates):
    """
    Group candidates by their prompt index.
    Assumes each candidate has an 'idx' field that indicates the prompt group.
    """
    groups = {}
    for cand in candidates:
        key = cand.get("idx", None)
        if key is None:
            warnings.warn("Candidate missing 'idx' key for grouping.")
            continue
        groups.setdefault(key, []).append(cand)
    return groups


def compute_pass_at_k(groups, k_vals, integration_method):
    """
    Compute pass@k metric for a given integration method.
    For each prompt group, among the top-k candidates (according to the integrated metric),
    choose the best candidate. If that candidate is correct, the group is considered passed.
    """
    metrics = {}
    for k in k_vals:
        correct_count = 0
        total = 0
        for whole_group in groups.values():
            if not whole_group:
                continue

            group = whole_group[:k].copy()  # Use first k candidates.

            for candidate in group:
                own = candidate.get("our_reward", None)
                if own is None:
                    raise ValueError("Candidate missing 'our_reward' field.")

                ext = candidate.get("ext_reward", None)
                if ext is None and integration_method != "our_only":
                    raise ValueError("Candidate missing 'ext_reward' field.")

            if integration_method == "our_only":
                sorted_candidates = sorted(group, key=lambda x: x["our_reward"], reverse=True)
                if sorted_candidates[0].get("correctness", 0):
                    correct_count += 1

            elif integration_method == "ext_only":
                sorted_candidates = sorted(group, key=lambda x: x["ext_reward"], reverse=True)
                if sorted_candidates[0].get("correctness", 0):
                    correct_count += 1

            elif integration_method == "ranking":
                # Compute ranking for our_reward and ext_reward separately.
                sorted_by_our = sorted(group, key=lambda x: x["our_reward"], reverse=True)
                our_ranks = {id(c): rank for rank, c in enumerate(sorted_by_our)}
                sorted_by_ext = sorted(group, key=lambda x: x["ext_reward"], reverse=True)
                ext_ranks = {id(c): rank for rank, c in enumerate(sorted_by_ext)}
                # Combined ranking: the worse (i.e. larger rank) of the two.
                for candidate in group:
                    candidate["combined_rank"] = max(
                        our_ranks.get(id(candidate), float('inf')),
                        ext_ranks.get(id(candidate), float('inf'))
                    )
                sorted_candidates = sorted(group, key=lambda x: x["combined_rank"])
                if sorted_candidates[0].get("correctness", 0):
                    correct_count += 1

            elif integration_method == "scaling":
                # Scale our_reward and ext_reward to [0,1] and take their average.
                our_values = [c["our_reward"] for c in group]
                ext_values = [c["ext_reward"] for c in group]
                min_our, max_our = min(our_values), max(our_values)
                min_ext, max_ext = min(ext_values), max(ext_values)
                for candidate in group:
                    if max_our - min_our != 0:
                        norm_our = (candidate["our_reward"] - min_our) / (max_our - min_our)
                    else:
                        norm_our = 1.0
                    if max_ext - min_ext != 0:
                        norm_ext = (candidate["ext_reward"] - min_ext) / (max_ext - min_ext)
                    else:
                        norm_ext = 1.0
                    candidate["integrated_score"] = (norm_our + norm_ext) / 2.0
                sorted_candidates = sorted(group, key=lambda x: x["integrated_score"], reverse=True)
                if sorted_candidates[0].get("correctness", 0):
                    correct_count += 1

            else:
                raise ValueError(f"Invalid integration method: {integration_method}")
            total += 1
        metrics[k] = round((correct_count / total) * 100, 1) if total > 0 else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute integration metrics (pass@k) using external reward with our reward model outputs and the original dataset."
    )
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the original dataset JSON file containing candidate outputs and external rewards (score).")
    parser.add_argument("--reward_file", type=str, required=True,
                        help="Path to the JSON file containing our computed reward model outputs.")
    parser.add_argument("--k_vals", type=int, nargs="+", default=[1, 4, 16, 64],
                        help="List of candidate numbers (k) for pass@k evaluation.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the computed integration metrics.")
    parser.add_argument("--no_aggregate", action="store_true",
                        help="Do not use external reward scores. Evaluate only our reward model outputs.")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = "res/" + os.path.basename(args.dataset_file).replace(".json", "_integration_metrics_orm.json")

    dataset = load_data(args.dataset_file)
    rewards = load_data(args.reward_file)
    assert len(dataset) == len(rewards), "Dataset and reward file have different lengths."

    merged_candidates = []
    for sample, reward_entry in zip(dataset, rewards):
        merged = sample.copy()
        if not args.no_aggregate:
            merged["ext_reward"] = aggregate_external_reward(sample)
        merged["our_reward"] = aggregate_our_reward(reward_entry)
        merged_candidates.append(merged)

    groups = group_by_prompt(merged_candidates)

    integration_methods = ["our_only", "ext_only", "ranking", "scaling"] if not args.no_aggregate else ["our_only"]
    all_metrics = {}
    for method in integration_methods:
        metrics = compute_pass_at_k(groups, args.k_vals, method)
        all_metrics[method] = metrics
        print(f"\nIntegration method [{method}] BoN@k metrics:")
        for k, val in metrics.items():
            print(f"BoN@{k}: {val}%")

    with open(args.output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {args.output_file}")


if __name__ == "__main__":
    main()
