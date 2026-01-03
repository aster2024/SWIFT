#!/usr/bin/env python
import argparse
import os
import sys
import time
import warnings
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def save_rewards(args):
    """
    Compute reward for each candidate sample using the trained reward model,
    then save the computed reward to an output file as a single aggregated value per sample.
    """
    instruction = get_instruction(args.dataset)

    print("\n========== Starting Reward Computation ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ds = load_data(args.dataset_file)
    print(f"Loaded {len(ds)} candidate outputs from {args.dataset_file}.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_lm.eval()

    sample0 = None
    for sample in ds:
        if len(sample.get("steps", [])) > 0:
            sample0 = sample
            break
    if sample0 is None:
        print("No samples with reasoning steps found; exiting.")
        return

    if args.logits_mode:
        detailed_info0 = extract_logits_info_for_reasoning_path(
            sample0["prompt"],
            instruction,
            sample0["steps"],
            args.separator,
            tokenizer,
            model_lm,
            to_cpu=False
        )
        feature_dim = detailed_info0["logits"].shape[-1]
        print(f"Reward model feature dimension (logits mode): {feature_dim}")
    else:
        detailed_info0 = extract_detailed_info_for_reasoning_path(
            sample0["prompt"],
            instruction,
            sample0["steps"],
            args.separator,
            args.layers,
            tokenizer,
            model_lm,
            apply_norm=args.apply_norm,
            to_cpu=False
        )
        hidden_states = detailed_info0["hidden_states"]
        sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
        feature_dim = 0
        for layer in sorted_layers:
            if hidden_states[layer] is not None:
                feature_dim += hidden_states[layer].shape[-1]
        print(f"Reward model feature dimension: {feature_dim}")

    base_model = LinearRewardModel(feature_dim, disable_gate=args.disable_gate).to(device)
    if args.use_dim_reduction:
        dim_reduction = DimReduction(feature_dim, args.dim_reduction_dim).to(device)
        reward_model = RewardModelWithDimReduction(base_model, dim_reduction).to(device)
    else:
        reward_model = base_model

    checkpoint = torch.load(args.reward_model_load, map_location=device)
    reward_model.load_state_dict(checkpoint)
    reward_model.eval()
    print(f"Loaded reward model from {args.reward_model_load}")

    num_params = sum(p.numel() for p in reward_model.parameters())
    print(f"Reward model parameter count: {num_params}")

    results = []
    total_time = 0.0

    rollout_counter = {}

    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc="Processing samples"):
        unique_id = sample.get("idx", 0)

        if args.max_rollouts is not None:
            current_count = rollout_counter.get(unique_id, 0)
            if current_count >= args.max_rollouts:
                continue
            rollout_counter[unique_id] = current_count + 1

        prompt = sample["prompt"]

        if args.logits_mode:
            detailed_info = extract_logits_info_for_reasoning_path(
                prompt,
                instruction,
                sample.get("steps", []),
                args.separator,
                tokenizer,
                model_lm,
                to_cpu=False
            )
        else:
            detailed_info = extract_detailed_info_for_reasoning_path(
                prompt,
                instruction,
                sample.get("steps", []),
                args.separator,
                args.layers,
                tokenizer,
                model_lm,
                apply_norm=args.apply_norm,
                to_cpu=False
            )

        token_features = get_token_features(detailed_info)
        if token_features is None:
            warnings.warn(f"Sample at index {idx} has no valid token features, skipping.")
            continue

        seq_len = token_features.size(0)
        token_features = token_features.unsqueeze(0).to(device)
        boundaries = detailed_info.get("boundaries", None)
        if boundaries is not None and len(boundaries) >= 2:
            step_boundaries = boundaries[1:]
        else:
            warnings.warn("No valid boundaries found, using the whole sequence as one step.")
            step_boundaries = [(0, seq_len)]

        result_entry = {
            "idx": sample.get("idx", idx),
            "prompt": prompt,
            "reference": sample.get("reference", ""),
            "correctness": int(sample.get("correctness", False))
        }
        start_time = time.perf_counter()
        reward_score = reward_model(token_features, [seq_len])
        result_entry["reward"] = reward_score.item()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        results.append(result_entry)
    print(f"Total reward computation time: {total_time:.4f} seconds")

    # Save computed reward results into the output file.
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reward results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained reward model and compute rewards for candidate samples, then save the reward to a file."
    )
    # Model and dataset arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path (e.g., 'gpt2').")
    parser.add_argument("--dataset_file", type=str, default=None,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--dataset", type=str, default="math",
                        help="Dataset name (default: math).")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join reasoning steps (default: two newlines).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer indices to extract; if not provided, extract all layers.")
    # Reward model arguments.
    parser.add_argument("--method", type=str, choices=["ce", "hinge", "dpo", "infonca", "nca"],
                        default=None,
                        help="Reward method to evaluate. If set, reward model checkpoints are automatically loaded from the default directory.")
    parser.add_argument("--reward_model_load", type=str, default=None,
                        help="Path to a saved reward model checkpoint.")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism in the reward model.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states (or logits) before reward computation.")
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="Add a dimension reduction layer before the reward model, if set.")
    parser.add_argument("--dim_reduction_dim", type=int, default=128,
                        help="Target dimension for dimension reduction (default: 128).")
    parser.add_argument("--max_rollouts", type=int, default=None,
                        help="If specified, process only the first max_rollouts samples for each unique idx.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output JSON file to save computed rewards.")
    parser.add_argument("--logits_mode", action="store_true",
                        help="If set, use the final output logits instead of hidden states. This is useful when hidden states are unavailable (e.g., for closed-source models).")
    args = parser.parse_args()

    if args.dataset_file is None:
        args.dataset_file = f"data/{args.dataset}/test_{args.dataset}_{args.model_name.split('/')[-1]}.json"

    if args.method is not None:
        if args.reward_model_load:
            raise ValueError("Either --method or --reward_model_load must be provided, not both.")
        norm_part = "norm_" if args.apply_norm else ""
        file_name = f"model/{args.dataset}/reward_model_{args.method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
        args.reward_model_load = file_name
    else:
        if args.reward_model_load is None:
            raise ValueError("Either --method or --reward_model_load must be provided.")

    if args.output_file is None:
        if args.method:
            args.output_file = os.path.splitext(args.dataset_file)[0] + f"_extracted_rewards_{args.method}.json"
        else:
            args.output_file = os.path.splitext(args.dataset_file)[0] + "_extracted_rewards.json"

    save_rewards(args)


if __name__ == "__main__":
    main()
