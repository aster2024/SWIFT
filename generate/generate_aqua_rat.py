import argparse
import re
import sys
import os
import json
import math
from vllm import LLM, SamplingParams
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from generate.generate_utils import *
from utils import set_seed


def split_steps(text: str):
    """
    Split the reasoning steps using multiple strategies.

    1. Try splitting on "Step X:" patterns.
    2. If not, split by lines with numbering or bullets.
    3. Finally, split on punctuation (.?!), followed by a space and a capital letter.
    """
    pattern = re.compile(
        r'(?i)(?:^|\n)(step\s*\d+[\.\:]?\s+.*?)'
        r'((?=\nstep\s*\d+[\.\:]?\s+)|$)',
        re.DOTALL
    )
    matches = pattern.findall(text)
    if matches:
        steps = [m[0].strip() for m in matches]
        if len(steps) > 1:
            return steps

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) > 1:
        steps = []
        current = ""
        for line in lines:
            if re.match(r"^(step\s*\d+[\.\:]|[\d\-\*\u2022])", line, re.IGNORECASE):
                if current:
                    steps.append(current.strip())
                current = line
            else:
                current += " " + line
        if current:
            steps.append(current.strip())
        if len(steps) > 1:
            return steps

    # Fallback split
    steps = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in steps if s.strip()]


def format_prompt(question: str):
    """
    Build the chat-format prompt for AQUA‑RAT multiple choice.
    We instruct the model to reason step by step, then give the final answer
    as \\boxed{Letter} where Letter is one of A,B,C,D,E.
    """
    instruction = (
        "You are given a multiple-choice question with five options (A–E).\n"
        "Solve it step by step, then present only one letter (A-E) in the form \\boxed{Letter}.\n"
        "Remember to output \\boxed{Letter} at the end of your answer or it will be considered incorrect.\n"
    )
    full = instruction + question
    return [{"role": "user", "content": full}]


@torch.no_grad()
def main(args):
    if args.input_file is None:
        args.input_file = f"data/aqua_rat/{args.split}_aqua_rat.jsonl"

    if args.n_rollouts is None:
        args.n_rollouts = 8 if args.split == "train" else 64

    set_seed(args.seed)
    print("Loading model with vLLM...")

    engine_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_num_seqs": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.awq:
        engine_kwargs["quantization"] = "AWQ"

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    with open(args.input_file, "r", encoding="utf-8") as fin:
        samples = [json.loads(line) for line in fin]
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_prompts = []
    for sample in samples:
        q = sample["question"]
        prom = format_prompt(q)
        prom = tokenizer.apply_chat_template(prom, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prom)

    if args.n_rollouts > args.batch_size:
        print(
            f"n_rollouts ({args.n_rollouts}) is greater than batch_size (max_num_seqs={args.batch_size}).\n"
            "Generating in multiple rounds for all instances simultaneously..."
        )
        rounds = math.ceil(args.n_rollouts / args.batch_size)
        outputs_all = [[] for _ in range(len(all_prompts))]
        for r in range(rounds):
            cur_n = min(args.n_rollouts - r * args.batch_size, args.batch_size)
            sp = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=cur_n,
                stop=["\n###\nProblem: "],
            )
            round_out = llm.generate(all_prompts, sp, use_tqdm=True)
            for i, inst in enumerate(round_out):
                outputs_all[i].extend(inst.outputs)
        outputs = outputs_all
    else:
        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.n_rollouts,
            stop=["\n###\nProblem: "],
        )
        outputs = llm.generate(all_prompts, sp, use_tqdm=True)

    total = 0
    correct = 0
    output_samples = []

    for idx, (sample, out) in enumerate(zip(samples, outputs)):
        question = sample["question"]
        reference = sample["answer"].strip()
        comps = [c.text for c in (out.outputs if hasattr(out, "outputs") else out)]

        correctness_list = []
        steps_list = []

        for comp in comps:
            _, is_corr, extracted = evaluate_math(comp, reference)
            total += 1
            if is_corr:
                correct += 1

            steps = split_steps(comp)

            if args.split == "train":
                correctness_list.append(is_corr)
                steps_list.append(steps)
            else:
                output_samples.append({
                    "task": "aqua-rat",
                    "idx": idx,
                    "prompt": question,
                    "response": comp,
                    "steps": steps,
                    "extracted_output": extracted,
                    "reference": reference,
                    "correctness": is_corr
                })

        if args.split == "train":
            output_samples.append({
                "prompt": question,
                "reference": reference,
                "dataset": "AQUA-RAT",
                "completions": comps,
                "correctness": correctness_list,
                "steps": steps_list
            })

    acc = correct / total if total > 0 else 0.0
    print(f"Total rollouts: {total}, Correct: {correct}, Accuracy: {acc:.4f}")

    if args.output_file is None:
        model_tag = args.model_name.split("/")[-1]
        args.output_file = f"data/aqua_rat/{args.split}_aqua_rat_{model_tag}.json"

    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(output_samples, fout, ensure_ascii=False, indent=2)

    print(f"Finished. Output saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate step-by-step reasoning for AQUA-RAT multiple-choice questions using vLLM"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to AQUA-RAT JSONL input file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the JSON output")
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Which split to process (train or test)")
    parser.add_argument("--n_rollouts", type=int, default=None,
                        help="Number of rollouts per sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max total model context length")
    parser.add_argument("--model_name", type=str, required=True,
                        help="vLLM model identifier")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="vLLM max_num_seqs / batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Truncate the dataset to this many samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--awq", action="store_true",
                        help="Use AWQ quantization")
    args = parser.parse_args()
    main(args)
