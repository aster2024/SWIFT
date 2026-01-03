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

    1. First, try to split based on the "Step X:" or "step X:" format (supports colon or period).
    2. If multiple steps are found, return the list immediately.
    3. Otherwise, try splitting by newline and check if each line starts with a number, symbol (e.g., -, *, •) or the word "step".
       If not, merge adjacent lines into a single step.
    4. Finally, as a fallback, split based on punctuation (.?!), followed by a space and a capital letter.
    """
    # Try to capture patterns like "Step 1:" using regex
    pattern = re.compile(r'(?i)(?:^|\n)(step\s*\d+[\.\:]?\s+.*?)((?=\nstep\s*\d+[\.\:]?\s+)|$)', re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        steps = [m[0].strip() for m in matches]
        if len(steps) > 1:
            return steps

    # If the regex finds few matches, try splitting by newlines
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if len(lines) > 1:
        steps = []
        current_step = ""
        for line in lines:
            if re.match(r"^(step\s*\d+[\.\:]|[\d\-\*\u2022])", line, re.IGNORECASE):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        if current_step:
            steps.append(current_step.strip())
        if len(steps) > 1:
            return steps

    # Fallback: split based on punctuation (.?!), followed by a space and a capital letter.
    steps = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [step.strip() for step in steps if step.strip()]


def format_prompt(problem):
    """
    Format the prompt for the model.
    The prompt instructs the model to solve the math problem step-by-step,
    and to present the final answer as \\boxed{Your Answer}.
    """
    prompt = (
        "Solve the following math problem step-by-step.\n"
        "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        + problem
    )
    return [{"role": "user", "content": prompt}, ]


@torch.no_grad()
def main(args):
    if args.input_file is None:
        args.input_file = f"data/math/{args.split}_math.jsonl"

    if args.n_rollouts is None:
        if args.split == "train":
            args.n_rollouts = 8
        else:
            args.n_rollouts = 64

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

    output_samples = []

    with open(args.input_file, "r", encoding="utf-8") as fin:
        samples = [json.loads(line) for line in fin]

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_prompts = []
    for idx, sample in enumerate(samples):
        problem = sample.get("problem", "")
        formatted_prompt = format_prompt(problem)
        formatted_prompt = tokenizer.apply_chat_template(formatted_prompt, tokenize=False, add_generation_prompt=True)
        all_prompts.append(formatted_prompt)

    # Check if n_rollouts exceeds the maximum sequences (batch_size) allowed by the engine.
    # If so, perform multi-round generation for all instances in one batch per round.
    outputs = []
    if args.n_rollouts > args.batch_size:
        print(
            f"n_rollouts ({args.n_rollouts}) is greater than batch_size (max_num_seqs={args.batch_size}).\n"
            "Generating in multiple rounds for all instances simultaneously..."
        )
        rounds = math.ceil(args.n_rollouts / args.batch_size)

        outputs_all_rounds = [ [] for _ in range(len(all_prompts)) ]
        for r in range(rounds):
            current_n = min(args.n_rollouts - r * args.batch_size, args.batch_size)
            round_sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=current_n,
                stop=["\n###\nProblem: "],
            )
            round_outputs = llm.generate(all_prompts, round_sampling_params, use_tqdm=True)
            for i, instance in enumerate(round_outputs):
                outputs_all_rounds[i].extend(instance.outputs)
        outputs = outputs_all_rounds
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.n_rollouts,
            stop=["\n###\nProblem: "],
        )
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    total_count = 0
    correct_count = 0

    for idx, (sample, output) in enumerate(zip(samples, outputs)):
        problem = sample.get("problem", "")
        reference = sample.get("answer", "")

        if isinstance(output, list):
            completions_objs = output
        else:
            completions_objs = output.outputs

        completions = [comp.text for comp in completions_objs]
        correctness = []
        steps_list = []

        for completion in completions:
            _, is_correct, extracted_output = evaluate_math(completion, reference)
            total_count += 1
            if is_correct:
                correct_count += 1

            steps = split_steps(completion)

            if args.split == "train":
                correctness.append(is_correct)
                steps_list.append(steps)
            else:
                processed_sample = {
                    "task": "math",
                    "idx": idx,
                    "prompt": problem,
                    "response": completion,
                    "steps": steps,
                    "extracted_output": extracted_output,
                    "reference": reference,
                    "correctness": is_correct
                }
                output_samples.append(processed_sample)

        if args.split == "train":
            processed_sample = {
                "prompt": problem,
                "reference": reference,
                "dataset": "MATH",
                "completions": completions,
                "correctness": correctness,
                "steps": steps_list
            }
            output_samples.append(processed_sample)

    print(f"Total rollouts: {total_count}, Correct rollouts: {correct_count}, Accuracy: {correct_count / total_count:.4f}")

    if args.output_file is None:
        args.output_file = f"data/math/{args.split}_math_{args.model_name.split('/')[-1]}.json"

    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(output_samples, fout, ensure_ascii=False, indent=2)

    print(f"Processing complete! Output file saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reasoning steps for MATH dataset samples using an instruct LLM (with vLLM)"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSONL file containing MATH dataset samples")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to save processed samples")
    parser.add_argument("--split", choices=["train", "test"], required=True, help="Split of the MATH dataset to process")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of rollouts to generate for each sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling ratio")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model length (input + output)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the instruct model to use")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use for vLLM")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (also used as max_num_seqs for vLLM)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--awq", action="store_true", help="Use AWQ quantization")

    args = parser.parse_args()
    main(args)
