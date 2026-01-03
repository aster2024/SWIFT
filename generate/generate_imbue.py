import argparse
import re
import sys
import os
import json
import math
import ast
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
    3. Otherwise, try splitting by newline and check if each line starts with a number, symbol (e.g., -, *, •)
       or the word "step". If not, merge adjacent lines into a single step.
    4. Finally, as a fallback, split based on punctuation (.?!), followed by a space and a capital letter.
    """
    # Try to capture patterns like "Step 1:" using regex.
    pattern = re.compile(r'(?i)(?:^|\n)(step\s*\d+[\.\:]?\s+.*?)((?=\nstep\s*\d+[\.\:]?\s+)|$)', re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        steps = [m[0].strip() for m in matches]
        if len(steps) > 1:
            return steps

    # If the regex finds few matches, try splitting by newlines.
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


def format_prompt(problem: str):
    """
    Format the prompt for the model.
    """
    prompt = (
        "Analyze the following problem step-by-step. The question includes a list of choices. "
        "Select the most appropriate choice from the provided options and output your final answer enclosed within \\boxed{...},\n"
        "ensuring that the content inside \\boxed{...} is valid Python literal syntax.\n"
        + problem
    )
    return [{"role": "user", "content": prompt}]


def extract_boxed_content(text: str):
    """
    Extracts all candidate answers from text by finding occurrences of \\boxed{...}
    with support for nested curly braces.

    This function scans the text to locate the starting point of each \\boxed{ and then correctly identifies the corresponding
    closing brace, handling nested braces properly.

    Returns:
        A list of extracted candidate strings.
    """
    candidates = []
    pattern = r'\\boxed\s*\{'
    for match in re.finditer(pattern, text):
        start_index = match.end()  # Index right after the opening '{'
        brace_count = 1
        i = start_index
        # Iterate over the text while counting nested braces.
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            # i-1 is the position of the matching closing '}'
            candidate = text[start_index:i - 1].strip()
            candidates.append(candidate)
    return candidates


def clean_candidate_text(candidate: str) -> str:
    """
    Clean the candidate string by removing unwanted LaTeX commands and extra Markdown formatting.

    - Remove any occurrences of '\\text{...}' by replacing them with the inner text.
    - Replace escaped braces '\\{' and '\\}' with regular braces '{' and '}'.
    - Remove Markdown code fences if present.
    - Remove any extraneous whitespace.
    """
    # Remove \text{...} occurrences.
    candidate = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', candidate)
    # Replace escaped curly braces with unescaped ones.
    candidate = candidate.replace(r'\{', '{').replace(r'\}', '}')
    # Remove Markdown code fences if any.
    candidate = re.sub(r'```(?:python)?\s*(.*?)\s*```', r'\1', candidate, flags=re.DOTALL)
    return candidate.strip()


def remove_surrounding_quotes(s: str) -> str:
    """
    Remove surrounding single or double quotes from the string, if present.
    """
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def evaluate_imbue(model_answer: str, reference: str):
    """
    Evaluate the correctness of the provided answer by extracting the candidate final answer
    from the model's response.

    Extraction strategy:
      1. Search for the last occurrence of content within a properly nested \\boxed{...}.
         For instance, \\boxed{{0: None}} will yield {0: None}.
      2. If a boxed answer is not found or is empty, search for fallback patterns such as:
         "answer is", "answer:", "output is", or "output:" (case-insensitive) and extract the following content.
      3. If none of the above yield a candidate, use the entire model answer.

    The candidate answer and the reference answer are then cleaned (to remove unwanted LaTeX artifacts)
    and parsed using ast.literal_eval. If parsing fails, we attempt a fallback: treat the extracted content as a string,
    remove any surrounding quotes, and then compare that with the reference answer (also cleaned of surrounding quotes).

    Returns:
        tuple: (the original model answer, is_correct (bool), parsed candidate answer or fallback string)
    """
    candidate = None

    # First, try extracting with the improved nested braces function.
    boxed_candidates = extract_boxed_content(model_answer)
    if boxed_candidates:
        candidate = boxed_candidates[-1].strip()

    # If no valid boxed answer is found, use fallback regex patterns.
    if candidate is None or candidate == "":
        fallback_patterns = [
            r"answer\s*is\s*[:]*\s*(.*)",
            r"answer:\s*(.*)",
            r"output\s*is\s*[:]*\s*(.*)",
            r"output:\s*(.*)",
            r"result\s*is\s*[:]*\s*(.*)",
            r"result:\s*(.*)",
            r"result = \s*[:]*\s*(.*)",
            r"result=\s*(.*)",
        ]
        for pattern in fallback_patterns:
            matches = re.findall(pattern, model_answer, flags=re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip()
                break

    # If still not found, use the entire response.
    if candidate is None or candidate == "":
        candidate = model_answer.strip()

    # Clean the candidate text to remove unwanted LaTeX fragments.
    candidate = clean_candidate_text(candidate)

    try:
        parsed_model = ast.literal_eval(candidate)
        parsed_ref = ast.literal_eval(reference)
        is_correct = (parsed_model == parsed_ref)
        return (model_answer, is_correct, parsed_model)
    except Exception as e:
        # Literal eval failed, fallback: treat as raw strings.
        fallback_candidate = candidate.strip()
        fallback_reference = reference.strip()
        fallback_candidate = remove_surrounding_quotes(fallback_candidate)
        fallback_reference = remove_surrounding_quotes(fallback_reference)
        is_correct = (fallback_candidate == fallback_reference)
        return (model_answer, is_correct, fallback_candidate)


@torch.no_grad()
def main(args):
    # Set the default input file based on IMBUE dataset.
    if args.input_file is None:
        args.input_file = f"data/imbue/{args.split}_imbue.jsonl"

    # Set default rollouts (number of generations) per sample.
    if args.n_rollouts is None:
        if args.split == "train":
            args.n_rollouts = 8
        else:
            args.n_rollouts = 64

    set_seed(args.seed)
    print("Loading model with vLLM...")

    # Configure model engine parameters.
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
        problem = sample.get("question", "")
        formatted_prompt = format_prompt(problem)
        formatted_prompt = tokenizer.apply_chat_template(formatted_prompt, tokenize=False, add_generation_prompt=True)
        all_prompts.append(formatted_prompt)

    # If the number of rollouts exceeds batch_size, perform generation in multiple rounds.
    outputs = []
    if args.n_rollouts > args.batch_size:
        print(
            f"n_rollouts ({args.n_rollouts}) is greater than batch_size (max_num_seqs={args.batch_size}).\n"
            "Generating in multiple rounds for all instances simultaneously..."
        )
        rounds = math.ceil(args.n_rollouts / args.batch_size)
        outputs_all_rounds = [[] for _ in range(len(all_prompts))]
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

    # Process each sample and its corresponding completions.
    for idx, (sample, output) in enumerate(zip(samples, outputs)):
        problem = sample.get("question", "")
        reference = sample.get("answer", "")

        # Handle both list and instance.outputs formats.
        if isinstance(output, list):
            completions_objs = output
        else:
            completions_objs = output.outputs

        completions = [comp.text for comp in completions_objs]
        correctness = []
        steps_list = []

        for completion in completions:
            # Evaluate the generated answer using the updated evaluation method.
            _, is_correct, extracted_output = evaluate_imbue(completion, reference)
            total_count += 1
            if is_correct:
                correct_count += 1

            steps = split_steps(completion)

            if args.split == "train":
                correctness.append(is_correct)
                steps_list.append(steps)
            else:
                processed_sample = {
                    "task": "imbue",
                    "idx": idx,
                    "prompt": problem,
                    "response": completion,
                    "steps": steps,
                    "extracted_output": str(extracted_output),
                    "reference": reference,
                    "correctness": is_correct
                }
                output_samples.append(processed_sample)

        if args.split == "train":
            processed_sample = {
                "prompt": problem,
                "reference": reference,
                "dataset": "IMBUE",
                "completions": completions,
                "correctness": correctness,
                "steps": steps_list
            }
            output_samples.append(processed_sample)

    print(
        f"Total rollouts: {total_count}, Correct rollouts: {correct_count}, Accuracy: {correct_count / total_count:.4f}"
    )

    # Set the default output file based on the IMBUE dataset if not provided.
    if args.output_file is None:
        args.output_file = f"data/imbue/{args.split}_imbue_{args.model_name.split('/')[-1]}.json"

    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(output_samples, fout, ensure_ascii=False, indent=2)

    print(f"Processing complete! Output file saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reasoning steps for IMBUE dataset samples using an instruct LLM (with vLLM)"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSONL file containing IMBUE dataset samples")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to save processed samples")
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Split of the IMBUE dataset to process")
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
