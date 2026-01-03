import os
import re
import json
from datasets import load_dataset


def extract_answer(answer_str: str) -> str:
    """
    Extract the final answer from the answer string.
    The answer string is expected to contain a pattern like:

      ... #### 72

    This function extracts and returns '72' (after the ####).
    If the pattern is not found, it returns the original string.
    """
    match = re.search(r"####\s*(.*)", answer_str)
    if match:
        return match.group(1).strip()
    return answer_str.strip()


def process_split(split_dataset, is_test: bool = False):
    """
    Process a dataset split (train or test). For each record, extract
    the question and process the answer to keep only the content after '####'.

    If is_test is True then only the first 500 examples are processed.
    """
    processed_samples = []
    count = 0
    for item in split_dataset:
        # Retrieve 'question' field as the problem (fallback to empty string).
        question = item.get("question", "")
        # Retrieve the raw answer and process it.
        answer_raw = item.get("answer", "")
        answer = extract_answer(answer_raw)
        processed_samples.append({
            "question": question,
            "answer": answer
        })
        count += 1
        if is_test and count >= 500:
            break
    return processed_samples


def save_jsonl(data, filepath: str):
    """
    Save the list of dictionaries to a jsonl file.
    Each dictionary is saved as a JSON on a separate line.
    """
    with open(filepath, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    # Load the 'main' subset of openai/gsm8k dataset from Hugging Face.
    print("Loading the GSM8K 'main' subset from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main")

    output_dir = "data/gsm8k"
    os.makedirs(output_dir, exist_ok=True)

    # Process the train split: process all examples.
    print("Processing training split...")
    train_samples = process_split(dataset["train"], is_test=False)
    train_output_path = os.path.join(output_dir, "train_gsm8k.jsonl")
    save_jsonl(train_samples, train_output_path)
    print(f"Saved {len(train_samples)} training examples to {train_output_path}")

    # Process the test split: only use the first 500 examples.
    print("Processing test split (first 500 examples)...")
    test_samples = process_split(dataset["test"], is_test=True)
    test_output_path = os.path.join(output_dir, "test_gsm8k.jsonl")
    save_jsonl(test_samples, test_output_path)
    print(f"Saved {len(test_samples)} test examples to {test_output_path}")


if __name__ == "__main__":
    main()
