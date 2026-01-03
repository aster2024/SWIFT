import os
import json
from datasets import load_dataset

def process_split(dataset_split, max_samples=None):
    """
    Process a dataset split. For each record, build the question string by
    concatenating the question, a newline, and the options joined by spaces.
    The answer is taken directly from the 'correct' field.

    If max_samples is provided, only the first max_samples examples are processed.
    """
    processed = []
    for idx, item in enumerate(dataset_split):
        if max_samples is not None and idx >= max_samples:
            break
        question = item.get("question", "")
        options = item.get("options", [])
        # Build the full question text
        question_text = question + "\n" + " ".join(options)
        answer = item.get("correct", "").strip()
        processed.append({
            "question": question_text,
            "answer": answer
        })
    return processed

def save_jsonl(data, filepath: str):
    """
    Save a list of dicts to a JSONL file.
    """
    with open(filepath, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    # Load the 'raw' subset of deepmind/aqua_rat
    print("Loading the AQUA‑RAT 'raw' subset from Hugging Face...")
    dataset = load_dataset("deepmind/aqua_rat", "raw")

    output_dir = "data/aqua_rat"
    os.makedirs(output_dir, exist_ok=True)

    # Process the train split
    print("Processing training split...")
    train_samples = process_split(dataset["train"])
    train_path = os.path.join(output_dir, "train_aqua_rat.jsonl")
    save_jsonl(train_samples, train_path)
    print(f"Saved {len(train_samples)} training examples to {train_path}")

    # Merge validation and test, then take first 500
    print("Merging validation and test splits; selecting first 500 examples...")
    # Materialize splits into lists for easy concatenation
    val_list = list(dataset["validation"])
    test_list = list(dataset["test"])
    merged = val_list + test_list

    test_samples = process_split(merged, max_samples=500)
    test_path = os.path.join(output_dir, "test_aqua_rat.jsonl")
    save_jsonl(test_samples, test_path)
    print(f"Saved {len(test_samples)} test examples to {test_path}")


if __name__ == "__main__":
    main()