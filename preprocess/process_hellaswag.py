import os
import json
from datasets import load_dataset

RANDOM_SEED = 42

def process_split(dataset_split, max_samples=None):
    """
    Process a HellaSWAG dataset split.
    For each record, build the question string by:
      1) Taking the 'ctx' field
      2) Adding a newline
      3) Appending each ending on its own line, prefixed with "A) ", "B) ", etc.
    The answer is derived from the integer 'label' (0-3), mapped to "A"-"D".

    If max_samples is provided, only the first max_samples examples are processed.
    """
    label_map = ["A", "B", "C", "D"]
    processed = []
    for idx, item in enumerate(dataset_split):
        if max_samples is not None and idx >= max_samples:
            break

        ctx = item.get("ctx", "").strip()
        endings = item.get("endings", [])
        label = int(item.get("label", None))

        # Build the question: context + each ending on a new line with A)-D)
        options_lines = []
        for opt_letter, ending in zip(label_map, endings):
            # Ensure no stray whitespace/newlines
            ending_text = ending.strip()
            options_lines.append(f"{opt_letter}) {ending_text}")
        question_text = ctx + "\n" + "\n".join(options_lines)

        # Map numeric label to letter
        answer = label_map[label] if (label is not None and 0 <= label < len(label_map)) else ""

        processed.append({
            "question": question_text,
            "answer": answer
        })

    return processed

def save_jsonl(data, filepath: str):
    """
    Save a list of dicts to a JSONL file (one JSON object per line).
    """
    with open(filepath, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    print("Loading the HellaSWAG dataset (Rowan/hellaswag) from Hugging Face...")
    dataset = load_dataset("Rowan/hellaswag")

    output_dir = "data/hellaswag"
    os.makedirs(output_dir, exist_ok=True)

    print("Shuffling and selecting 6000 training examples...")
    train_shuffled = dataset["train"].shuffle(seed=RANDOM_SEED)
    train_samples = process_split(train_shuffled, max_samples=6000)

    train_path = os.path.join(output_dir, "train_hellaswag.jsonl")
    save_jsonl(train_samples, train_path)
    print(f"Saved {len(train_samples)} training examples to {train_path}")

    print("Shuffling and selecting 500 test examples...")
    test_shuffled = dataset["validation"].shuffle(seed=RANDOM_SEED)
    test_samples = process_split(test_shuffled, max_samples=500)

    test_path = os.path.join(output_dir, "test_hellaswag.jsonl")
    save_jsonl(test_samples, test_path)
    print(f"Saved {len(test_samples)} test examples to {test_path}")

if __name__ == "__main__":
    main()
