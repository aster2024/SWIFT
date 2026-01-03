import os
import json
from datasets import load_dataset


def process_samples(split_dataset):
    """
    Process the samples from the imbue/code-comprehension dataset.
    For each sample, create a new sample with:
      - "question": The original question field appended with "\n\nChoices: " and a comma-separated string of choices.
      - "answer": Taken from the correct_answer field.
    """
    processed_samples = []
    for item in split_dataset:
        original_question = item.get("question", "")
        choices = item.get("choices", [])
        question = f"{original_question}\n\nChoices: {', '.join(choices)}"
        answer = item.get("correct_answer", "")
        processed_samples.append({
            "question": question,
            "answer": answer
        })
    return processed_samples


def save_jsonl(data, filepath: str):
    """
    Save the list of dictionaries to a JSONL file.
    Each dictionary is saved as a separate JSON line.
    """
    with open(filepath, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    # Load the first 6500 instances from the train split of the imbue/code-comprehension dataset from Hugging Face.
    print("Loading the imbue/code-comprehension train split (first 6500 instances)...")
    dataset = load_dataset("imbue/code-comprehension", split="train[:6500]")

    output_dir = "data/imbue"
    os.makedirs(output_dir, exist_ok=True)

    print("Processing imbue dataset...")
    samples = process_samples(dataset)

    # Use the first 6000 examples for training and the remaining 500 for testing.
    train_samples = samples[:6000]
    test_samples = samples[6000:]
    train_output_path = os.path.join(output_dir, "train_imbue.jsonl")
    test_output_path = os.path.join(output_dir, "test_imbue.jsonl")

    save_jsonl(train_samples, train_output_path)
    print(f"Saved {len(train_samples)} training examples to {train_output_path}")

    save_jsonl(test_samples, test_output_path)
    print(f"Saved {len(test_samples)} test examples to {test_output_path}")


if __name__ == "__main__":
    main()
