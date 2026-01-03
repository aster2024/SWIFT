#!/usr/bin/env python
import argparse
import json
import os
from datasets import load_dataset


def main(args):
    dataset = load_dataset("skrishna/coin_flip")
    train_data = dataset["train"]
    valid_data = dataset["validation"]

    os.makedirs("data/coinflip", exist_ok=True)

    train_file = os.path.join("data/coinflip", "train_coinflip.jsonl")
    with open(train_file, "w", encoding="utf-8") as fout:
        for example in train_data.select(range(6000)):
            question = example.get("inputs", "")
            answer = example.get("targets", "").strip()
            answer = "Yes" if answer == "yes" else "No"
            record = {"question": question, "answer": answer}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    test_file = os.path.join("data/coinflip", "test_coinflip.jsonl")
    with open(test_file, "w", encoding="utf-8") as fout:
        for example in valid_data.select(range(500)):
            question = example.get("inputs", "")
            answer = example.get("targets", "").strip()
            answer = "Yes" if answer == "yes" else "No"
            record = {"question": question, "answer": answer}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed train data (first 6000 examples of train) saved to: {train_file}")
    print(f"Processed test data (first 500 examples of validation) saved to: {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CoinFlip dataset to create train_coinflip.jsonl and test_coinflip.jsonl files."
    )

    args = parser.parse_args()
    main(args)
