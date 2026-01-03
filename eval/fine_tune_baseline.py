import argparse
import os
import json
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import sys

# Ensure the utility functions from the parent directory can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class PairwiseRewardTrainDataset(Dataset):
    """
    Dataset for training a reward model with pairwise preference data.
    This is modified to align with the training approach from the Eurus paper.

    Reads each JSONL line where the sample contains:
        "prompt", "completions", "correctness"

    For each prompt, it identifies correct (chosen) and incorrect (rejected) completions.
    It then creates pairs by matching each correct completion with a randomly selected
    incorrect completion.
    """

    def __init__(self, file_path, tokenizer, max_length=2048, instruction="", max_train_samples=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        data = load_data(file_path)

        # Process data to create pairwise samples
        num_processed_samples = 0
        for sample in data:
            if max_train_samples and num_processed_samples >= max_train_samples:
                break

            prompt = sample.get("prompt", "")
            completions = sample.get("completions", [])
            correctness_list = sample.get("correctness", [])

            # Separate completions into correct (chosen) and incorrect (rejected) lists
            correct_completions = [comp for comp, corr in zip(completions, correctness_list) if corr]
            incorrect_completions = [comp for comp, corr in zip(completions, correctness_list) if not corr]

            # If there are both correct and incorrect completions, create pairs
            if correct_completions and incorrect_completions:
                for chosen_comp in correct_completions:
                    # Pair each chosen completion with a randomly selected rejected completion
                    rejected_comp = random.choice(incorrect_completions)

                    # Format the input text for both chosen and rejected responses
                    chosen_text = f"[INST] {instruction + prompt} [/INST] {chosen_comp}"
                    rejected_text = f"[INST] {instruction + prompt} [/INST] {rejected_comp}"

                    self.samples.append({"chosen_text": chosen_text, "rejected_text": rejected_text})

            num_processed_samples += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Tokenize both the chosen and rejected texts
        tokenized_chosen = self.tokenizer(
            item["chosen_text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized_rejected = self.tokenizer(
            item["rejected_text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": tokenized_chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": tokenized_chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": tokenized_rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": tokenized_rejected["attention_mask"].squeeze(0)
        }


class RewardEvalDataset(Dataset):
    """
    Each JSONL line should have keys:
        "task", "idx", "prompt", "response", "steps", "extracted_output",
        "reference", "correctness"
    The input text is formulated as:
        "[INST] {prompt} [/INST] {response}"
    """

    def __init__(self, file_path, tokenizer, max_length=2048, instruction="", max_eval_samples=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        data = load_data(file_path)
        for i, sample in enumerate(data):
            if max_eval_samples and i >= max_eval_samples:
                break

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        input_text = f"[INST] {self.instruction + prompt} [/INST] {response}"
        tokenized = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Save tokenized output in the sample for use during evaluation.
        sample["tokenized"] = {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }
        return sample


def pairwise_train_collate_fn(batch):
    """
    Pads variable-length sequences for pairwise training.
    It handles both chosen and rejected inputs separately.
    """
    # Create lists for each part of the paired data
    chosen_ids_list = [item["chosen_input_ids"] for item in batch]
    chosen_mask_list = [item["chosen_attention_mask"] for item in batch]
    rejected_ids_list = [item["rejected_input_ids"] for item in batch]
    rejected_mask_list = [item["rejected_attention_mask"] for item in batch]

    # Pad each list of tensors
    chosen_ids_padded = torch.nn.utils.rnn.pad_sequence(chosen_ids_list, batch_first=True, padding_value=0)
    chosen_mask_padded = torch.nn.utils.rnn.pad_sequence(chosen_mask_list, batch_first=True, padding_value=0)
    rejected_ids_padded = torch.nn.utils.rnn.pad_sequence(rejected_ids_list, batch_first=True, padding_value=0)
    rejected_mask_padded = torch.nn.utils.rnn.pad_sequence(rejected_mask_list, batch_first=True, padding_value=0)

    return {
        "chosen_input_ids": chosen_ids_padded,
        "chosen_attention_mask": chosen_mask_padded,
        "rejected_input_ids": rejected_ids_padded,
        "rejected_attention_mask": rejected_mask_padded
    }


def eval_collate_fn(batch):
    """
    Pads sequences for evaluation and returns the original sample dicts.
    """
    input_ids_list = [item["tokenized"]["input_ids"] for item in batch]
    attention_mask_list = [item["tokenized"]["attention_mask"] for item in batch]
    samples = batch  # preserve original sample dicts
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "samples": samples}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    instruction = get_instruction(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)

    # Use the new pairwise dataset and collate function
    train_dataset = PairwiseRewardTrainDataset(args.train_file, tokenizer, max_length=args.max_length,
                                               instruction=instruction, max_train_samples=args.max_train_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=max(args.batch_size//2, 1),
                                  shuffle=True, collate_fn=pairwise_train_collate_fn)

    model = AutoModel.from_pretrained(
        args.reward_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device).to(torch.bfloat16)
    model.train()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # The loss function is now implemented directly in the training loop, so no criterion object is needed.

    print("Starting training with pairwise preference loss...")
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in progress:
            optimizer.zero_grad()

            # Get chosen and rejected inputs and move to device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            # Get rewards for both chosen and rejected responses
            chosen_rewards = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask).squeeze(-1)
            rejected_rewards = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask).squeeze(-1)

            # --- Custom Loss Calculation (as per Eurus paper) ---
            # L_ULTRAINTERACT = L_BT + L_DR

            # L_BT: Bradley-Terry component to optimize relative rewards
            # This term maximizes the margin between chosen and rejected rewards.
            # -log(sigmoid(r_chosen - r_rejected))
            margins = chosen_rewards - rejected_rewards
            loss_bt = -torch.nn.functional.logsigmoid(margins).mean()

            # L_DR: Direct Reward component to optimize absolute rewards
            # This term pushes chosen rewards to be positive and rejected rewards to be negative.
            # -log(sigmoid(r_chosen)) - log(sigmoid(-r_rejected))
            loss_dr_chosen = -torch.nn.functional.logsigmoid(chosen_rewards).mean()
            loss_dr_rejected = -torch.nn.functional.logsigmoid(-rejected_rewards).mean()

            # The final loss is the sum of the two components
            loss = loss_bt + loss_dr_chosen + loss_dr_rejected

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Fine tuned LoRA parameters saved to {args.output_dir}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    instruction = get_instruction(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)

    eval_dataset = RewardEvalDataset(args.eval_file, tokenizer, max_length=args.max_length,
                                     instruction=instruction, max_eval_samples=args.max_eval_samples)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=eval_collate_fn)

    model = AutoModel.from_pretrained(
        args.reward_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.output_dir)
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.splitext(args.eval_file)[0] + "_with_scores.jsonl"

    with open(output_file, "w", encoding="utf-8") as fout:
        progress = tqdm(eval_dataloader, desc="Evaluating")
        for batch in progress:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rewards = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
                rewards = rewards.detach().cpu().tolist()
            samples = batch["samples"]
            for sample, score in zip(samples, rewards):
                sample["score"] = score
                if "tokenized" in sample:
                    del sample["tokenized"]
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Evaluation results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine tune and evaluate the reward model with PEFT LoRA using pairwise preference loss. "
                    "The base model is assumed to directly output a reward scalar without an extra regression head."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--reward_name_or_path", type=str, required=True,
                        help="Pretrained base model name or path.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (for loading samples).")
    parser.add_argument("--train_file", type=str, help="Path to the training JSONL file.")
    parser.add_argument("--eval_file", type=str, help="Path to the evaluation JSONL file.")
    parser.add_argument("--output_dir", type=str, default="model/lora_reward_model_pairwise",
                        help="Directory to save the fine tuned LoRA parameters.")
    parser.add_argument("--output_file", type=str,
                        help="Output JSONL file to save evaluation results (with added 'score' field).")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum token length.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training/evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine tuning.")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha scaling factor.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of evaluation samples.")
    parser.add_argument("--do_train", action="store_true", help="Run fine tuning.")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.do_train:
        if args.train_file is None:
            raise ValueError("Training file must be provided when --do_train is set.")
        train(args)
    if args.do_eval:
        if args.eval_file is None:
            raise ValueError("Evaluation file must be provided when --do_eval is set.")
        evaluate(args)


if __name__ == "__main__":
    main()
