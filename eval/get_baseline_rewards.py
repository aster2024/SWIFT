#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, LlamaModel, LlamaConfig, PreTrainedModel, LlamaTokenizer
import time
import os
import argparse
from tqdm import tqdm
import sys
from typing import Optional, List
import time
from huggingface_hub import snapshot_download
import json
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Loads a JSON file containing samples, computes rewards using a reward model, and outputs a new JSON file with a 'score' field."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--reward_name_or_path", type=str,
                        required=True,
                        help="Path or name of the reward model.")
    parser.add_argument("--dataset", type=str,
                        required=True,
                        help="Dataset name (for loading samples).")
    parser.add_argument("--input_file", type=str,
                        required=True,
                        help="Path to the input JSON file containing the samples.")
    parser.add_argument("--output_file", type=str,
                        default=None,
                        help="Path to the output JSON file with updated samples.")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="Maximum number of samples to process.")
    parser.add_argument("--model_type", type=str,
                        choices=["Eurus", "Skywork", "Ultra", "Starling", "Deepseek", "Shepherd"], required=True,
                        help="Model type (for formatting answer).")

    return parser.parse_args()


def compute_reward(sample, model, tokenizer, instruction, device, model_type):
    """
    Computes the reward score for a given sample.
    """
    prompt = instruction + sample["prompt"]
    response = sample["response"]

    if model_type == "Skywork":
        conversation = [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    elif model_type == "Eurus":
        input_text = f"[INST] {prompt} [/INST] {response}"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif model_type == "Ultra":
        input_text = f"""Human: {prompt}

        Assistant: {response}"""
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif model_type == "Starling":
        input_text = f"<s>[INST] {prompt} </s> [/INST] {response}</s>"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif model_type == "Deepseek":
        conversation = [{"content": prompt + " " + response, "role": "user"}, {"content": "+", "role": "assistant"}]
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    elif model_type == "Shepherd":
        input_ids = torch.tensor([tokenizer.encode(f"{prompt} {response} ки")], device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    with torch.no_grad():
        if model_type in ["Eurus", "Ultra"]:
            score_value = model(**inputs).item()
        elif model_type == "Skywork":
            score_value = model(input_ids).logits[0][0].item()
        elif model_type == "Starling":
            score_value = model(**inputs)[0].item()
        elif model_type == "Deepseek":
            logits = model(input_ids).logits[:, -3, candidate_tokens]  # simple version for llama3.1-instruct, the +/- is predicted by the '-3' position
            score_value = logits.softmax(dim=-1)[:, 0].item()  # 0 means the prob of + (1 mean -)
        elif model_type == "Shepherd":
            logits = model(input_ids).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            score_value = scores[input_ids == final_tag_id].item()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return score_value



def process_samples(args, model, tokenizer, samples, instruction, device):
    """
    Processes each sample in the list and computes the reward.
    """
    total_time = 0.0
    updated_samples = []

    for sample in tqdm(samples, desc="Processing samples"):
        start_time = time.perf_counter()
        score = compute_reward(sample, model, tokenizer, instruction, device, args.model_type)
        total_time += time.perf_counter() - start_time

        sample["score"] = score
        updated_samples.append(sample)

        if args.max_samples is not None and len(updated_samples) >= args.max_samples:
            break

    print(f"Total time taken: {total_time:.4f} seconds")
    return updated_samples


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + f"_with_rewards_orm{args.reward_name_or_path.replace('/', '_')}.jsonl"

    instruction = get_instruction(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_data(args.input_file)

    print(f"Loaded {len(samples)} samples")

    print("Loading reward model and tokenizer...")

    if args.model_type == "Eurus":
        tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
        model = AutoModel.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    elif args.model_type == "Skywork":
        tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2", num_labels=1
        )
    elif args.model_type == "Ultra":
        tokenizer = LlamaTokenizer.from_pretrained(args.reward_name_or_path)
        model = LlamaRewardModel.from_pretrained(args.reward_name_or_path, torch_dtype=torch.bfloat16)
    elif args.model_type == "Starling":
        model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
        tokenizer = model.tokenizer
        tokenizer.truncation_side = "left"

        directory = snapshot_download(args.reward_name_or_path)
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        model.load_state_dict(torch.load(checkpoint), strict=False)
        model.eval().requires_grad_(False)
    elif args.model_type in ["Deepseek", "Shepherd"]:
        tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Reward model parameter count: {num_params}")

    model.to(device)
    model.eval()

    updated_samples = process_samples(args, model, tokenizer, samples, instruction, device)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
