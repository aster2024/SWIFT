#!/usr/bin/env python
import argparse
import warnings
import os
import sys
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def dpo_loss(r_pos, r_neg):
    """
    DPO loss using logsigmoid for numerical stability.
    L_DPO(θ) = -log(sigmoid(r₊ - r₋))
    """
    return -F.logsigmoid(r_pos - r_neg).mean()


def infonca_loss(r_pred, r_gt, alpha):
    """
    InfoNCA loss implementation using log_softmax.
    L_InfoNCA(θ) = -Σ_i [ softmax(r_gt/α)_i * log_softmax(r_pred)_i ]
    """
    p_gt = F.softmax(r_gt / alpha, dim=0)
    log_probs = F.log_softmax(r_pred, dim=0)
    loss = - (p_gt * log_probs).sum()
    return loss


def nca_loss(r_pred, r_gt, alpha):
    """
    NCA loss using logsigmoid for numerical stability.
    L_NCA(θ) = - Σ_i [ softmax(r_gt/α)_i * log(sigmoid(r_pred_i)) + (1/K)*log(sigmoid(-r_pred_i)) ]
    """
    K = r_pred.size(0)
    weights = F.softmax(r_gt / alpha, dim=0)
    optimization = weights * F.logsigmoid(r_pred)
    regularization = (1.0 / K) * F.logsigmoid(-r_pred)
    loss = -torch.sum(optimization + regularization)
    return loss


class InMemoryRewardDataset(Dataset):
    """
    Dataset that builds examples from in-memory summary and detailed samples.
    Each example corresponds to a single candidate.
    Used for methods 'ce' and 'hinge'.
    """
    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        self.index_pairs = []
        for i, summary_sample in enumerate(summary_data):
            detailed_sample = detailed_data[i]
            labels = summary_sample.get("correctness", [])
            for j in range(min(len(labels), len(detailed_sample.get("detailed_paths", [])))):
                self.index_pairs.append((i, j))

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        i, j = self.index_pairs[idx]
        summary_sample = self.summary_data[i]
        detailed_sample = self.detailed_data[i]
        candidate = extract_candidate_features(summary_sample, detailed_sample, j)
        if candidate is None:
            warnings.warn(f"Skipping sample {i}, reasoning index {j}")
            return None
        return candidate


class DPORewardDataset(Dataset):
    """
    Dataset for DPO method that yields all positive-negative candidate pairs.
    For each sample, it selects all combinations of positive and negative candidates.
    """
    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        self.pairs = []
        for i, summary_sample in enumerate(summary_data):
            detailed_sample = detailed_data[i]
            labels = summary_sample.get("correctness", [])
            pos_indices = []
            neg_indices = []
            num_candidates = min(len(labels), len(detailed_sample.get("detailed_paths", [])))
            for j in range(num_candidates):
                if labels[j]:
                    pos_indices.append(j)
                else:
                    neg_indices.append(j)
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    self.pairs.append((i, pos_idx, neg_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_idx, pos_idx, neg_idx = self.pairs[idx]
        summary_sample = self.summary_data[sample_idx]
        detailed_sample = self.detailed_data[sample_idx]
        pos_candidate = extract_candidate_features(summary_sample, detailed_sample, pos_idx)
        neg_candidate = extract_candidate_features(summary_sample, detailed_sample, neg_idx)
        if pos_candidate is None or neg_candidate is None:
            warnings.warn(f"Skipping sample {sample_idx}, positive index {pos_idx}, negative index {neg_idx}")
            return None
        pos_feat, pos_label, pos_length = pos_candidate
        neg_feat, neg_label, neg_length = neg_candidate
        return (pos_feat, pos_length), (neg_feat, neg_length)


class MultiCandidateRewardDataset(Dataset):
    """
    Dataset for multi-candidate methods (InfoNCA and NCA).
    Each sample contains all valid candidates for a given prompt.
    Only samples with at least two valid candidates are kept.
    """
    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        self.valid_sample_indices = []
        for i, summary_sample in enumerate(summary_data):
            detailed_sample = detailed_data[i]
            labels = summary_sample.get("correctness", [])
            num_paths = len(detailed_sample.get("detailed_paths", []))
            num_valid_candidates = 0
            for j in range(min(len(labels), num_paths)):
                if j < len(labels) and j < len(detailed_sample.get("detailed_paths", [])):
                    num_valid_candidates += 1
            if num_valid_candidates >= 2:
                self.valid_sample_indices.append(i)

    def __len__(self):
        return len(self.valid_sample_indices)

    def __getitem__(self, idx):
        i = self.valid_sample_indices[idx]
        summary_sample = self.summary_data[i]
        detailed_sample = self.detailed_data[i]
        candidates = []
        labels = summary_sample.get("correctness", [])
        num_paths = len(detailed_sample.get("detailed_paths", []))
        for j in range(min(len(labels), num_paths)):
            candidate = extract_candidate_features(summary_sample, detailed_sample, j)
            if candidate is not None:
                candidates.append(candidate)
        features, labels, lengths = [], [], []
        for feat, lbl, length in candidates:
            features.append(feat)
            labels.append(lbl)
            lengths.append(length)
        return features, labels, lengths


def collate_fn(batch):
    """
    Collate function for single-candidate methods (CE and hinge).
    Pads variable-length sequences.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    token_features_list, labels_list, lengths_list = zip(*batch)
    padded_features = pad_sequence(token_features_list, batch_first=True, padding_value=0)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return padded_features, lengths_list, labels


def collate_fn_dpo(batch):
    """
    Collate function for DPO dataset.
    Gathers positive and negative candidates.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    pos_features, pos_lengths, neg_features, neg_lengths = [], [], [], []
    for (pos, pos_len), (neg, neg_len) in batch:
        pos_features.append(pos)
        pos_lengths.append(pos_len)
        neg_features.append(neg)
        neg_lengths.append(neg_len)
    padded_pos = pad_sequence(pos_features, batch_first=True, padding_value=0)
    padded_neg = pad_sequence(neg_features, batch_first=True, padding_value=0)
    return padded_pos, pos_lengths, padded_neg, neg_lengths


def collate_fn_multi(batch):
    """
    For multi-candidate methods, simply return the batch as is.
    Each element is a tuple (features_list, labels_list, lengths_list).
    """
    return batch


def get_collate_fn_memory_efficient(params):
    """
    Returns a collate function for single-candidate methods in memory efficient mode,
    extracting candidate features on the fly. If logits_mode is True, SWIFT will be trained
    on the model's final output logits instead of concatenated hidden states.
    """
    instruction = params["instruction"]
    separator = params["separator"]
    layers = params["layers"]
    tokenizer = params["tokenizer"]
    model_lm = params["model_lm"]
    apply_norm = params["apply_norm"]
    logits_mode = params.get("logits_mode", False)

    def collate_fn_memory_efficient(batch):
        candidate_list = []
        batched_prompts = []
        batched_reasoning = []
        candidate_meta = []  # Maps candidate to its corresponding raw sample and candidate index.
        for raw_sample in batch:
            try:
                prompt = raw_sample.get("prompt", "")
                raw_paths = raw_sample.get("steps", [])
                correctness = raw_sample.get("correctness", [])
                for j, reasoning_steps in enumerate(raw_paths):
                    batched_prompts.append(prompt)
                    batched_reasoning.append(reasoning_steps)
                    candidate_meta.append((raw_sample, correctness, j))
            except Exception as e:
                continue
        if len(batched_prompts) == 0:
            return None

        # Choose extraction function based on logits_mode.
        if logits_mode:
            detailed_info_list = extract_logits_info_for_reasoning_path_batch(
                batched_prompts, batched_reasoning, instruction, separator, tokenizer, model_lm, to_cpu=False
            )
        else:
            detailed_info_list = extract_detailed_info_for_reasoning_path_batch(
                batched_prompts, batched_reasoning, instruction, separator, layers, tokenizer, model_lm, apply_norm, to_cpu=False
            )

        sample_dict = {}
        for meta, detailed_info in zip(candidate_meta, detailed_info_list):
            raw_sample, correctness, cand_index = meta
            key = id(raw_sample)
            if key not in sample_dict:
                sample_dict[key] = (raw_sample, correctness, [])
            sample_dict[key][2].append(detailed_info)
        for raw_sample, correctness, detailed_infos in sample_dict.values():
            for j in range(min(len(correctness), len(detailed_infos))):
                detailed_sample = {
                    "prompt": raw_sample.get("prompt", ""),
                    "steps": raw_sample.get("steps", []),
                    "detailed_paths": detailed_infos
                }
                candidate = extract_candidate_features(raw_sample, detailed_sample, j)
                if candidate is not None:
                    candidate_list.append(candidate)
        if not candidate_list:
            return None
        token_features_list, labels_list, lengths_list = zip(*candidate_list)
        padded_features = pad_sequence(token_features_list, batch_first=True, padding_value=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
        return padded_features, lengths_list, labels_tensor

    return collate_fn_memory_efficient


def get_collate_fn_dpo_memory_efficient(params):
    """
    Returns a collate function for the DPO method in memory efficient mode using batch extraction.
    If logits_mode is True, logits extraction is used.
    """
    instruction = params["instruction"]
    separator = params["separator"]
    layers = params["layers"]
    tokenizer = params["tokenizer"]
    model_lm = params["model_lm"]
    apply_norm = params["apply_norm"]
    logits_mode = params.get("logits_mode", False)

    def collate_fn_dpo_memory_efficient(batch):
        pos_features, pos_lengths, neg_features, neg_lengths = [], [], [], []
        for raw_sample in batch:
            try:
                prompt = raw_sample.get("prompt", "")
                raw_paths = raw_sample.get("steps", [])
                correctness = raw_sample.get("correctness", [])
                batch_prompts = []
                batch_reasoning = []
                for j, reasoning_steps in enumerate(raw_paths):
                    batch_prompts.append(prompt)
                    batch_reasoning.append(reasoning_steps)
                if len(batch_prompts) == 0:
                    continue
                if logits_mode:
                    detailed_info_list = extract_logits_info_for_reasoning_path_batch(
                        batch_prompts, batch_reasoning, instruction, separator, tokenizer, model_lm, to_cpu=False
                    )
                else:
                    detailed_info_list = extract_detailed_info_for_reasoning_path_batch(
                        batch_prompts, batch_reasoning, instruction, separator, layers, tokenizer, model_lm, apply_norm, to_cpu=False
                    )
                detailed_sample = {
                    "prompt": prompt,
                    "steps": raw_paths,
                    "detailed_paths": detailed_info_list
                }
                pos_indices = []
                neg_indices = []
                num_candidates = min(len(correctness), len(detailed_info_list))
                for j in range(num_candidates):
                    if correctness[j]:
                        pos_indices.append(j)
                    else:
                        neg_indices.append(j)
                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        pos_candidate = extract_candidate_features(raw_sample, detailed_sample, pos_idx)
                        neg_candidate = extract_candidate_features(raw_sample, detailed_sample, neg_idx)
                        if pos_candidate is not None and neg_candidate is not None:
                            pos_feat, _, pos_len = pos_candidate
                            neg_feat, _, neg_len = neg_candidate
                            pos_features.append(pos_feat)
                            pos_lengths.append(pos_len)
                            neg_features.append(neg_feat)
                            neg_lengths.append(neg_len)
            except Exception as e:
                continue
        if not pos_features:
            return None
        padded_pos = pad_sequence(pos_features, batch_first=True, padding_value=0)
        padded_neg = pad_sequence(neg_features, batch_first=True, padding_value=0)
        return padded_pos, pos_lengths, padded_neg, neg_lengths

    return collate_fn_dpo_memory_efficient


def get_collate_fn_multi_memory_efficient(params):
    """
    Returns a collate function for multi-candidate methods (InfoNCA and NCA) in memory efficient mode using batch extraction.
    If logits_mode is True, logits extraction is used.
    """
    instruction = params["instruction"]
    separator = params["separator"]
    layers = params["layers"]
    tokenizer = params["tokenizer"]
    model_lm = params["model_lm"]
    apply_norm = params["apply_norm"]
    logits_mode = params.get("logits_mode", False)

    def collate_fn_multi_memory_efficient(batch):
        new_batch = []
        for raw_sample in batch:
            try:
                prompt = raw_sample.get("prompt", "")
                raw_paths = raw_sample.get("steps", [])
                correctness = raw_sample.get("correctness", [])
                if len(correctness) < 2 or len(raw_paths) < 2:
                    continue
                batch_prompts = []
                batch_reasoning = []
                for reasoning_steps in raw_paths:
                    batch_prompts.append(prompt)
                    batch_reasoning.append(reasoning_steps)
                if logits_mode:
                    detailed_info_list = extract_logits_info_for_reasoning_path_batch(
                        batch_prompts, batch_reasoning, instruction, separator, tokenizer, model_lm, to_cpu=False
                    )
                else:
                    detailed_info_list = extract_detailed_info_for_reasoning_path_batch(
                        batch_prompts, batch_reasoning, instruction, separator, layers, tokenizer, model_lm, apply_norm, to_cpu=False
                    )
                if len(detailed_info_list) < 2:
                    continue
                candidates = []
                detailed_sample = {"prompt": prompt, "steps": raw_paths, "detailed_paths": detailed_info_list}
                for j in range(min(len(correctness), len(detailed_info_list))):
                    candidate = extract_candidate_features(raw_sample, detailed_sample, j)
                    if candidate is not None:
                        candidates.append(candidate)
                if len(candidates) < 2:
                    continue
                features, labels, lengths = zip(*candidates)
                new_batch.append((list(features), list(labels), list(lengths)))
            except Exception as e:
                continue
        return new_batch

    return collate_fn_multi_memory_efficient


class RawRewardDataset(Dataset):
    """
    Raw dataset that returns the raw samples.
    Used for on-the-fly extraction in memory efficient mode.
    """
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_reward_model(reward_model, optimizer, summary_data, detailed_data, args, device, raw_data=None,
                       extra_params=None):
    """
    Train the reward model on the provided dataset.
    If memory efficient mode is enabled (args.memory_efficient=True), the raw_data is used
    along with on-the-fly candidate extraction via a custom collate function.
    Otherwise, pre-extracted summary_data and detailed_data are used.
    """
    if args.memory_efficient:
        dataset = RawRewardDataset(raw_data)
        if args.method in ["ce", "hinge"]:
            collate_function = get_collate_fn_memory_efficient(extra_params)
        elif args.method == "dpo":
            collate_function = get_collate_fn_dpo_memory_efficient(extra_params)
        elif args.method in ["infonca", "nca"]:
            collate_function = get_collate_fn_multi_memory_efficient(extra_params)
        else:
            raise ValueError(f"Unknown method: {args.method}")
    else:
        if args.method in ["ce", "hinge"]:
            dataset = InMemoryRewardDataset(summary_data, detailed_data)
            collate_function = collate_fn
        elif args.method == "dpo":
            dataset = DPORewardDataset(summary_data, detailed_data)
            collate_function = collate_fn_dpo
        elif args.method in ["infonca", "nca"]:
            dataset = MultiCandidateRewardDataset(summary_data, detailed_data)
            collate_function = collate_fn_multi
        else:
            raise ValueError(f"Unknown method: {args.method}")

    if args.train_val_split_ratio is not None and 0 < args.train_val_split_ratio < 1:
        total = len(dataset)
        train_size = int(total * (1 - args.train_val_split_ratio))
        val_size = total - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_function)
        val_dataloader = DataLoader(val_dataset, batch_size=args.reward_batch_size, shuffle=False,
                                    collate_fn=collate_function)
    else:
        train_dataloader = DataLoader(dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_function)
        val_dataloader = None

    if args.method == "hinge":
        def hinge_loss(outputs, labels):
            # Transform labels from {0,1} to {-1, +1}.
            labels_transformed = labels * 2 - 1
            return torch.mean(torch.clamp(1 - labels_transformed * outputs, min=0))

        criterion = hinge_loss
    elif args.method == "ce":
        criterion = nn.BCEWithLogitsLoss()

    reward_model.train()
    patience = 3  # Early stopping patience.
    best_val_loss = float('inf')
    patience_counter = 0
    best_reward_model = None

    for epoch in range(args.reward_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Reward Model Training Epoch {epoch + 1}/{args.reward_epochs}",
                            leave=False)
        if args.method in ["ce", "hinge"]:
            for batch in progress_bar:
                if batch is None:
                    continue
                features, lengths, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = reward_model(features, lengths)  # shape: (batch_size,)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        elif args.method == "dpo":
            for batch in progress_bar:
                if batch is None:
                    continue
                pos_features, pos_lengths, neg_features, neg_lengths = batch
                pos_features = pos_features.to(device)
                neg_features = neg_features.to(device)
                optimizer.zero_grad()
                r_pos = reward_model(pos_features, pos_lengths)
                r_neg = reward_model(neg_features, neg_lengths)
                loss = dpo_loss(r_pos, r_neg)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        elif args.method in ["infonca", "nca"]:
            for batch in progress_bar:
                if batch is None:
                    continue
                batch_losses = []
                # Each item in the batch is (features_list, labels_list, lengths_list)
                for features_list, labels_list, lengths_list in batch:
                    candidate_rewards = []
                    for feat, length in zip(features_list, lengths_list):
                        feat = feat.to(device).unsqueeze(0)  # add batch dimension
                        r = reward_model(feat, [length])
                        candidate_rewards.append(r.squeeze(0))
                    candidate_rewards = torch.stack(candidate_rewards)  # shape: (K,)
                    gt_rewards = torch.tensor(labels_list, dtype=torch.float32, device=device)
                    if args.method == "infonca":
                        loss_sample = infonca_loss(candidate_rewards, gt_rewards, args.alpha)
                    else:
                        loss_sample = nca_loss(candidate_rewards, gt_rewards, args.alpha)
                    batch_losses.append(loss_sample)
                if batch_losses:
                    loss = torch.stack(batch_losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1}/{args.reward_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase.
        if val_dataloader is not None:
            reward_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                if args.method in ["ce", "hinge"]:
                    for batch in val_dataloader:
                        if batch is None:
                            continue
                        features, lengths, labels = batch
                        features = features.to(device)
                        labels = labels.to(device)
                        outputs = reward_model(features, lengths)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * features.size(0)
                elif args.method == "dpo":
                    for batch in val_dataloader:
                        if batch is None:
                            continue
                        pos_features, pos_lengths, neg_features, neg_lengths = batch
                        pos_features = pos_features.to(device)
                        neg_features = neg_features.to(device)
                        r_pos = reward_model(pos_features, pos_lengths)
                        r_neg = reward_model(neg_features, neg_lengths)
                        loss = dpo_loss(r_pos, r_neg)
                        val_loss += loss.item() * pos_features.size(0)
                elif args.method in ["infonca", "nca"]:
                    sample_losses = []
                    for batch in val_dataloader:
                        for features_list, labels_list, lengths_list in batch:
                            candidate_rewards = []
                            for feat, length in zip(features_list, lengths_list):
                                feat = feat.to(device).unsqueeze(0)
                                r = reward_model(feat, [length])
                                candidate_rewards.append(r.squeeze(0))
                            candidate_rewards = torch.stack(candidate_rewards)
                            gt_rewards = torch.tensor(labels_list, dtype=torch.float32, device=device)
                            if args.method == "infonca":
                                loss_sample = infonca_loss(candidate_rewards, gt_rewards, args.alpha)
                            else:
                                loss_sample = nca_loss(candidate_rewards, gt_rewards, args.alpha)
                            sample_losses.append(loss_sample)
                    if sample_losses:
                        val_loss = torch.stack(sample_losses).mean().item() * len(sample_losses)
            avg_val_loss = val_loss / (len(val_dataloader.dataset) if val_dataloader is not None else 1)
            print(f"Epoch {epoch + 1}/{args.reward_epochs}, Val Loss: {avg_val_loss:.4f}")
            reward_model.train()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_reward_model = reward_model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss.")
                    break

    return best_reward_model if best_reward_model is not None else reward_model


def process_data_and_train(args):
    # Get the instruction for the dataset.
    instruction = get_instruction(args.dataset)

    ds = load_data(f"data/{args.dataset}/train_{args.dataset}_{args.model_name.split('/')[-1]}.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_lm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If memory_efficient mode is enabled, extract features on the fly.
    if args.memory_efficient:
        print("Running in memory efficient mode: Extracting candidate features on-the-fly during training.")
        # Determine feature dimension and n_rollouts from the first valid sample.
        feature_dim = None
        for sample in ds:
            try:
                prompt = sample.get("prompt", "")
                raw_paths = sample.get("steps", [])
                if not raw_paths:
                    continue
                n_rollouts = len(raw_paths)
                if args.logits_mode:
                    # Use logits extraction: our reward model will use final logits.
                    detailed_info = extract_logits_info_for_reasoning_path(
                        prompt, instruction, raw_paths[0], args.separator, tokenizer, model_lm, to_cpu=True
                    )
                    logits_tensor = detailed_info["logits"]  # shape (1, L, vocab_size)
                    feature_dim = logits_tensor.shape[-1]  # Use the logits dimension
                    print(f"Detected feature dimension (logits mode): {feature_dim}")
                else:
                    detailed_info = extract_detailed_info_for_reasoning_path(
                        prompt, instruction, raw_paths[0], args.separator, args.layers, tokenizer, model_lm, args.apply_norm
                    )
                    hs = detailed_info["hidden_states"]
                    sorted_layers = sorted(hs.keys(), key=lambda x: int(x))
                    fd = 0
                    for layer in sorted_layers:
                        if hs[layer] is not None:
                            fd += hs[layer].shape[-1]
                    feature_dim = fd
                    print(f"Detected feature dimension: {feature_dim}")
                break
            except Exception as e:
                continue
        if feature_dim is None:
            raise ValueError("No valid sample with detailed information was found in memory efficient mode.")
        args.reward_batch_size = max(args.reward_batch_size // n_rollouts, 1)
        print(f"Actual batch size for training: {args.reward_batch_size * n_rollouts}")

        # Truncate to max_samples if specified.
        if args.max_samples is not None and args.max_samples < len(ds):
            new_ds = []
            for i, sample in enumerate(ds):
                if i >= args.max_samples:
                    break
                new_ds.append(sample)
            ds = new_ds
    else:
        summary_samples = []
        detailed_samples = []
        feature_dim = None
        total_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
        sample_count = 0
        with tqdm(total=total_samples, desc="Extracting samples") as pbar:
            for i, sample in enumerate(ds):
                prompt = sample.get("prompt", "")
                raw_paths = sample.get("steps", [])
                if not isinstance(raw_paths, list):
                    print(f"Skipping sample {i}: steps field format error.")
                    pbar.update(1)
                    continue
                try:
                    detailed_info_paths = []
                    for reasoning_steps in raw_paths:
                        if args.logits_mode:
                            info = extract_logits_info_for_reasoning_path(
                                prompt, instruction, reasoning_steps, args.separator, tokenizer, model_lm, to_cpu=True
                            )
                        else:
                            info = extract_detailed_info_for_reasoning_path(
                                prompt, instruction, reasoning_steps, args.separator, args.layers, tokenizer, model_lm,
                                args.apply_norm
                            )
                        detailed_info_paths.append(info)
                except Exception as e:
                    print(f"Skipping sample {i}: {str(e)}")
                    pbar.update(1)
                    continue
                output_sample = {
                    "prompt": prompt,
                    "reference": sample.get("reference", ""),
                    "dataset": sample.get("dataset", ""),
                    "completions": sample.get("completions", []),
                    "correctness": sample.get("correctness", []),
                    "steps": raw_paths
                }
                summary_samples.append(output_sample)
                detailed_sample = {
                    "prompt": prompt,
                    "steps": raw_paths,
                    "detailed_paths": detailed_info_paths
                }
                detailed_samples.append(detailed_sample)
                if feature_dim is None and len(detailed_info_paths) > 0:
                    if args.logits_mode:
                        logits_tensor = detailed_info_paths[0]["logits"]
                        feature_dim = logits_tensor.shape[-1]
                        print(f"Detected feature dimension (logits mode): {feature_dim}")
                    else:
                        hs = detailed_info_paths[0]["hidden_states"]
                        sorted_layers = sorted(hs.keys(), key=lambda x: int(x))
                        feature_dim = 0
                        for layer in sorted_layers:
                            if hs[layer] is not None:
                                feature_dim += hs[layer].shape[-1]
                        print(f"Detected feature dimension: {feature_dim}")
                sample_count += 1
                pbar.update(1)
                if args.max_samples and sample_count >= args.max_samples:
                    break
        del model_lm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    for method in args.methods:
        print(f"\nStarting training for method: {method}")
        args.method = method
        norm_part = "norm_" if args.apply_norm else ""
        if args.output_model_file is None:
            output_file = f"model/{args.dataset}/reward_model_{method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
        else:
            output_file = args.output_model_file

        base_model = LinearRewardModel(feature_dim, disable_gate=args.disable_gate).to(device)
        if args.use_dim_reduction:
            dim_reduction = DimReduction(feature_dim, args.dim_reduction_dim).to(device)
            base_model = RewardModelWithDimReduction(base_model, dim_reduction).to(device)
        reward_model = base_model
        reward_optimizer = optim.AdamW(
            reward_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        if args.memory_efficient:
            print(f"Training reward model for method '{method}' on {len(ds)} raw samples (memory efficient mode)...")
            extra_params = {
                "instruction": instruction,
                "separator": args.separator,
                "layers": args.layers,
                "tokenizer": tokenizer,
                "model_lm": model_lm,
                "apply_norm": args.apply_norm,
                "logits_mode": args.logits_mode
            }
            reward_model = train_reward_model(reward_model, reward_optimizer, None, None, args, device,
                                              raw_data=ds, extra_params=extra_params)
        else:
            print(f"Training reward model for method '{method}' on {len(summary_samples)} samples...")
            reward_model = train_reward_model(reward_model, reward_optimizer, summary_samples, detailed_samples, args,
                                              device)
        torch.save(reward_model.state_dict(), output_file)
        print(f"Training completed for method '{method}'. Reward model saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a reward model for reasoning paths using various methods."
    )
    # Extraction-related arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path, e.g., 'gpt2'")
    parser.add_argument("--dataset", type=str, default="math",
                        help="Dataset name (default: math)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to save hidden states; if not provided, save all layers")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join prompt and reasoning steps (default: two newlines)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: process all samples)")
    # Reward model training arguments.
    parser.add_argument("--reward_epochs", type=int, default=40,
                        help="Number of training epochs for the reward model (default: 40)")
    parser.add_argument("--reward_batch_size", type=int, default=16,
                        help="Batch size for reward model training (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the reward model (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer (default: 1e-5)")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism in the linear reward model")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply layer normalization to hidden states before reward model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_model_file", type=str, default=None,
                        help="Base path to save the trained reward models. Method name will be appended if not provided explicitly.")
    # Training arguments.
    parser.add_argument("--train_val_split_ratio", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--methods", type=str, nargs="+", choices=["ce", "hinge", "dpo", "infonca", "nca"],
                        default=["ce", "hinge", "dpo", "infonca", "nca"],
                        help="List of reward methods: 'ce' for cross entropy, 'hinge' for hinge loss, 'dpo', 'infonca', or 'nca' for contrastive losses")
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="If set, apply fixed dimensionality reduction before the reward model")
    parser.add_argument("--dim_reduction_dim", type=int, default=128,
                        help="Target dimension for dimensionality reduction (default: 128)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Alpha hyperparameter for InfoNCA and NCA losses (default: 0.01)")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="If set, use memory efficient mode where candidate features are extracted on-the-fly during training.")
    parser.add_argument("--logits_mode", action="store_true",
                        help="If set, use the final logits of the model instead of hidden states. This "
                             "enables training when hidden states are unavailable (e.g., for closed-source models).")

    args = parser.parse_args()

    if args.output_model_file is not None:
        output_dir = os.path.dirname(args.output_model_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    set_seed(args.seed)
    process_data_and_train(args)


if __name__ == "__main__":
    main()
