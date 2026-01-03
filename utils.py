import torch
import torch.nn as nn
import random
import numpy as np
import warnings
import json
from torch.nn.utils.rnn import pad_sequence
import re

def segment_token_ids(full_token_id_list, boundaries):
    """
    Segment token id sequence based on boundaries.
    """
    segments = []
    for start, end in boundaries:
        segments.append(full_token_id_list[start:end])
    return segments


def extract_detailed_info_for_reasoning_path(prompt, instruction, reasoning_steps, separator, layers, tokenizer, model,
                                             apply_norm=False, to_cpu=True):
    """
    For a given prompt and list of reasoning steps, construct a dialogue, obtain model outputs,
    and determine token boundaries using robust conversation prefix processing.

    If apply_norm is True, apply the model's normalization function (retrieved using get_norm_and_head)
    to all hidden states before they are saved.
    """
    full_assistant_answer = separator.join(reasoning_steps)
    prompt = instruction + prompt
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_assistant_answer}
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    full_token_ids = inputs["input_ids"][0].tolist()

    boundaries = []

    conversation_prefix = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]
    conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
    conv = conv.strip()

    if conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
        conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
    if conv.endswith("<|eot_id|>"):
        conv = conv[:-len("<|eot_id|>")]
    prompt_ids = tokenizer.encode(conv, add_special_tokens=False)
    boundaries.append((0, len(prompt_ids)))
    last_length = len(prompt_ids)

    for i in range(len(reasoning_steps)):
        partial_answer = separator.join(reasoning_steps[: i + 1])
        conversation_prefix = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": partial_answer}
        ]
        conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
        conv = conv.strip()
        # If this is NOT the final reasoning step, remove trailing special tokens.
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|eot_id|>"):
            conv = conv[:-len("<|eot_id|>")]
        if (i + 1) != len(reasoning_steps):
            conv += separator
        current_ids = tokenizer.encode(conv, add_special_tokens=False)
        current_length = len(current_ids)
        boundaries.append((last_length, current_length))
        last_length = current_length

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    if apply_norm:
        norm_fn = get_norm(model)
        hidden_states = tuple(norm_fn(state) for state in hidden_states)

    if layers is None:
        layers_to_save = list(range(len(hidden_states)))
    else:
        layers_to_save = layers

    selected_hidden_states = {}
    if to_cpu:
        for layer in layers_to_save:
            if layer < len(hidden_states):
                selected_hidden_states[layer] = hidden_states[layer].detach().cpu()
            else:
                selected_hidden_states[layer] = None

        segments = segment_token_ids(full_token_ids, boundaries)

        detailed_info = {
            "conversations": conversation,
            "inputs": {k: v.detach().cpu().tolist() for k, v in inputs.items()},
            "hidden_states": selected_hidden_states,
            "boundaries": boundaries,
            "segments": segments
        }
    else:
        for layer in layers_to_save:
            if layer < len(hidden_states):
                selected_hidden_states[layer] = hidden_states[layer].detach()
            else:
                selected_hidden_states[layer] = None

        segments = segment_token_ids(full_token_ids, boundaries)

        detailed_info = {
            "conversations": conversation,
            "inputs": {k: v.detach().tolist() for k, v in inputs.items()},
            "hidden_states": selected_hidden_states,
            "boundaries": boundaries,
            "segments": segments
        }
    return detailed_info


def extract_detailed_info_for_reasoning_path_batch(prompts, reasoning_steps_list, instruction, separator, layers,
                                                   tokenizer, model, apply_norm=False, to_cpu=True):
    """
    Batch version of extract_detailed_info_for_reasoning_path.
    """
    batched_input_ids = []
    batched_attention_masks = []
    conversations_batch = []
    full_token_ids_batch = []
    boundaries_batch = []

    batch_size = len(prompts)

    for idx in range(batch_size):
        prompt = prompts[idx]
        reasoning_steps = reasoning_steps_list[idx]

        full_assistant_answer = separator.join(reasoning_steps)
        conversation = [
            {"role": "user", "content": instruction + prompt},
            {"role": "assistant", "content": full_assistant_answer}
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]  # shape: (L,)
        attention_mask = inputs["attention_mask"][0]  # shape: (L,)

        batched_input_ids.append(input_ids)
        batched_attention_masks.append(attention_mask)
        full_token_ids = input_ids.tolist()
        full_token_ids_batch.append(full_token_ids)
        conversations_batch.append(conversation)

        boundaries = []
        # First boundary: process conversation prefix with an empty assistant answer.
        conversation_prefix = [
            {"role": "user", "content": instruction + prompt},
            {"role": "assistant", "content": ""}
        ]
        conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
        conv = conv.strip()
        # Remove trailing special tokens if present.
        if conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        if conv.endswith("<|eot_id|>"):
            conv = conv[:-len("<|eot_id|>")]
        prompt_ids = tokenizer.encode(conv, add_special_tokens=False)
        boundaries.append((0, len(prompt_ids)))
        last_length = len(prompt_ids)

        # Compute boundaries for each partial reasoning step.
        for i in range(len(reasoning_steps)):
            partial_answer = separator.join(reasoning_steps[: i + 1])
            conversation_prefix = [
                {"role": "user", "content": instruction + prompt},
                {"role": "assistant", "content": partial_answer}
            ]
            conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
            conv = conv.strip()
            # For non-final steps, remove any trailing special tokens.
            if (i + 1) != len(reasoning_steps) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
                conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
            if (i + 1) != len(reasoning_steps) and conv.endswith("<|eot_id|>"):
                conv = conv[:-len("<|eot_id|>")]
            # If not the final step, add the separator.
            if (i + 1) != len(reasoning_steps):
                conv += separator
            current_ids = tokenizer.encode(conv, add_special_tokens=False)
            current_length = len(current_ids)
            boundaries.append((last_length, current_length))
            last_length = current_length

        boundaries_batch.append(boundaries)

    # Pad the input_ids and attention masks to the maximum sequence length in the batch.
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_input_ids = pad_sequence(batched_input_ids, batch_first=True, padding_value=pad_token_id)
    padded_attention_mask = pad_sequence(batched_attention_masks, batch_first=True, padding_value=0)

    batched_inputs = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask
    }
    batched_inputs = {k: v.to(model.device) for k, v in batched_inputs.items()}

    with torch.no_grad():
        outputs = model(**batched_inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # Tuple of tensors; each tensor has shape (B, max_seq_length, hidden_size)

    if apply_norm:
        norm_fn = get_norm(model)
        hidden_states = tuple(norm_fn(state) for state in hidden_states)

    if layers is None:
        layers_to_save = list(range(len(hidden_states)))
    else:
        layers_to_save = layers

    detailed_infos = []
    for i in range(batch_size):
        selected_hidden_states = {}
        for layer in layers_to_save:
            if layer < len(hidden_states):
                h_state = hidden_states[layer][i].unsqueeze(0).detach()
                if to_cpu:
                    h_state = h_state.cpu()
                selected_hidden_states[layer] = h_state
            else:
                warnings.warn(f"Layer {layer} not found in hidden states.")
                selected_hidden_states[layer] = None

        segments = segment_token_ids(full_token_ids_batch[i], boundaries_batch[i])

        sample_inputs = {
            "input_ids": full_token_ids_batch[i],
            "attention_mask": batched_attention_masks[i].tolist()
        }

        detailed_info = {
            "conversations": conversations_batch[i],
            "inputs": sample_inputs,
            "hidden_states": selected_hidden_states,
            "boundaries": boundaries_batch[i],
            "segments": segments
        }
        detailed_infos.append(detailed_info)

    return detailed_infos


def get_norm(model):
    """
    Retrieve the appropriate normalization function and lm_head for logit lens extraction.
    """
    if model.config.is_encoder_decoder:
        pointer = model.decoder
    else:
        pointer = model

    if hasattr(pointer, "final_layer_norm"):
        norm_fn = pointer.final_layer_norm
    elif hasattr(pointer, "gpt_neox"):
        norm_fn = pointer.gpt_neox.final_layer_norm
    elif hasattr(pointer.model, "norm"):
        norm_fn = pointer.model.norm
    elif hasattr(pointer.model, "final_layernorm"):
        norm_fn = pointer.model.final_layernorm
    else:
        raise NotImplementedError("Could not find a suitable LayerNorm function.")

    return norm_fn


class LinearRewardModel(nn.Module):
    """
    A simple linear layer model for reward prediction using averaging.
    """

    def __init__(self, feature_dim, disable_gate=False):
        super(LinearRewardModel, self).__init__()
        self.disable_gate = disable_gate
        if not disable_gate:
            # Fused layer that predicts both a gate value and a reward.
            self.fused_layer = nn.Linear(feature_dim, 2)
        else:
            # Simple single-layer for reward prediction.
            self.reward_layer = nn.Linear(feature_dim, 1)

    def forward(self, x, lengths, is_eval=False, boundaries=None, reward_mode=None):
        """
        x: Tensor of shape (batch_size, max_seq_length, feature_dim).
        lengths: List of actual sequence lengths per example.
        is_eval, boundaries, reward_mode are deprecated and not used in this implementation.

        Returns:
            Averaged reward score per sample (one value per candidate).
        """
        batch_size, max_seq_len, _ = x.size()
        device = x.device

        mask = torch.zeros((batch_size, max_seq_seq_len := max_seq_len), dtype=torch.float32, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0

        if not self.disable_gate:
            fused_output = self.fused_layer(x)  # (batch_size, seq_len, 2)
            gates = torch.sigmoid(fused_output[..., 0])  # (batch_size, seq_len)
            rewards = fused_output[..., 1]  # (batch_size, seq_len)
            sum_weighted_scores = torch.sum(gates * rewards * mask, dim=1)  # (batch_size)
            sum_gates = torch.sum(gates * mask, dim=1)  # (batch_size)
            avg_scores = sum_weighted_scores / sum_gates.clamp(min=1e-8)
        else:
            rewards = self.reward_layer(x).squeeze(-1)  # (batch_size, seq_len)
            masked_rewards = rewards * mask
            avg_scores = torch.sum(masked_rewards, dim=1) / torch.sum(mask, dim=1).clamp(min=1)
        return avg_scores


class DimReduction(nn.Module):
    """
    A fixed linear projection for dimensionality reduction.
    This module ensures that both training and testing use a consistent projection.
    """

    def __init__(self, input_dim, output_dim):
        raise NotImplementedError("This class is not ready for use.")
        super(DimReduction, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        raise NotImplementedError("This class is not ready for use.")
        # x: (batch_size, seq_length, input_dim)
        return self.proj(x)


class RewardModelWithDimReduction(nn.Module):
    """
    A wrapper that first applies a fixed dimension reduction,
    then applies the base reward model.
    """

    def __init__(self, base_model, dim_reduction):
        raise NotImplementedError("This class is not ready for use.")
        super(RewardModelWithDimReduction, self).__init__()
        self.dim_reduction = dim_reduction
        self.base_model = base_model

    def forward(self, x, lengths):
        raise NotImplementedError("This class is not ready for use.")
        # Reduce the feature dimension before feeding to the base model.
        x_reduced = self.dim_reduction(x)
        return self.base_model(x_reduced, lengths)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def extract_candidate_features(summary_sample, detailed_sample, reasoning_idx):
    """
    Extract token features, candidate label, and sequence length for a given candidate
    (reasoning path) from a sample.

    When the candidate contains 'logits' (logits_mode), the function extracts token features from
    the final output logits instead of concatenated hidden states.

    Returns:
       token_features (Tensor): Token features of shape (seq_length, feature_dim)
       label: Correctness label for the candidate.
       candidate_length (int): Sequence length of the candidate.
    """
    reasoning_paths = detailed_sample.get("detailed_paths", [])
    labels = summary_sample.get("correctness", [])
    if reasoning_idx >= len(reasoning_paths) or reasoning_idx >= len(labels):
        warnings.warn(f"Index mismatch for sample, reasoning index {reasoning_idx}")
        return None
    path = reasoning_paths[reasoning_idx]
    label = labels[reasoning_idx]


    if "logits" in path:
        logits = path.get("logits")
        boundaries = path.get("boundaries", None)
        if boundaries is not None and len(boundaries) >= 2:
            reward_start = boundaries[1][0]
            reward_end = boundaries[-1][1]
        else:
            reward_start = 0
            token_seq_length = logits.shape[1]  # assuming shape: (1, seq_length, vocab_size)
            reward_end = token_seq_length

        token_features = logits[0][reward_start:reward_end].clone().detach().to(torch.float32)
        candidate_length = token_features.size(0)
        return token_features, label, candidate_length

    hidden_states = path.get("hidden_states", {})
    if not hidden_states:
        warnings.warn(f"No hidden states found for reasoning index {reasoning_idx}. Skipping.")
        return None

    boundaries = path.get("boundaries", None)
    if boundaries is not None and len(boundaries) >= 2:
        # Use the reward segment from the start of the first reasoning step to the end of the last step.
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        reward_start = 0
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        token_seq_length = len(hidden_states[first_layer_key])
        reward_end = token_seq_length

    # Concatenate hidden states from selected layers.
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    layer_tensors = []
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)  # shape: (seq_length, hidden_dim)
        tensor = tensor[reward_start:reward_end]
        layer_tensors.append(tensor)

    seq_lengths = [t.size(0) for t in layer_tensors]
    if len(set(seq_lengths)) != 1:
        warnings.warn(f"Sequence lengths do not match for reasoning index {reasoning_idx}. Skipping.")
        return None

    token_features = torch.cat(layer_tensors, dim=-1)
    candidate_length = token_features.size(0)
    return token_features, label, candidate_length

def load_data(file_name):
    """
    Load the dataset from a JSON file that contains a list of candidate outputs.
    """
    if file_name.endswith('json'):
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(file_name, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    return data


def get_token_features(detailed_info):
    """
    Extract token features for the candidate portion from detailed_info.
    Uses the boundaries returned in detailed_info to select the tokens
    corresponding to the assistant's (i.e. reasoning) part.

    This function supports both hidden states and logits based extraction.
    If 'logits' are present in detailed_info (i.e., logits_mode is enabled),
    the extracted token features will be taken from the logits.
    Otherwise, the hidden states will be concatenated.
    """
    if "logits" in detailed_info:
        logits = detailed_info["logits"]
        boundaries = detailed_info.get("boundaries", None)
        if boundaries is not None and len(boundaries) >= 2:
            reward_start = boundaries[1][0]
            reward_end = boundaries[-1][1]
        else:
            warnings.warn("No boundaries found in detailed_info, using all tokens from logits.")
            reward_start = 0
            reward_end = logits.shape[1]  # logits assumed to be of shape (1, seq_length, vocab_size)
        token_features = logits[0][reward_start:reward_end].clone().detach().to(torch.float32)
        return token_features

    hidden_states = detailed_info.get("hidden_states", None)
    if not hidden_states:
        return None

    boundaries = detailed_info.get("boundaries", None)
    if boundaries is not None and len(boundaries) >= 2:
        # Use tokens from the start of the first reasoning step to the end of the last step.
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        warnings.warn("No boundaries found in detailed_info, using all tokens from hidden states.")
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        reward_start = 0
        reward_end = len(hidden_states[first_layer_key][0])

    # Extract and concatenate hidden states from all saved layers.
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    layer_tensors = []
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)
        tensor = tensor[reward_start:reward_end]
        layer_tensors.append(tensor)
    if len(layer_tensors) == 0:
        warnings.warn("No valid token features extracted from hidden states, skipping.")
        return None
    token_features = torch.cat(layer_tensors, dim=-1)
    return token_features



def compute_metrics(groups, k_vals, ext_reward_mode="none"):
    """
    Calculate pass@k accuracy for each k value.

    Depending on the chosen external reward integration mode, the candidate selection is modified.
    Modes:
      - "none": Use only the reward model's reward (original behavior).
      - "ranking": For each group, compute separate rankings for the reward and the external reward,
                   then compute a combined ranking (the worse, i.e., larger, of the two).
                   Then select the candidate with the lowest combined rank among the top k.
      - "scaling": For each group, scale the reward and external reward to [0,1] and average them.
                   Then select the candidate with the highest combined score among the top k.
    """
    metrics = {}
    for k in k_vals:
        correct_count = 0
        total = len(groups)
        for group in groups:
            if ext_reward_mode == "none":
                topk = group[:k]
                if topk:
                    best_candidate = max(topk, key=lambda x: x["reward"])
                    if best_candidate["correctness"]:
                        correct_count += 1
            elif ext_reward_mode == "ranking":
                sorted_by_reward = sorted(group, key=lambda x: x["reward"], reverse=True)
                reward_ranks = {id(c): idx for idx, c in enumerate(sorted_by_reward)}
                # Compute ranking from the external reward (higher is better)
                sorted_by_ext = sorted(group, key=lambda x: x["ext_reward"], reverse=True)
                ext_ranks = {id(c): idx for idx, c in enumerate(sorted_by_ext)}
                # Compute combined ranking: the worse (larger index) of the two rankings for each candidate
                for candidate in group:
                    candidate["combined_rank"] = max(reward_ranks[id(candidate)], ext_ranks[id(candidate)])
                # Sort candidates by combined ranking in ascending order (lower rank is better)
                sorted_candidates = sorted(group, key=lambda x: x["combined_rank"])
                topk = sorted_candidates[:k]
                if topk:
                    best_candidate = topk[0]
                    if best_candidate["correctness"]:
                        correct_count += 1
            elif ext_reward_mode == "scaling":
                # Combined scaling integration method
                rewards = [c["reward"] for c in group]
                ext_rewards = [c["ext_reward"] for c in group]
                r_min, r_max = min(rewards), max(rewards)
                er_min, er_max = min(ext_rewards), max(ext_rewards)
                for candidate in group:
                    # Scale the reward model's reward to [0,1]
                    if r_max - r_min != 0:
                        norm_reward = (candidate["reward"] - r_min) / (r_max - r_min)
                    else:
                        norm_reward = 1.0
                    # Scale the external reward to [0,1]
                    if er_max - er_min != 0:
                        norm_ext = (candidate["ext_reward"] - er_min) / (er_max - er_min)
                    else:
                        norm_ext = 1.0
                    candidate["integrated_score"] = (norm_reward + norm_ext) / 2.0
                # Sort candidates by integrated score in descending order (higher is better)
                sorted_candidates = sorted(group, key=lambda x: x["integrated_score"], reverse=True)
                topk = sorted_candidates[:k]
                if topk:
                    best_candidate = topk[0]
                    if best_candidate["correctness"]:
                        correct_count += 1
            else:
                raise ValueError(f"Invalid external reward integration mode: {ext_reward_mode}")
        metrics[k] = round((correct_count / total) * 100, 1) if total > 0 else 0.0
    return metrics


def extract_avg_response_feature(prompt, reasoning_steps, separator, layers, tokenizer, model, apply_norm=False):
    """
    Extract an averaged hidden state feature vector for the response segment.

    For a given prompt and candidate reasoning steps, this function:
      1. Constructs a conversation by joining the reasoning steps with the provided separator.
      2. Tokenizes the conversation (using the custom chat template of the tokenizer).
      3. Obtains the hidden states from the model.
      4. Concatenates the hidden states from the specified layers.
      5. Determines the token boundaries so that only the assistant's response is considered.
      6. Averages the concatenated hidden states over the token dimension for the response part.

    Args:
        prompt (str): The input prompt.
        reasoning_steps (list): List of reasoning steps (strings) representing the candidate response.
        separator (str): String used to join the reasoning steps.
        layers (list or None): List of layer indices to use. If None, use all layers.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model (transformers.PreTrainedModel): The language model.
        apply_norm (bool): If True, apply layer normalization on the concatenated hidden states.

    Returns:
        np.ndarray: Averaged hidden state feature vector (1D numpy array) for the response.
    """
    full_answer = separator.join(reasoning_steps)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_answer}
    ]

    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    full_token_ids = inputs["input_ids"][0].tolist()

    model_output = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = model_output.hidden_states

    if layers is None:
        layers_to_use = list(range(len(hidden_states)))
    else:
        layers_to_use = layers

    layer_features = []
    for layer in layers_to_use:
        if layer < len(hidden_states) and hidden_states[layer] is not None:
            layer_features.append(hidden_states[layer].squeeze(0))  # shape: (seq_len, hidden_dim)

    concatenated_features = torch.cat(layer_features, dim=-1)

    if apply_norm:
        concatenated_features = torch.nn.functional.layer_norm(
            concatenated_features, concatenated_features.shape[-1:]
        )

    conversation_prefix = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]

    conv_prefix = tokenizer.apply_chat_template(
        conversation_prefix,
        tokenize=False,
        add_generation_prompt=False
    ).strip()

    prefix_ids = tokenizer.encode(conv_prefix, add_special_tokens=False)
    prefix_length = len(prefix_ids)

    response_features = concatenated_features[prefix_length:]

    if response_features.size(0) == 0:
        warnings.warn("No response tokens detected. Returning a zero vector.")
        avg_feature = torch.zeros(concatenated_features.size(1))
    else:
        avg_feature = torch.mean(response_features, dim=0)

    return avg_feature.cpu().detach().float().numpy()


def extract_logits_info_for_reasoning_path(prompt, instruction, reasoning_steps, separator, tokenizer, model,
                                           to_cpu=True):
    """
    For a given prompt and list of reasoning steps, construct a dialogue, obtain model outputs,
    and extract logits (instead of hidden states). This is useful when hidden states are unavailable.
    """
    full_assistant_answer = separator.join(reasoning_steps)
    prompt = instruction + prompt
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_assistant_answer}
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    full_token_ids = inputs["input_ids"][0].tolist()

    boundaries = []
    conversation_prefix = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]
    conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False).strip()
    if conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
        conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
    if conv.endswith("<|eot_id|>"):
        conv = conv[:-len("<|eot_id|>")]
    prompt_ids = tokenizer.encode(conv, add_special_tokens=False)
    boundaries.append((0, len(prompt_ids)))
    last_length = len(prompt_ids)

    for i in range(len(reasoning_steps)):
        partial_answer = separator.join(reasoning_steps[: i + 1])
        conversation_prefix = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": partial_answer}
        ]
        conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False).strip()
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|eot_id|>"):
            conv = conv[:-len("<|eot_id|>")]
        if (i + 1) != len(reasoning_steps):
            conv += separator
        current_ids = tokenizer.encode(conv, add_special_tokens=False)
        current_length = len(current_ids)
        boundaries.append((last_length, current_length))
        last_length = current_length

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    # Extract the final logits from model outputs (shape: 1 x seq_length x vocab_size)
    logits = outputs.logits
    if to_cpu:
        logits = logits.detach().cpu()

    # Assume segmentation is computed similarly:
    segments = segment_token_ids(full_token_ids, boundaries)

    detailed_info = {
        "conversations": conversation,
        "inputs": {k: v.detach().cpu().tolist() for k, v in inputs.items()},
        "logits": logits,
        "boundaries": boundaries,
        "segments": segments
    }
    return detailed_info


def extract_logits_info_for_reasoning_path_batch(prompts, reasoning_steps_list, instruction, separator,
                                                 tokenizer, model, to_cpu=True):
    """
    Batch version for logits extraction. Constructs dialogues for a batch of prompts and reasoning steps,
    obtains model outputs, and extracts logits.
    """
    batched_input_ids = []
    batched_attention_masks = []
    conversations_batch = []
    full_token_ids_batch = []
    boundaries_batch = []
    batch_size = len(prompts)

    for idx in range(batch_size):
        prompt = prompts[idx]
        reasoning_steps = reasoning_steps_list[idx]

        full_assistant_answer = separator.join(reasoning_steps)
        conversation = [
            {"role": "user", "content": instruction + prompt},
            {"role": "assistant", "content": full_assistant_answer}
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        batched_input_ids.append(input_ids)
        batched_attention_masks.append(attention_mask)
        full_token_ids = input_ids.tolist()
        full_token_ids_batch.append(full_token_ids)
        conversations_batch.append(conversation)

        boundaries = []
        conversation_prefix = [
            {"role": "user", "content": instruction + prompt},
            {"role": "assistant", "content": ""}
        ]
        conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False).strip()
        if conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        if conv.endswith("<|eot_id|>"):
            conv = conv[:-len("<|eot_id|>")]
        prompt_ids = tokenizer.encode(conv, add_special_tokens=False)
        boundaries.append((0, len(prompt_ids)))
        last_length = len(prompt_ids)

        for i in range(len(reasoning_steps)):
            partial_answer = separator.join(reasoning_steps[: i + 1])
            conversation_prefix = [
                {"role": "user", "content": instruction + prompt},
                {"role": "assistant", "content": partial_answer}
            ]
            conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False).strip()
            if (i + 1) != len(reasoning_steps) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
                conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
            if (i + 1) != len(reasoning_steps) and conv.endswith("<|eot_id|>"):
                conv = conv[:-len("<|eot_id|>")]
            if (i + 1) != len(reasoning_steps):
                conv += separator
            current_ids = tokenizer.encode(conv, add_special_tokens=False)
            current_length = len(current_ids)
            boundaries.append((last_length, current_length))
            last_length = current_length

        boundaries_batch.append(boundaries)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_input_ids = pad_sequence(batched_input_ids, batch_first=True, padding_value=pad_token_id)
    padded_attention_mask = pad_sequence(batched_attention_masks, batch_first=True, padding_value=0)

    batched_inputs = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask
    }
    batched_inputs = {k: v.to(model.device) for k, v in batched_inputs.items()}

    with torch.no_grad():
        outputs = model(**batched_inputs, use_cache=False)
    logits = outputs.logits
    if to_cpu:
        logits = logits.detach().cpu()

    detailed_infos = []
    for i in range(batch_size):
        segments = segment_token_ids(full_token_ids_batch[i], boundaries_batch[i])
        sample_inputs = {
            "input_ids": full_token_ids_batch[i],
            "attention_mask": batched_attention_masks[i].tolist()
        }
        detailed_info = {
            "conversations": conversations_batch[i],
            "inputs": sample_inputs,
            "logits": logits[i].unsqueeze(0),  # Extract the i-th sample logits. Shape: (1, seq_length, vocab_size)
            "boundaries": boundaries_batch[i],
            "segments": segments
        }
        detailed_infos.append(detailed_info)

    return detailed_infos


def get_instruction(dataset):
    if dataset in ["math", "gsm8k"]:
        instruction = (
            "Solve the following math problem step-by-step.\n"
            "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        )
    elif dataset == "aqua_rat":
        instruction = (
            "You are given a multiple-choice question with five options (A–E).\n"
            "Solve it step by step, then present only one letter (A-E) in the form \\boxed{Letter}.\n"
            "Remember to output \\boxed{Letter} at the end of your answer or it will be considered incorrect.\n"
        )
    elif dataset == "imbue":
        instruction = (
            "Analyze the following problem step-by-step. The question includes a list of choices. "
            "Select the most appropriate choice from the provided options and output your final answer enclosed within \\boxed{...},\n"
            "ensuring that the content inside \\boxed{...} is valid Python literal syntax.\n"
        )
    elif dataset == "coinflip":
        instruction = (
            "Answer the following Boolean question with detailed reasoning.\n"
            "Explain your reasoning step-by-step and conclude with a final answer as \\boxed{Yes} or \\boxed{No}.\n"
        )
    elif dataset == "hellaswag":
        instruction = (
            "You are given a context and four possible continuations (A–D). Decide which option best continues the context.\n"
            "Explain your reasoning step by step, then present only one letter (A-D) in the form \\boxed{Letter}.\n"
            "Remember to output \\boxed{Letter} at the end of your answer or it will be considered incorrect.\n"
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return instruction
