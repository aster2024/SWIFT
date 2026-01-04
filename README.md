# SWIFT: Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling (KDD 2026)

This repository is the official code release for **SWIFT** (*Simple Weighted Intrinsic Feedback Technique*), introduced in our KDD 2026 paper.

## Paper

Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling

- Status: **Accepted to KDD 2026 (Research Track)**
- Paper: [https://arxiv.org/abs/2505.12225](https://arxiv.org/abs/2505.12225)
- Slides (promo): [Google Slides](https://docs.google.com/presentation/d/1MtUwGVC2xGDMu0b1TuJAujmfjxFn_TGt/edit?usp=sharing&ouid=101998927371723280988&rtpof=true&sd=true)

https://github.com/user-attachments/assets/8be2008d-51de-456f-b785-fefbb66f325a

SWIFT learns a reward function directly from a task-performing LLM’s **intrinsic signals** (hidden states or logits), enabling efficient Best-of-$N$ selection without a massive text-based reward model.

![SWIFT vs Traditional RM](Figures/overall_illustration.png)

## Abstract

Best-of-N sampling is a powerful method for improving Large Language Model (LLM) performance, but it is often limited by its dependence on massive, text-based reward models. These models are not only computationally expensive but also data-hungry, requiring extensive labeled datasets for training. This creates a significant data challenge, as they overlook a rich, readily available data source: the LLM's own internal hidden states. To address this data and efficiency gap, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel and lightweight method that learns a reward function directly from the rich information embedded in LLM hidden states. Operating at the token embedding level, SWIFT employs simple linear layers to effectively distinguish between preferred and dispreferred generations, eliminating the need for computationally intensive text-based modeling. Extensive experiments on standard benchmarks show that SWIFT outperforms existing baselines (12.7\% higher accuracy than EurusRM-7B on MATH dataset) while using less than 0.005\% of their parameters. Its robust scalability, compatibility with certain closed-source models via logit access, and ability to combine with traditional reward models for additional performance highlight SWIFT's practical value and contribution to more efficient data-driven LLM post-training.

## Why SWIFT (motivation)

Best-of-$N$ sampling relies on a reward model (RM) to pick the best response from a set of candidates. Conventional RMs are often:

- Massive: frequently a fine-tuned LLM with billions of parameters.
- Costly: both training and inference are heavy in GPU time/memory.
- Data-hungry: they depend on large-scale preference data.

SWIFT bypasses these issues by mining intrinsic signals from the task LLM itself, rather than modeling reward from the final text.

## Method overview

At a high level:

1. For each generated token, extract hidden states from selected transformer layers (or use final logits in `--logits_mode`).
2. Feed token features into a lightweight linear reward model with an optional gating mechanism.
3. Aggregate token-level scores into a single reward for each candidate response, then select the best candidate in Best-of-$N$.

## Key features

- Extreme efficiency: orders of magnitude smaller than LLM-based RMs.
- Strong performance: competitive or better Best-of-$N$ accuracy on standard benchmarks.
- Data efficiency: works well even with a few thousand training samples.
- Flexible signals: supports hidden states (open models) and logits-only training/eval (restricted/closed settings).

## Headline results (example: MATH Best-of-$N$ @64)

| Reward Model        | Llama-3.2-3B | Llama-3.1-8B | Ministral-8B | Avg. |
| ------------------- | ------------ | ------------ | ------------ | ---- |
| Eurus-7B            | 46.8         | 52.2         | 55.0         | 51.0 |
| Skywork-Llama3.1-8B | 48.8         | 53.4         | 61.6         | 52.9 |
| Starling-7B         | 39.8         | 49.0         | 47.0         | 46.7 |
| Ultra-13B           | 44.4         | 50.4         | 54.0         | 50.1 |
| RLHFlow-8B-Deepseek | 47.6         | 49.8         | 57.8         | 51.1 |
| Math-Shepherd-7B    | 43.6         | 49.0         | 54.8         | 49.8 |
| SWIFT (ours)        | 53.6         | 62.6         | 62.8         | 57.5 |

![Efficiency Comparison](Figures/efficiency.png)

## Repository layout

- `generate/`: generate candidate rollouts (reasoning paths) from task-performing LMs
- `preprocess/`: download / preprocess datasets (we only ship MATH in-repo to keep size small)
- `train/`: train SWIFT reward models from intrinsic features
- `eval/`: score candidates with SWIFT / baselines and run Best-of-$N$ evaluation
- `script/train_and_eval.sh`: end-to-end reproduction script

## Where to start

If you are new to this codebase, these three entry points cover the full pipeline:

- `script/train_and_eval.sh`: end-to-end reproduction (generate rollouts → baselines → train SWIFT → score → Best-of-$N$ eval).
- `train/extract_train.py`: trains the SWIFT reward model from intrinsic features; supports `--memory_efficient`, `--layers`, and `--logits_mode`.
- `eval/get_rewards.py`: loads a trained SWIFT checkpoint and computes per-candidate rewards for downstream Best-of-$N$ evaluation.

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:

- Some dependencies (e.g., `flash_attn`, `vllm`, CUDA-related libs) may require a matching CUDA toolchain and a compatible PyTorch build.
- The main pipeline assumes GPU availability.

### 2) Prepare datasets

- **MATH** is already provided under `data/`.
- For other datasets, use the scripts in `preprocess/` (e.g., `preprocess/process_gsm8k.py`, `preprocess/process_hellaswag.py`, ...).

## Quickstart (reproduce main results)

The simplest way is to run the provided end-to-end script:

```bash
bash script/train_and_eval.sh
```

This script will:

1. Generate training/test rollouts for multiple task LMs.
2. Score rollouts with several open-source reward-model baselines.
3. Train SWIFT.
4. Score rollouts with SWIFT.
5. Compute Best-of-$N$ accuracy curves under `eval_results/`.

## Step-by-step pipeline

Below is the minimal structure of the workflow. See `script/train_and_eval.sh` for a complete set of models/baselines and logging.

### A) Generate candidate rollouts

Example for MATH:

```bash
python generate/generate_math.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --split train \
  --batch_size 16 \
  --output_file data/math/train_math_Llama-3.1-8B-Instruct.json
```

### B) Train SWIFT

```bash
python train/extract_train.py \
  --dataset math \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --max_samples 6000 \
  --methods ce \
  --output_model_file model/math/reward_model_ce-llama3-8b-6000.pt \
  --reward_batch_size 16 \
  --memory_efficient
```

### C) Score rollouts with SWIFT

```bash
python eval/get_rewards.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset math \
  --dataset_file data/math/test_math_Llama-3.1-8B-Instruct.json \
  --reward_model_load model/math/reward_model_ce-llama3-8b-6000.pt \
  --output_file data/math/extracted_rewards_ce-llama3-8b-6000.json
```

### D) Best-of-$N$ evaluation

```bash
python eval/bon_eval.py \
  --dataset_file data/math/extracted_rewards_baseline_Eurus-llama3-8b.jsonl \
  --reward_file data/math/extracted_rewards_ce-llama3-8b-6000.json \
  --k_vals 1 2 4 8 16 32 64 \
  --output_file eval_results/math/bon_eval_Eurus-ce-llama3-8b-6000.json
```

## Key options

SWIFT is intentionally simple, but the code exposes several knobs that are useful for ablations and practical usage:

- `--memory_efficient` (training): reduce peak memory by extracting intrinsic features **on-the-fly** per batch.
- `--layers` (training & scoring): select a subset of hidden layers to extract.
- `--logits_mode` (training & scoring): use final output logits instead of hidden states (useful when hidden states are inaccessible).
- `--apply_norm`: apply normalization to hidden states/logits before reward computation.
- `--disable_gate`: disable the token-level gating mechanism.

## Implementation notes (important for reproducibility)

### Memory-efficient feature extraction

During SWIFT reward model training, enabling `--memory_efficient` avoids caching all hidden states for all candidates in RAM.
Instead, the dataloader collate function dynamically feeds each candidate’s **original prompt** and its **reasoning steps** back into the task LLM to extract the intrinsic features (hidden states or logits) for that batch.

This choice favors research simplicity and avoids memory blow-ups, but it can be slower.

**Engineering note:** a potentially better production implementation is to **precompute** and **store** candidate features (or hidden states) on disk (e.g., memory-mapped tensors / chunked files) and load them lazily during training.

### Using only a subset of layers

We provide `--layers` to train/score SWIFT with only selected transformer layers.
In our experiments, using only a small set (e.g., the **last 4 layers**) can already yield strong performance, while further reducing compute and feature dimensionality.

Example:

```bash
python train/extract_train.py ... --layers 28 29 30 31
python eval/get_rewards.py ... --layers 28 29 30 31
```

## Results

![Efficiency Comparison](Figures/efficiency.png)

## Citation

If you find SWIFT useful, please cite:

```bibtex
@misc{guo2025mining,
  title={Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling},
  author={Jizhou Guo and Zhaomin Wu and Hanchen Yang and Philip S. Yu},
  year={2025},
  eprint={2505.12225},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.12225}
}
```
