#!/bin/bash
set -euo pipefail

declare -A model_name_map=(
    ["llama3-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama3-3b"]="meta-llama/Llama-3.2-3B-Instruct"
    ["ministral-8b"]="mistralai/Ministral-8B-Instruct-2410"
)

declare -A baseline_name_map=(
    ["Eurus"]="openbmb/Eurus-RM-7b"
    ["Skywork"]="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    ["Ultra"]="openbmb/UltraRM-13b"
    ["Starling"]="berkeley-nest/Starling-RM-7B-alpha"
    ["Deepseek"]="RLHFlow/Llama3.1-8B-ORM-Deepseek-Data"
    ["Shepherd"]="peiyi9979/math-shepherd-mistral-7b-prm"
)

declare -A bs_map=(
    ["llama3-8b"]=16
    ["llama3-3b"]=16
    ["ministral-8b"]=16
)

cuda_device=0
dataset="math"
method="ce"
max_samples=6000

mkdir -p model/$dataset
mkdir -p log/$dataset
mkdir -p eval_results/$dataset
mkdir -p data/$dataset

for model in "llama3-3b" "llama3-8b" "ministral-8b"; do
    model_name=${model_name_map[${model}]}
    batch_size=${bs_map[${model}]}
    echo "Generating training data for $model"
    output_file="data/$dataset/train_${dataset}_$(echo "${model_name}" | awk -F'/' '{print $NF}').json"
    CUDA_VISIBLE_DEVICES=$cuda_device python generate/generate_$dataset.py --model_name $model_name --split train --batch_size $batch_size --output_file ${output_file} > log/$dataset/generate_train_$model.log
done

for model in "llama3-3b" "llama3-8b" "ministral-8b"; do
    model_name=${model_name_map[${model}]}
    batch_size=${bs_map[${model}]}
    echo "Generating test data for $model"
    output_file="data/$dataset/test_${dataset}_$(echo "${model_name}" | awk -F'/' '{print $NF}').json"
    CUDA_VISIBLE_DEVICES=$cuda_device python generate/generate_$dataset.py --model_name $model_name --split test --batch_size $batch_size --output_file ${output_file} > log/$dataset/generate_test_$model.log
done

for model in "llama3-3b" "llama3-8b" "ministral-8b"; do
    model_name=${model_name_map[${model}]}
    batch_size=${bs_map[${model}]}

    input_file=data/$dataset/test_${dataset}_$(echo "$model_name" | awk -F'/' '{print $NF}').json
    for baseline in "Eurus" "Skywork" "Ultra" "Starling" "Deepseek" "Shepherd"; do
      baseline_name=${baseline_name_map[${baseline}]}
      echo "Getting baseline rewards for $input_file by $baseline"
      CUDA_VISIBLE_DEVICES=$cuda_device python eval/get_baseline_rewards.py --dataset $dataset --reward_name_or_path $baseline_name --input_file $input_file --model_type $baseline --output_file data/$dataset/extracted_rewards_baseline_$baseline-$model.jsonl > log/$dataset/get_baseline_rewards_baseline_$baseline-$model.log
    done


    baseline="Eurus"
    baseline_name=${baseline_name_map[${baseline}]}
    train_file=data/$dataset/train_${dataset}_$(echo "$model_name" | awk -F'/' '{print $NF}').json
    eval_file=data/$dataset/test_${dataset}_$(echo "$model_name" | awk -F'/' '{print $NF}').json
    output_dir=model/$dataset/fine_tune_baseline_$baseline-$model
    mkdir -p $output_dir
    echo "Fine-tuning baseline reward model for $model_name with method $method and max_samples $max_samples"
    CUDA_VISIBLE_DEVICES=$cuda_device python eval/fine_tune_baseline.py --dataset $dataset --reward_name_or_path $baseline_name --train_file $train_file --eval_file $eval_file --output_dir $output_dir --output_file data/$dataset/extracted_rewards_fine_tune_$baseline-$model.jsonl --do_train --do_eval --batch_size $batch_size --max_train_samples $max_samples > log/$dataset/get_baseline_rewards_fine_tune_$baseline-$model.log

    echo "Training reward model for $model_name with method $method and max_samples $max_samples"
    CUDA_VISIBLE_DEVICES=$cuda_device python train/extract_train.py --dataset $dataset --model_name $model_name --max_samples $max_samples --methods $method --output_model_file model/$dataset/reward_model_$method-$model-${max_samples}.pt --reward_batch_size $batch_size --memory_efficient > log/$dataset/train_$method-$model-${max_samples}.log

    echo "Getting rewards for $model_name with method $method and max_samples $max_samples"
    CUDA_VISIBLE_DEVICES=$cuda_device python eval/get_rewards.py --model_name $model_name --dataset $dataset --reward_model_load model/$dataset/reward_model_$method-$model-${max_samples}.pt --output_file data/$dataset/extracted_rewards_$method-$model-${max_samples}.json > log/$dataset/get_rewards_$method-$model-${max_samples}.log

    echo "Evaluating rewards"
    for baseline in "Eurus" "Skywork" "Ultra" "Starling" "Deepseek" "Shepherd"; do
      baseline_name=${baseline_name_map[${baseline}]}
      CUDA_VISIBLE_DEVICES=$cuda_device python eval/bon_eval.py --dataset_file data/$dataset/extracted_rewards_baseline_$baseline-$model.jsonl --reward_file data/$dataset/extracted_rewards_$method-$model-${max_samples}.json --k_vals 1 2 4 8 16 32 64 --output_file eval_results/$dataset/bon_eval_$baseline-$method-$model-${max_samples}.json > log/$dataset/bon_eval_$baseline-$method-$model-${max_samples}.log
    done
    for baseline in "Eurus"; do
      CUDA_VISIBLE_DEVICES=$cuda_device python eval/bon_eval.py --dataset_file data/$dataset/extracted_rewards_fine_tune_$baseline-$model.jsonl --reward_file data/$dataset/extracted_rewards_$method-$model-${max_samples}.json --k_vals 1 2 4 8 16 32 64 --output_file eval_results/$dataset/bon_eval_fine_tune_$baseline-$method-$model-${max_samples}.json > log/$dataset/bon_eval_fine_tune_$baseline-$method-$model-${max_samples}.log
    done

done
