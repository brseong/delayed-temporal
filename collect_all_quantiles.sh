#!/bin/bash

# BERT Quantiles
for task in sst2 agnews imdb; do
    echo "Collecting quantiles for BERT - $task..."
    python error_analysis_bert.py \
        --task $task \
        --model_backend hf \
        --max_eval_batches 10 \
        --collect-quantiles \
        --experiment_name quantile_bert_$task
done

# GPT2 Quantiles
echo "Collecting quantiles for GPT2..."
python error_analysis_gpt2.py \
    --model_backend hf \
    --max_eval_batches 10 \
    --collect-quantiles \
    --experiment_name quantile_gpt2

# ViT Quantiles
# Small, Base, Large
models=("google/vit-small-patch16-224" "google/vit-base-patch16-224" "google/vit-large-patch16-224")
for model in "${models[@]}"; do
    model_name=$(echo $model | sed 's/\//_/g')
    echo "Collecting quantiles for ViT - $model..."
    python error_analysis_vit.py \
        --model_id "$model" \
        --dataset_id cifar10 \
        --model_backend hf \
        --max_eval_batches 10 \
        --collect-quantiles \
        --experiment_name quantile_vit_$model_name
done

echo "Quantile collection complete."
