#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices="0,4,5,6,7"
device="cuda"
batch_size=80 # 16 * 4 (기존 vit_base 배치 사이즈)
model_backend="spiking"
dataset_id="imagenet-1k"

# ViT-Base 모델만 사용
model_id="/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"

# 1. HF Baseline (이상적인 원본 GELU 및 모든 연산)
# 2. Spiking Model + Spiking MLP (GELU 근사 함수 사용)
# 3. Spiking Model + No Spiking MLP (Spiking 전파 과정 중 GELU만 exact Pytorch GELU 사용)
flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    "--spiking-layernorm --spiking-mlp --spiking-mlp-exact-gelu --spiking-attention --model_backend ${model_backend} --precision float32"
)
expr_names=(
    "spiking-approx-gelu-cubic-tanh"
    "gelu-cubic-tanh"
)

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=$cuda_devices python3 error_analysis_vit.py \
        --experiment_name vit_base-${expr_names[$index]} --device ${device} \
        --batch_size ${batch_size} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        ${flags[$index]} --theta 2000"
    echo $script
    eval $script
done

wait
