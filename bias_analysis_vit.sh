#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices=(${GPUS:-0 1 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/gpu_pool.sh"
device="cuda"
# model_id="WinKawaks/vit-small-patch16-224"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32

# Stage flags per experiment (mul=off; isolating log and expdiff):
# GPU 0: standard only (baseline for LN stages)
# GPU 1: log only
# GPU 2: log + expdiff (full SNN LN without mul)
flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 1e-2 --bias-noise-std 1e-2 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 2e-2 --bias-noise-std 2e-2 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 3e-2 --bias-noise-std 3e-2 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 4e-2 --bias-noise-std 4e-2 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 5e-2 --bias-noise-std 5e-2 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 1e-1 --bias-noise-std 1e-1 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --weight-noise-std 3e-1 --bias-noise-std 3e-1 --model_backend ${backend}"
)
expr_names=(
    "weight-bias-std_1e-2"
    "weight-bias-std_2e-2"
    "weight-bias-std_3e-2"
    "weight-bias-std_4e-2"
    "weight-bias-std_5e-2"
    # "weight-bias-std_1e-1"
    # "weight-bias-std_3e-1"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running error analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device}\
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta 400.0"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait