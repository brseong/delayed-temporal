#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(3)
cuda_devices=(0 1 2 3)
source ./venv/bin/activate
device="cuda"
model_id="WinKawaks/vit-small-patch16-224"
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
    "weight-bias-std_1e-1"
    "weight-bias-std_3e-1"
)

for index in "${indices[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device}\
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta 400.0"
    echo $script
    eval $script &
done

wait