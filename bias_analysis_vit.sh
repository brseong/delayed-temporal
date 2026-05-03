#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(0)
cuda_devices=(5 6 7)
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
ln_flags=(
    # "--no-spiking-ln-mul --no-spiking-ln-log --no-spiking-ln-expdiff"
    # "--no-spiking-ln-mul --spiking-ln-log --no-spiking-ln-expdiff"
    # "--no-spiking-ln-mul --spiking-ln-log --spiking-ln-expdiff"
    ""
)
flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 5e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-4 --model_backend ${backend}"
    # "--no-spiking-layernorm --no-spiking-mlp --spiking-attention --noise-std 1e-4"
    # "--no-spiking-layernorm --spiking-mlp --no-spiking-attention --noise-std 1e-4"
    # "--spiking-layernorm --no-spiking-mlp --no-spiking-attention --noise-std 1e-4"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-4 --model_backend ${backend}"
    # "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-3 --model_backend ${backend}"
    # "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-3"
    # "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-2"
    # "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-2"
)
expr_names=(
    "std_5e-4"
    "std_1e-5"
    "std_3e-5"
    "std_1e-4"
    "std_3e-4"
    # "std_1e-3"
    # "std_5e-3"
    # "std_1e-2"
    # "std_5e-2"
)

for index in "${indices[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device}\
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} ${ln_flags[$index]} --theta 400.0"
    echo $script
    eval $script &
done

wait