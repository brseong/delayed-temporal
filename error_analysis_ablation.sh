#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

source ./venv/bin/activate
device="cuda"
theta=400

expr_names=(
    "spiking_attn"
    "sln"
    "smlp"
    "all"
    "control"
)

# Ablation flags
flags=(
    "--spiking-attention --no-spiking-layernorm --no-spiking-mlp" # spiking_attn
    "--no-spiking-attention --spiking-layernorm --no-spiking-mlp" # sln
    "--no-spiking-attention --no-spiking-layernorm --spiking-mlp" # smlp
    "--spiking-attention --spiking-layernorm --spiking-mlp"       # all
    "--no-spiking-attention --no-spiking-layernorm --no-spiking-mlp --activation gelu" # control (ANN only)
)

cuda_devices=(0 1 2 3 7) # Adjust if you want to run on different GPUs

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device} \
        ${flags[$index]} --theta ${theta}"
    echo $script
    eval $script &
done

wait
