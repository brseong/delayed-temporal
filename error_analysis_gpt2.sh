#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

source ./venv/bin/activate
device="cuda"
theta=1000

expr_names=(
    "spiking_attn"
    "sln"
    "smlp"
    # "all"
    # "control"
)

# Ablation flags
flags=(
    "--no-spiking-ln-mul --no-spiking-ln-log --model_backend spiking" # spiking_attn
    "--no-spiking-ln-mul --no-spiking-ln-expdiff --model_backend spiking" # spiking_attn
    "--no-spiking-ln-log --no-spiking-ln-expdiff --model_backend spiking" # spiking_attn
    # "--no-spiking-attention --spiking-layernorm --no-spiking-mlp --model_backend spiking" # sln
    # "--no-spiking-attention --no-spiking-layernorm --spiking-mlp --model_backend spiking" # smlp
    # "--spiking-attention --spiking-layernorm --spiking-mlp --model_backend spiking --model_backend spiking"       # all
    # "--no-spiking-attention --no-spiking-layernorm --no-spiking-mlp --activation gelu --model_backend hf" # control (ANN only)
)

cuda_devices=(0 1 2) # Adjust if you want to run on different GPUs

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_gpt2.py \
        --experiment_name ${expr_names[$index]} --device ${device} \
        --batch_size 8 \
        ${flags[$index]} --theta ${theta}"
    if [[ -n "${dataset_config_name}" ]]; then
        script+=" --dataset_config_name ${dataset_config_name}"
    fi
    echo $script
    eval $script &
done

wait
