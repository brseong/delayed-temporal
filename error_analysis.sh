#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(0)
sln="spiking-layernorm"
sa="spiking-attention"
cuda_devices=(1)
source ./venv/bin/activate
device="cuda"

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
expr_names=(
    "PWM"
)

for index in "${indices[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device} \
        --${sln} --${sa} ${ln_flags[$index]} --theta 400 --spiking-mlp --activation relu"
    echo $script
    eval $script
done

wait