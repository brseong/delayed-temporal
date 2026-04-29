#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

source ./venv/bin/activate
device="cuda"
theta=400
model_backend="hf" # Currently mostly testing HF backend with optional spiking attention

expr_names=(
    "llama3_hf"
    # "llama3_spiking_attn"
)

# Ablation flags
flags=(
    "--no-spiking-attention" # llama3_hf
    # "--spiking-attention"    # llama3_spiking_attn
)

cuda_devices=(0 1 2 3) # Adjust if you want to run on different GPUs

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_llama3.py \
        --experiment_name ${expr_names[$index]} --device ${device} \
        --model_backend ${model_backend} --batch_size 2 \
        ${flags[$index]} --theta ${theta}"
    if [[ -n "${dataset_config_name}" ]]; then
        script+=" --dataset_config_name ${dataset_config_name}"
    fi
    echo $script
    eval $script &
done

wait
