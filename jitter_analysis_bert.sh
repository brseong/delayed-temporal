#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

source ./venv/bin/activate
device="cuda"
theta=1000
activation="relu"  # relu | gelu
task="sst2"  # sst2 | agnews | imdb
model_id="textattack/bert-base-uncased-SST-2"
dataset_name="glue"
dataset_config_name="sst2"
dataset_split="validation"

cuda_devices=(${GPUS:-0 1 2 3})   # override with e.g. GPUS="4 5 6 7"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_pool.sh"
backend="hf"
batch_size=$((32 * 4)) # Adjust based on the number of GPUs and memory constraints

stds=(0 1e-5 2e-5 3e-5) # Add more std values as needed

flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --spike_time_noise_std ${stds[0]} --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --spike_time_noise_std ${stds[1]} --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --spike_time_noise_std ${stds[2]} --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --spike_time_noise_std ${stds[3]} --model_backend ${backend}"
)
expr_names=(
    "std_${stds[0]}"
    "std_${stds[1]}"
    "std_${stds[2]}"
    "std_${stds[3]}"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running error analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 error_analysis_bert.py \
        --experiment_name ${expr_names[$index]}_${task} --device ${device} \
        --task ${task} \
        --model_id ${model_id} \
        --dataset_name ${dataset_name} --dataset_split ${dataset_split} \
        ${flags[$index]} --theta ${theta} --activation ${activation}"
    if [[ -n "${dataset_config_name}" ]]; then
        script+=" --dataset_config_name ${dataset_config_name}"
    fi
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait