#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

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
source "$repo_root/scripts/lib/gpu_pool.sh"
backend="hf"
batch_size=$((32 * 4)) # Adjust based on the number of GPUs and memory constraints
time_noise_seed="${TIME_NOISE_SEED:-0}"

# Fractions are measured against the identity encoder's [0, 2 * theta]
# coding window. The evaluator converts each fraction into one absolute
# spike-time standard deviation shared by all encoders.
time_noise_std_fracs=(0 1e-5 2e-5 3e-5)

gpu_pool_init "${cuda_devices[@]}"
for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
    expr_name="std_frac_${time_noise_std_frac}"
    gaussian_flag=""
    if [[ "${time_noise_std_frac}" != "0" ]]; then
        gaussian_flag="--gaussian-time-noise"
    fi

    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running Gaussian time-noise analysis on GPU ${gpu}: ${expr_name}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_bert.py \
        --experiment_name ${expr_name}_${task} --device ${device} \
        --task ${task} \
        --model_id ${model_id} \
        --dataset_name ${dataset_name} --dataset_split ${dataset_split} \
        --spiking-layernorm --spiking-mlp --spiking-attention \
        --model_backend ${backend} --theta ${theta} --activation ${activation} \
        ${gaussian_flag} --time-noise-std-frac ${time_noise_std_frac} \
        --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}"
    if [[ -n "${dataset_config_name}" ]]; then
        script+=" --dataset_config_name ${dataset_config_name}"
    fi
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
