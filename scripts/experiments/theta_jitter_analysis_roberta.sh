#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
model_id="Bhumika/roberta-base-finetuned-sst2"
backend="spiking"
batch_size=16   # per-GPU (was 16*8 for 8-way DataParallel; now one job per GPU via the pool)
dataset_name="glue"
dataset_config_name="sst2"
dataset_split="validation"
task="sst2"
time_noise_seed="${TIME_NOISE_SEED:-0}"

# These fractions stay fixed while theta changes. The evaluator converts each
# value with sigma_t = fraction * (2 * theta), so every row represents the
# same relative timing perturbation across coding-window sizes.
time_noise_std_fracs=(0 1e-5 2e-5 3e-5 4e-5)
thetas=(2000 1000 500 250)

gpu_pool_init "${cuda_devices[@]}"
for theta in "${thetas[@]}"; do
    for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
        expr_name="std_frac_${time_noise_std_frac}"
        gaussian_flag=""
        if [[ "${time_noise_std_frac}" != "0" ]]; then
            gaussian_flag="--gaussian-time-noise"
        fi

        gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
        echo "Running Gaussian time-noise analysis on GPU ${gpu}: ${expr_name} with theta ${theta}"
        script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_roberta.py \
            --experiment_name roberta_${expr_name}_${task}_theta_${theta} --device ${device} \
            --task ${task} --batch_size ${batch_size} \
            --model_id ${model_id} \
            --dataset_name ${dataset_name} --dataset_split ${dataset_split} \
            --spiking-layernorm --spiking-mlp --spiking-attention \
            --model_backend ${backend} --theta ${theta} ${gaussian_flag} \
            --time-noise-std-frac ${time_noise_std_frac} \
            --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}"
        if [[ -n "${dataset_config_name}" ]]; then
            script+=" --dataset_config_name ${dataset_config_name}"
        fi
        echo $script
        eval $script &
        gpu_pool_register $! "$gpu"
    done
done

wait
