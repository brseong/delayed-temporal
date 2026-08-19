#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32   # per-GPU (was 32*8 for 8-way DataParallel; now one job per GPU via the pool)
time_noise_seed="${TIME_NOISE_SEED:-0}"

# These fractions stay fixed while theta changes. Each evaluator invocation
# therefore derives sigma_t = fraction * (2 * theta), preserving the intended
# relative noise level for every coding-window size in the sweep.
time_noise_std_fracs=(0 1e-5 2e-5 3e-5 4e-5)
thetas=(1000 500 200 100 50 20 10 5 2 1)

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
        script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
            --experiment_name ${expr_name}_theta_${theta} --device ${device} \
            --model_id ${model_id} --dataset_id ${dataset_id} \
            --batch_size ${batch_size} \
            --spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend} \
            --theta ${theta} --quick-test ${gaussian_flag} \
            --time-noise-std-frac ${time_noise_std_frac} \
            --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}"
        echo $script
        eval $script &
        gpu_pool_register $! "$gpu"
    done
done

wait
