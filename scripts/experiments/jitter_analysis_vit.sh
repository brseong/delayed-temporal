#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-1 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
# model_id="WinKawaks/vit-small-patch16-224"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32
theta=400.0
time_noise_seed="${TIME_NOISE_SEED:-0}"

# Each value is a fraction of the identity encoder's full [0, 2 * theta]
# coding window. The evaluator converts it to one absolute spike-time sigma
# and applies that same sigma to every encoder in the model.
time_noise_std_fracs=(
    1e-5
    2e-5
    3e-5
    4e-5
    5e-5
    1e-4
    3e-4
    1e-3
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!time_noise_std_fracs[@]}"; do
    time_noise_std_frac="${time_noise_std_fracs[$index]}"
    expr_name="std_frac_${time_noise_std_frac}"
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running Gaussian time-noise analysis on GPU ${gpu}: ${expr_name}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
        --experiment_name ${expr_name} --device ${device} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} --theta ${theta} \
        --spiking-layernorm --spiking-mlp --spiking-attention \
        --model_backend ${backend} --gaussian-time-noise \
        --time-noise-std-frac ${time_noise_std_frac} \
        --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
