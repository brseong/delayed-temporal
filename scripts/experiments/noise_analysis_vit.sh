#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

# Neuromorphic noise-model sweep. Each experiment isolates one maintained
# component so its accuracy effect remains attributable: Gaussian spike-time
# noise or static threshold mismatch. Every evaluator flag is recorded in the
# W&B config through cfg = vars(args).
#
# Gaussian magnitudes are dimensionless fractions of the identity encoder's
# [0, 2 * theta] coding window. The evaluator converts each fraction into one
# absolute sigma_t and applies it at every encoder boundary. Static mismatch
# remains a separate parameter-variation axis and is never combined here.

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
batch_size=32
theta=2000
time_noise_seed="${TIME_NOISE_SEED:-0}"

base="--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"
gaussian_base="${base} --gaussian-time-noise --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}"

flags=(
    "${base}"
    "${gaussian_base} --time-noise-std-frac 1e-6"
    "${gaussian_base} --time-noise-std-frac 2e-6"
    "${gaussian_base} --time-noise-std-frac 5e-6"
    "${gaussian_base} --time-noise-std-frac 1e-5"
    "${base} --mismatch-enabled --mismatch-theta-std 1e-5"
    "${base} --mismatch-enabled --mismatch-theta-std 3e-5"
    "${base} --mismatch-enabled --mismatch-theta-std 5e-5"
)
expr_names=(
    "noise_off_baseline"
    "A_gaussian_frac_1e-6"
    "A_gaussian_frac_2e-6"
    "A_gaussian_frac_5e-6"
    "A_gaussian_frac_1e-5"
    "B_mismatch_1e-5"
    "B_mismatch_3e-5"
    "B_mismatch_5e-5"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire                       # blocks until a GPU is free (assigns to whichever frees first)
    gpu=$GPU_POOL_ACQUIRED
    echo "Running noise analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
        --experiment_name noise-${expr_names[$index]} --device ${device} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta ${theta} --quick-test"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
