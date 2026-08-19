#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

# Fine noise scan with two independent maintained axes. Eleven Gaussian
# spike-time fractions cover 1e-6..1e-5, and eleven static threshold-mismatch
# values cover 1e-5..5e-5. Including the noise-off baseline gives 23 runs.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
batch_size=32
theta=2000
time_noise_seed="${TIME_NOISE_SEED:-0}"
scan_logdir="${SCAN_LOGDIR:-$repo_root/artifacts/logs/noise_scan}"
mkdir -p "$scan_logdir"

time_noise_std_fracs=(
    1.000e-06 1.900e-06 2.800e-06 3.700e-06 4.600e-06 5.500e-06
    6.400e-06 7.300e-06 8.200e-06 9.100e-06 1.000e-05
)
mismatch_theta_stds=(
    1.000e-05 1.400e-05 1.800e-05 2.200e-05 2.600e-05 3.000e-05
    3.400e-05 3.800e-05 4.200e-05 4.600e-05 5.000e-05
)

base="--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking"
flags=("${base}")
expr_names=("noise_off_baseline")

# Gaussian timing noise is sampled at encoder boundaries. The evaluator maps
# each dimensionless fraction to sigma_t = fraction * (2 * theta), while one
# run-wide seed makes the scan reproducible.
for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
    flags+=("${base} --gaussian-time-noise --time-noise-std-frac ${time_noise_std_frac} --time-noise-mean 0.0 --time-noise-seed ${time_noise_seed}")
    expr_names+=("A_gaussian_frac_${time_noise_std_frac}")
done

# Threshold mismatch is static device variation, not event-time noise. These
# runs deliberately omit every Gaussian option so the two effects stay
# independently attributable.
for mismatch_theta_std in "${mismatch_theta_stds[@]}"; do
    flags+=("${base} --mismatch-enabled --mismatch-theta-std ${mismatch_theta_std}")
    expr_names+=("B_mismatch_${mismatch_theta_std}")
done

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running noise scan on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
        --experiment_name scan-${expr_names[$index]} --device ${device} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta ${theta} --quick-test"
    echo $script
    eval "$script" > "${scan_logdir}/${expr_names[$index]}.log" 2>&1 &
    gpu_pool_register $! "$gpu"
done

wait
