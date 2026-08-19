#!/bin/bash
# Test whether the spiking GELU approximation amplifies Gaussian spike-time
# noise. The approximate implementation is compared with exact PyTorch GELU
# (--spiking-mlp-exact-gelu) over the same seeded timing-noise fractions.
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-4 5 6 7})
source "$repo_root/scripts/lib/gpu_pool.sh"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
theta=2000; batches=60; bs=32
time_noise_seed="${TIME_NOISE_SEED:-0}"
logdir="${GELU_CONFIRM_LOGDIR:-$repo_root/artifacts/logs/gelu_confirm}"
mkdir -p "$logdir"
find "$logdir" -maxdepth 1 -type f -name '*.log' -delete

base_args=(
  --spiking-layernorm
  --spiking-mlp
  --spiking-attention
  --model_backend spiking
)

# Each value is a fraction of the identity encoder's [0, 2 * theta] window.
# The evaluator computes the absolute sigma_t, ensuring both GELU variants
# receive the same physical timing-noise scale.
time_noise_std_fracs=(0 5e-6 8e-6 1.1e-5 1.5e-5 2.5e-5)

gpu_pool_init "${cuda_devices[@]}"
for cfg in approx exact; do
  extra_args=()
  [[ "$cfg" == "exact" ]] && extra_args+=(--spiking-mlp-exact-gelu)

  for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
    # The zero entry exercises the ordinary noise-off path. Positive entries
    # explicitly enable Gaussian sampling while sharing one run-wide seed.
    time_noise_args=(
      --time-noise-std-frac "$time_noise_std_frac"
      --time-noise-mean 0.0
      --time-noise-seed "$time_noise_seed"
    )
    [[ "$time_noise_std_frac" != "0" ]] && time_noise_args+=(--gaussian-time-noise)

    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "GPU $gpu: cfg=$cfg time_noise_std_frac=$time_noise_std_frac"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/evaluation/error_analysis_vit.py \
      --experiment_name "gelu_${cfg}_frac_${time_noise_std_frac}" \
      --model_id "$model_id" --dataset_id imagenet-1k --batch_size $bs --theta $theta \
      "${base_args[@]}" "${extra_args[@]}" "${time_noise_args[@]}" \
      --max_eval_batches $batches \
      > "$logdir/${cfg}_${time_noise_std_frac}.log" 2>&1 &
    gpu_pool_register $! "$gpu"
  done
done
wait
echo "=== results ==="
for cfg in approx exact; do
  for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
    acc=$(grep -E "^Accuracy" "$logdir/${cfg}_${time_noise_std_frac}.log" | awk '{print $2}')
    echo "$cfg time_noise_std_frac=$time_noise_std_frac -> $acc"
  done
done
