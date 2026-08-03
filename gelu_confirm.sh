#!/bin/bash
# Confirm the spiking GELU is the jitter culprit: jitter sweep with the spiking GELU
# approximation vs exact PyTorch GELU (--spiking-mlp-exact-gelu). If GELU is the cause, the
# exact-GELU curve should stay flat while the approx curve falls off the cliff.
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices=(${GPUS:-4 5 6 7})
source "$(dirname "${BASH_SOURCE[0]}")/gpu_pool.sh"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
theta=2000; batches=60; bs=32
logdir="gelu_confirm_logs"; rm -rf "$logdir"; mkdir -p "$logdir"

base="--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking"
sigmas=(0 5e-6 8e-6 1.1e-5 1.5e-5 2.5e-5)

gpu_pool_init "${cuda_devices[@]}"
for cfg in approx exact; do
  extra=""; [ "$cfg" = "exact" ] && extra="--spiking-mlp-exact-gelu"
  for s in "${sigmas[@]}"; do
    jit="--jitter-enabled --noise-std $s"; [ "$s" = "0" ] && jit=""
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "GPU $gpu: cfg=$cfg sigma=$s"
    CUDA_VISIBLE_DEVICES=$gpu python3 error_analysis_vit.py \
      --experiment_name gelu_${cfg}_${s} --model_backend spiking \
      --model_id "$model_id" --dataset_id imagenet-1k --batch_size $bs --theta $theta \
      $base $extra $jit --max_eval_batches $batches \
      > "$logdir/${cfg}_${s}.log" 2>&1 &
    gpu_pool_register $! "$gpu"
  done
done
wait
echo "=== results ==="
for cfg in approx exact; do
  for s in "${sigmas[@]}"; do
    acc=$(grep -E "^Accuracy" "$logdir/${cfg}_${s}.log" | awk '{print $2}')
    echo "$cfg sigma=$s -> $acc"
  done
done
