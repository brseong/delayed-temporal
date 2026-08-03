#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

# Fine 11-step noise scan (auto-generated). jitter 1e-6..1e-5, mismatch 1e-5..5e-5,
# hazard 1e-6..5e-05 (bracketing its measured cliff). 34 experiments total.

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/gpu_pool.sh"
device="cuda"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
batch_size=32
theta=2000
scan_logdir="${SCAN_LOGDIR:-scan_logs}"; mkdir -p "$scan_logdir"

flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 1.000e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 1.900e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 2.800e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 3.700e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 4.600e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 5.500e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 6.400e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 7.300e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 8.200e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 9.100e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --jitter-enabled --jitter-mode potential --noise-std 1.000e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 1.000e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 5.900e-06"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 1.080e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 1.570e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 2.060e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 2.550e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 3.040e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 3.530e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 4.020e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 4.510e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --hazard-enabled --hazard-delta-u 5.000e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 1.000e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 1.400e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 1.800e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 2.200e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 2.600e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 3.000e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 3.400e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 3.800e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 4.200e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 4.600e-05"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend spiking --mismatch-enabled --mismatch-theta-std 5.000e-05"
)
expr_names=(
    "noise_off_baseline"
    "A_jitter_1.000e-06"
    "A_jitter_1.900e-06"
    "A_jitter_2.800e-06"
    "A_jitter_3.700e-06"
    "A_jitter_4.600e-06"
    "A_jitter_5.500e-06"
    "A_jitter_6.400e-06"
    "A_jitter_7.300e-06"
    "A_jitter_8.200e-06"
    "A_jitter_9.100e-06"
    "A_jitter_1.000e-05"
    "B_hazard_1.000e-06"
    "B_hazard_5.900e-06"
    "B_hazard_1.080e-05"
    "B_hazard_1.570e-05"
    "B_hazard_2.060e-05"
    "B_hazard_2.550e-05"
    "B_hazard_3.040e-05"
    "B_hazard_3.530e-05"
    "B_hazard_4.020e-05"
    "B_hazard_4.510e-05"
    "B_hazard_5.000e-05"
    "C_mismatch_1.000e-05"
    "C_mismatch_1.400e-05"
    "C_mismatch_1.800e-05"
    "C_mismatch_2.200e-05"
    "C_mismatch_2.600e-05"
    "C_mismatch_3.000e-05"
    "C_mismatch_3.400e-05"
    "C_mismatch_3.800e-05"
    "C_mismatch_4.200e-05"
    "C_mismatch_4.600e-05"
    "C_mismatch_5.000e-05"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running noise scan on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 error_analysis_vit.py \
        --experiment_name scan-${expr_names[$index]} --device ${device} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta ${theta} --quick-test"
    echo $script
    eval "$script" > "${scan_logdir}/${expr_names[$index]}.log" 2>&1 &
    gpu_pool_register $! "$gpu"
done

wait
