#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

# Neuromorphic noise-model sweep. Each experiment isolates ONE component so its effect on
# accuracy is attributable. All noise logic lives in utils/transforms/noise.py; the three
# components (A jitter, B escape hazard, C device mismatch) are independently toggleable and
# every flag is recorded in the wandb config (cfg = vars(args)).
#
# Magnitudes are small on purpose: the decorated encoders are reused as internal arithmetic
# primitives, so A/B perturb every spike-time sub-computation and noise compounds across
# hundreds of sites (see reports/NOISE.md). Values below bracket the ViT-S robustness cliffs measured
# at θ=2000 (jitter/mismatch ~1e-6…1e-4, hazard lower). Re-calibrate for other θ / model sizes.

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/gpu_pool.sh"
device="cuda"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32
theta=2000

base="--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"

flags=(
    "${base}"
    "${base} --jitter-enabled --jitter-mode potential --noise-std 1e-6"
    "${base} --jitter-enabled --jitter-mode potential --noise-std 2e-6"
    "${base} --jitter-enabled --jitter-mode potential --noise-std 5e-6"
    "${base} --jitter-enabled --jitter-mode potential --noise-std 1e-5"
    "${base} --hazard-enabled --hazard-delta-u 1e-6"
    "${base} --hazard-enabled --hazard-delta-u 5e-6"
    "${base} --mismatch-enabled --mismatch-theta-std 1e-5"
    "${base} --mismatch-enabled --mismatch-theta-std 3e-5"
    "${base} --mismatch-enabled --mismatch-theta-std 5e-5"
)
expr_names=(
    "noise_off_baseline"
    "A_jitter_pot_1e-6"
    "A_jitter_pot_2e-6"
    "A_jitter_pot_5e-6"
    "A_jitter_pot_1e-5"
    "B_hazard_du1e-6"
    "B_hazard_du5e-6"
    "C_mismatch_1e-5"
    "C_mismatch_3e-5"
    "C_mismatch_5e-5"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire                       # blocks until a GPU is free (assigns to whichever frees first)
    gpu=$GPU_POOL_ACQUIRED
    echo "Running noise analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 error_analysis_vit.py \
        --experiment_name noise-${expr_names[$index]} --device ${device} \
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta ${theta} --quick-test"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
