#!/usr/bin/env bash

set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

source "${repo_root}/scripts/lib/gpu_pool.sh"

read -r -a cuda_devices <<< "${CUDA_DEVICES:-0 1 2 3 4 5 6 7}"
read -r -a layer_indices <<< "${GELU_LAYER_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11}"
read -r -a timing_seeds <<< "${TIME_NOISE_SEEDS:-0}"
read -r -a python_command <<< "${PYTHON_BIN:-conda run -n dt python}"

scan_tag="${SCAN_TAG:-gaussian_gelu_layer_ablation_v1}"
log_dir="${SCAN_LOGDIR:-${repo_root}/artifacts/logs/noise_ablation/gelu_layer_v1}"
force="${FORCE:-0}"
time_noise_std_frac="${TIME_NOISE_STD_FRAC:-3.162e-10}"
model_id="${MODEL_ID:-/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k}"
batch_size="${BATCH_SIZE:-32}"
theta="${THETA:-2000}"

mkdir -p "${log_dir}"

is_complete_log() {
    local log_path="$1"

    # A reusable result must include both the evaluator's final metric and the
    # wrapper's zero exit marker. This distinguishes an interrupted final batch
    # from a genuinely completed condition.
    [[ -f "${log_path}" ]] || return 1
    grep -q '^Accuracy: ' "${log_path}" || return 1
    grep -q '^RUN_EXIT_STATUS=0$' "${log_path}" || return 1

    # Python failures can sometimes coexist with buffered output, so reject any
    # traceback even when a stale accuracy line is present in the same file.
    ! grep -q 'Traceback (most recent call last)' "${log_path}"
}

run_condition() {
    local gpu="$1"
    local layer="$2"
    local mode="$3"
    local seed="$4"
    local run_name="gelu-layer-${layer}-${mode}"
    local log_path="${log_dir}/layer_${layer}_${mode}.log"

    if [[ "${force}" != "1" ]] && is_complete_log "${log_path}"; then
        echo "Skipping complete condition: layer=${layer}, mode=${mode}"
        return 0
    fi

    local noise_args=()
    if [[ "${mode}" == noise_seed_* ]]; then
        noise_args=(
            --gaussian-time-noise
            --time-noise-std-frac "${time_noise_std_frac}"
            --time-noise-seed "${seed}"
        )
    fi

    echo "Starting layer=${layer}, mode=${mode} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" WANDB_RUN_GROUP="${scan_tag}" \
        "${python_command[@]}" scripts/evaluation/error_analysis_vit.py \
        --experiment_name "${run_name}" \
        --model_backend spiking \
        --model_id "${model_id}" \
        --dataset_id imagenet-1k \
        --batch_size "${batch_size}" \
        --device cuda \
        --precision float32 \
        --theta "${theta}" \
        --spiking-layernorm \
        --spiking-attention \
        --spiking-mlp \
        --quick-test \
        "${noise_args[@]}" \
        --spiking-mlp-exact-gelu-layers "${layer}" \
        > "${log_path}" 2>&1
    local status=$?
    echo "RUN_EXIT_STATUS=${status}" >> "${log_path}"
    echo "Finished layer=${layer}, mode=${mode}, status=${status}"
    return "${status}"
}

gpu_pool_init "${cuda_devices[@]}"

# Pair every noisy layer condition with its own deterministic run because even
# dense-versus-temporal floating-point differences can move a few classifications.
for layer in "${layer_indices[@]}"; do
    gpu_pool_acquire
    gpu="${GPU_POOL_ACQUIRED}"
    run_condition "${gpu}" "${layer}" baseline 0 &
    gpu_pool_register "$!" "${gpu}"

    for seed in "${timing_seeds[@]}"; do
        gpu_pool_acquire
        gpu="${GPU_POOL_ACQUIRED}"
        run_condition "${gpu}" "${layer}" "noise_seed_${seed}" "${seed}" &
        gpu_pool_register "$!" "${gpu}"
    done
done

wait

# Refuse to report success if any expected condition is absent or incomplete.
failed=0
for layer in "${layer_indices[@]}"; do
    is_complete_log "${log_dir}/layer_${layer}_baseline.log" || failed=1
    for seed in "${timing_seeds[@]}"; do
        is_complete_log "${log_dir}/layer_${layer}_noise_seed_${seed}.log" || failed=1
    done
done

if [[ "${failed}" != "0" ]]; then
    echo "One or more GELU layer-ablation conditions failed or remain incomplete."
    exit 1
fi

echo "All requested GELU layer-ablation conditions completed."
