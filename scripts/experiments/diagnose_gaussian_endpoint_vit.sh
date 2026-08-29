#!/usr/bin/env bash

set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

source "${repo_root}/scripts/lib/gpu_pool.sh"

read -r -a cuda_devices <<< "${CUDA_DEVICES:-0 1 2 3 4}"
read -r -a python_command <<< "${PYTHON_BIN:-conda run -n dt python}"

scan_tag="${SCAN_TAG:-gaussian_endpoint_float64_v1}"
log_dir="${SCAN_LOGDIR:-${repo_root}/artifacts/logs/noise_ablation/gelu_endpoint_float64_v1}"
force="${FORCE:-0}"
model_id="${MODEL_ID:-/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k}"
batch_size="${BATCH_SIZE:-16}"
theta="${THETA:-2000}"
precision="${PRECISION:-float64}"
time_noise_std_frac="${TIME_NOISE_STD_FRAC:-3.162e-10}"
time_noise_seed="${TIME_NOISE_SEED:-0}"

conditions=(
    baseline
    full_gaussian
    layer10_exact
    all_gelu_exact
    all_gelu_exact_no_ln_log
)

mkdir -p "${log_dir}"

is_complete_log() {
    local log_path="$1"

    # Require both the evaluator metric and a successful wrapper exit marker so an
    # interrupted buffered log cannot be mistaken for a reusable experiment result.
    [[ -f "${log_path}" ]] || return 1
    grep -q '^Accuracy: ' "${log_path}" || return 1
    grep -q '^RUN_EXIT_STATUS=0$' "${log_path}" || return 1

    # Reject tracebacks even if an older metric line is present in the same file.
    ! grep -q 'Traceback (most recent call last)' "${log_path}"
}

run_condition() {
    local gpu="$1"
    local condition="$2"
    local log_path="${log_dir}/${condition}.log"
    local noise_args=()
    local ablation_args=()

    if [[ "${force}" != "1" ]] && is_complete_log "${log_path}"; then
        echo "Skipping complete endpoint diagnostic: ${condition}"
        return 0
    fi

    # Every non-baseline condition uses the identical continuous-Gaussian request;
    # only the selected endpoint-heavy temporal stages differ across conditions.
    if [[ "${condition}" != baseline ]]; then
        noise_args=(
            --gaussian-time-noise
            --time-noise-std-frac "${time_noise_std_frac}"
            --time-noise-seed "${time_noise_seed}"
        )
    fi

    # Keep the mathematical cubic-tanh GELU formula when bypassing its temporal
    # composition, and change LayerNorm log encoding only in the final control.
    case "${condition}" in
        baseline|full_gaussian)
            ;;
        layer10_exact)
            ablation_args=(--spiking-mlp-exact-gelu-layers 10)
            ;;
        all_gelu_exact)
            ablation_args=(--spiking-mlp-exact-gelu)
            ;;
        all_gelu_exact_no_ln_log)
            ablation_args=(--spiking-mlp-exact-gelu --no-spiking-ln-log)
            ;;
        *)
            echo "Unknown endpoint diagnostic condition: ${condition}"
            return 2
            ;;
    esac

    echo "Starting ${condition} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" WANDB_RUN_GROUP="${scan_tag}" \
        "${python_command[@]}" scripts/evaluation/error_analysis_vit.py \
        --experiment_name "${scan_tag}-${condition}" \
        --model_backend spiking \
        --model_id "${model_id}" \
        --dataset_id imagenet-1k \
        --batch_size "${batch_size}" \
        --device cuda \
        --precision "${precision}" \
        --theta "${theta}" \
        --spiking-layernorm \
        --spiking-attention \
        --spiking-mlp \
        --quick-test \
        "${noise_args[@]}" \
        "${ablation_args[@]}" \
        > "${log_path}" 2>&1
    local status=$?
    echo "RUN_EXIT_STATUS=${status}" >> "${log_path}"
    echo "Finished ${condition}, status=${status}"
    return "${status}"
}

gpu_pool_init "${cuda_devices[@]}"
for condition in "${conditions[@]}"; do
    gpu_pool_acquire
    gpu="${GPU_POOL_ACQUIRED}"
    run_condition "${gpu}" "${condition}" &
    gpu_pool_register "$!" "${gpu}"
done

wait

# The diagnostic is complete only when every mechanism-isolating condition has a
# final 5k accuracy and a successful process exit.
failed=0
for condition in "${conditions[@]}"; do
    is_complete_log "${log_dir}/${condition}.log" || failed=1
done

if [[ "${failed}" != "0" ]]; then
    echo "One or more endpoint diagnostic conditions failed or remain incomplete."
    exit 1
fi

echo "All Gaussian endpoint diagnostic conditions completed."
