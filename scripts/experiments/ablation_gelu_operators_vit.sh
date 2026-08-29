#!/usr/bin/env bash

set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

source "${repo_root}/scripts/lib/gpu_pool.sh"

read -r -a cuda_devices <<< "${CUDA_DEVICES:-0 1 2 3 4 5 6 7}"
read -r -a timing_seeds <<< "${TIME_NOISE_SEEDS:-0}"
read -r -a python_command <<< "${PYTHON_BIN:-conda run -n dt python}"
read -r -a conditions <<< "${GELU_OPERATOR_CONDITIONS:-full dense_mul dense_exp dense_div only_div_noisy only_exp_noisy only_mul_noisy all_dense}"

scan_tag="${SCAN_TAG:-gaussian_gelu_operator_ablation_v1}"
log_dir="${SCAN_LOGDIR:-${repo_root}/artifacts/logs/noise_ablation/gelu_operator_float32_v1}"
force="${FORCE:-0}"
time_noise_std_frac="${TIME_NOISE_STD_FRAC:-3.162e-10}"
model_id="${MODEL_ID:-/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k}"
batch_size="${BATCH_SIZE:-32}"
theta="${THETA:-2000}"
precision="${PRECISION:-float32}"
max_eval_batches="${MAX_EVAL_BATCHES:-0}"

mkdir -p "${log_dir}"

is_complete_log() {
    local log_path="$1"

    # A reusable run must contain the evaluator's final metric and the wrapper's
    # successful exit marker; interrupted logs are rerun without deleting evidence.
    [[ -f "${log_path}" ]] || return 1
    grep -q '^Accuracy: ' "${log_path}" || return 1
    grep -q '^RUN_EXIT_STATUS=0$' "${log_path}" || return 1

    # Buffered output can leave an old metric above a later failure, so any Python
    # traceback invalidates the condition even if both markers are present.
    ! grep -q 'Traceback (most recent call last)' "${log_path}"
}

operator_args_for_condition() {
    local condition="$1"

    # Each label denotes the GELU-local operators evaluated on their nominal
    # temporal carriers without Gaussian draws. All omitted operators stay noisy.
    case "${condition}" in
        full)
            GELU_OPERATOR_ARGS=()
            ;;
        dense_mul)
            GELU_OPERATOR_ARGS=(multiplication)
            ;;
        dense_exp)
            GELU_OPERATOR_ARGS=(exponential)
            ;;
        dense_div)
            GELU_OPERATOR_ARGS=(division)
            ;;
        only_div_noisy)
            GELU_OPERATOR_ARGS=(multiplication exponential)
            ;;
        only_exp_noisy)
            GELU_OPERATOR_ARGS=(multiplication division)
            ;;
        only_mul_noisy)
            GELU_OPERATOR_ARGS=(exponential division)
            ;;
        all_dense)
            GELU_OPERATOR_ARGS=(multiplication exponential division)
            ;;
        *)
            echo "Unknown GELU operator-ablation condition: ${condition}" >&2
            return 2
            ;;
    esac
}

run_condition() {
    local gpu="$1"
    local condition="$2"
    local seed="$3"
    local log_path="${log_dir}/${condition}_seed_${seed}.log"
    local max_batch_args=()

    if [[ "${force}" != "1" ]] && is_complete_log "${log_path}"; then
        echo "Skipping complete condition: ${condition}, seed=${seed}"
        return 0
    fi

    operator_args_for_condition "${condition}" || return $?
    if [[ "${max_eval_batches}" != "0" ]]; then
        max_batch_args=(--max_eval_batches "${max_eval_batches}")
    fi

    echo "Starting ${condition}, seed=${seed} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" WANDB_RUN_GROUP="${scan_tag}" \
        "${python_command[@]}" scripts/analysis/gelu_operator_ablation_vit.py \
        --gelu-dense-operators "${GELU_OPERATOR_ARGS[@]}" \
        --experiment_name "${scan_tag}-${condition}-seed-${seed}" \
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
        --gaussian-time-noise \
        --time-noise-std-frac "${time_noise_std_frac}" \
        --time-noise-seed "${seed}" \
        "${max_batch_args[@]}" \
        > "${log_path}" 2>&1
    local status=$?
    echo "RUN_EXIT_STATUS=${status}" >> "${log_path}"
    echo "Finished ${condition}, seed=${seed}, status=${status}"
    return "${status}"
}

# Schedule one process per GPU. Every process owns one Gaussian generator, so RNG
# streams never cross conditions and no DataParallel replica shares mutable state.
gpu_pool_init "${cuda_devices[@]}"
for seed in "${timing_seeds[@]}"; do
    for condition in "${conditions[@]}"; do
        gpu_pool_acquire
        gpu="${GPU_POOL_ACQUIRED}"
        run_condition "${gpu}" "${condition}" "${seed}" &
        gpu_pool_register "$!" "${gpu}"
    done
done

wait

# Report success only when the complete requested matrix has a valid final metric.
failed=0
for seed in "${timing_seeds[@]}"; do
    for condition in "${conditions[@]}"; do
        is_complete_log "${log_dir}/${condition}_seed_${seed}.log" || failed=1
    done
done

if [[ "${failed}" != "0" ]]; then
    echo "One or more GELU operator-ablation conditions failed or remain incomplete."
    exit 1
fi

echo "All requested GELU operator-ablation conditions completed."
