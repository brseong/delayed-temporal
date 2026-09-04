#!/bin/bash
#SBATCH --job-name=sigma-margin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

: "${SIGMA_MARGIN_MANIFEST:?SIGMA_MARGIN_MANIFEST is required}"
: "${SIGMA_MARGIN_TASK_MANIFEST:?SIGMA_MARGIN_TASK_MANIFEST is required}"
: "${SIGMA_MARGIN_LOG_DIR:?SIGMA_MARGIN_LOG_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

task_index="${SLURM_ARRAY_TASK_ID:?array task id is required}"
runtime_parent="${SLURM_TMPDIR:-/tmp}/sigma-margin-runtime-${SLURM_JOB_ID}-${task_index}"
mkdir -p "$runtime_parent"
zstd --decompress --stdout "$THETA_ENV_ARCHIVE" | tar -xf - -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"

line_number="$((task_index + 2))"
row="$(sed -n "${line_number}p" "$SIGMA_MARGIN_TASK_MANIFEST")"
if [[ -z "$row" ]]; then
    echo "No pending-manifest row for task index $task_index" >&2
    exit 2
fi
IFS=$'\t' read -r \
    run_id stage backend theta time_noise_std_frac time_noise_std_abs \
    deadline_margin_std deadline_margin_abs seed split expected_samples \
    dataset_path dataset_fingerprint precision source_commit checkpoint_path \
    checkpoint_sha256 gpu_family theta_selection_sha256 \
    theta_selection_raw_sha256 theta_full_manifest_sha256 \
    gpu_selection_sha256 log_file <<< "$row"

actual_commit="$(git -C "$THETA_REMOTE_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$source_commit" ]]; then
    echo "Source commit mismatch: $actual_commit != $source_commit" >&2
    exit 2
fi

to_container_path() {
    case "$1" in
        "$THETA_REMOTE_ASSETS"/*)
            printf '/data/ubai-assets/%s' "${1#"$THETA_REMOTE_ASSETS"/}"
            ;;
        "$THETA_REMOTE_REPO"/*)
            printf '/data/delayed-temporal/%s' "${1#"$THETA_REMOTE_REPO"/}"
            ;;
        *)
            echo "Path is outside mounted roots: $1" >&2
            return 2
            ;;
    esac
}

container_manifest="$(to_container_path "$SIGMA_MARGIN_MANIFEST")"
container_log_dir="$(to_container_path "$SIGMA_MARGIN_LOG_DIR")"
mounts="$THETA_REMOTE_REPO:/data/delayed-temporal,$THETA_REMOTE_ASSETS/source-checkouts/transformers:/data/delayed-temporal/src/transformers,$THETA_REMOTE_ASSETS/source-checkouts/spikingjelly:/data/delayed-temporal/src/spikingjelly,$runtime_parent/dt:/opt/conda/envs/dt,$THETA_REMOTE_ASSETS:/data/ubai-assets"

check_complete() {
    srun \
        --ntasks=1 \
        --container-image="$THETA_CONTAINER_IMAGE" \
        --container-mounts="$mounts" \
        --container-workdir=/data/delayed-temporal \
        /usr/bin/env \
        PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
        /opt/conda/envs/dt/bin/python \
        scripts/analysis/summarize_sigma_margin_sweep.py \
        --manifest "$container_manifest" \
        --log-dir "$container_log_dir" \
        --check-run-id "$run_id"
}

mkdir -p "$SIGMA_MARGIN_LOG_DIR"
log_path="$SIGMA_MARGIN_LOG_DIR/$log_file"
if check_complete >/dev/null 2>&1; then
    echo "Skipping complete manifest identity: $run_id"
    exit 0
fi
if [[ -f "$log_path" ]]; then
    mv "$log_path" "$log_path.rejected.$(date -u +%Y%m%dT%H%M%SZ)"
fi
temporary_log="$log_path.partial.${SLURM_JOB_ID}.${task_index}"

gaussian_args=(--no-gaussian-time-noise --time-noise-seed 0)
if [[ "$stage" == "sigma_margin" ]]; then
    gaussian_args=(--gaussian-time-noise --time-noise-seed "$seed")
fi

export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"
if {
    echo "Slurm identity — job_id: ${SLURM_JOB_ID}, task_id: ${task_index}, node: ${SLURMD_NODENAME}, gpu_family: ${gpu_family}"
    srun \
        --ntasks=1 \
        --gres=gpu:1 \
        --container-image="$THETA_CONTAINER_IMAGE" \
        --container-mounts="$mounts" \
        --container-workdir=/data/delayed-temporal \
        /usr/bin/env \
        PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
        WANDB_MODE=disabled \
        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        /opt/conda/envs/dt/bin/python scripts/evaluation/error_analysis_vit.py \
        --experiment_name "vit_base_sigma_margin_5k_float64_v1-${run_id}" \
        --device cuda \
        --model_backend "$backend" \
        --model_id "$checkpoint_path" \
        --dataset_id imagenet-1k \
        --evaluation-dataset-path "$dataset_path" \
        --evaluation-split "$split" \
        --batch_size 32 \
        --quick-test \
        --theta "$theta" \
        --precision "$precision" \
        --calibration-mode none \
        "${gaussian_args[@]}" \
        --time-noise-std-frac "$time_noise_std_frac" \
        --time-noise-mean 0 \
        --time-noise-deadline-margin-std "$deadline_margin_std" \
        --no-mismatch-enabled \
        --mismatch-theta-std 0 \
        --weight-noise-std 0 \
        --bias-noise-std 0 \
        --source-commit "$source_commit" \
        --checkpoint-sha256 "$checkpoint_sha256" \
        --no-tensorboard \
        --report-clamp-stats \
        --spiking-layernorm \
        --spiking-mlp \
        --spiking-attention
} > "$temporary_log" 2>&1; then
    mv "$temporary_log" "$log_path"
else
    echo "Evaluator failed; partial log retained at $temporary_log" >&2
    exit 1
fi

if ! check_complete; then
    mv "$log_path" "$log_path.failed.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Completed evaluator output failed manifest validation: $run_id" >&2
    exit 1
fi
echo "Completed $run_id -> $log_path"
