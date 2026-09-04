#!/bin/bash
#SBATCH --job-name=theta-select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

: "${THETA_MANIFEST:?THETA_MANIFEST is required}"
: "${THETA_LOG_DIR:?THETA_LOG_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

task_index="${SLURM_ARRAY_TASK_ID:?array task id is required}"
runtime_parent="${SLURM_TMPDIR:-/tmp}/theta-runtime-${SLURM_JOB_ID}-${task_index}"
mkdir -p "$runtime_parent"
tar --zstd -xf "$THETA_ENV_ARCHIVE" -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"
line_number="$((task_index + 2))"
row="$(sed -n "${line_number}p" "$THETA_MANIFEST")"
if [[ -z "$row" ]]; then
    echo "No manifest row for task index $task_index" >&2
    exit 2
fi

IFS=$'\t' read -r \
    run_id stage backend theta split expected_samples dataset_path \
    dataset_fingerprint precision source_commit checkpoint_path \
    checkpoint_sha256 gpu_family quick_test benchmark_warmup \
    benchmark_measure log_file <<< "$row"

mkdir -p "$THETA_LOG_DIR"
log_path="$THETA_LOG_DIR/$log_file"
temporary_log="$log_path.partial.${SLURM_JOB_ID}.${task_index}"

actual_commit="$(git -C "$THETA_REMOTE_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$source_commit" ]]; then
    echo "Source commit mismatch: $actual_commit != $source_commit" >&2
    exit 2
fi

if [[ -f "$log_path" ]] \
    && grep -q "^Slurm identity — .* gpu_family: ${gpu_family}$" "$log_path" \
    && grep -q "^Artifact identity — source_commit: ${source_commit}, checkpoint_sha256: ${checkpoint_sha256}$" "$log_path" \
    && grep -q "fingerprint: ${dataset_fingerprint}$" "$log_path" \
    && grep -q "^Evaluated samples: ${expected_samples}$" "$log_path" \
    && grep -Eq '^Accuracy: [0-9]+([.][0-9]+)?$' "$log_path" \
    && ! grep -q 'Traceback (most recent call last)' "$log_path"; then
    echo "Skipping complete manifest identity: $run_id"
    exit 0
fi

quick_args=()
if [[ "$quick_test" == "1" ]]; then
    quick_args+=(--quick-test)
fi
benchmark_args=()
if [[ "$benchmark_measure" != "0" ]]; then
    benchmark_args+=(
        --benchmark-warmup-batches "$benchmark_warmup"
        --benchmark-measure-batches "$benchmark_measure"
    )
fi

{
    echo "Slurm identity — job_id: ${SLURM_JOB_ID}, task_id: ${task_index}, node: ${SLURMD_NODENAME}, gpu_family: ${gpu_family}"
    srun \
        --ntasks=1 \
        --container-image="$THETA_CONTAINER_IMAGE" \
        --container-mounts="$THETA_REMOTE_REPO:/data/delayed-temporal,$runtime_parent/dt:/opt/conda/envs/dt,$THETA_REMOTE_ASSETS:/data/ubai-assets" \
        --container-workdir=/data/delayed-temporal \
        /usr/bin/env \
        PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
        WANDB_MODE=disabled \
        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        /opt/conda/envs/dt/bin/python scripts/evaluation/error_analysis_vit.py \
        --experiment_name "$run_id" \
        --device cuda \
        --model_backend "$backend" \
        --model_id "$checkpoint_path" \
        --dataset_id imagenet-1k \
        --evaluation-dataset-path "$dataset_path" \
        --evaluation-split "$split" \
        --batch_size 32 \
        --theta "$theta" \
        --precision "$precision" \
        --calibration-mode none \
        --no-gaussian-time-noise \
        --time-noise-std-frac 0 \
        --time-noise-mean 0 \
        --time-noise-deadline-margin-std 0 \
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
        --spiking-attention \
        "${quick_args[@]}" \
        "${benchmark_args[@]}"
} > "$temporary_log" 2>&1

mv "$temporary_log" "$log_path"
echo "Completed $run_id -> $log_path"
