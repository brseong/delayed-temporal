#!/bin/bash
#SBATCH --job-name=theta-reduce
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

: "${THETA_REDUCE_MODE:?THETA_REDUCE_MODE is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"
: "${THETA_LOG_DIR:?THETA_LOG_DIR is required}"
: "${THETA_OUTPUT_DIR:?THETA_OUTPUT_DIR is required}"
: "${THETA_MANIFESTS:?THETA_MANIFESTS is required}"

mkdir -p "$THETA_OUTPUT_DIR"
runtime_parent="${SLURM_TMPDIR:-/tmp}/theta-reducer-runtime-${SLURM_JOB_ID}"
mkdir -p "$runtime_parent"
zstd --decompress --stdout "$THETA_ENV_ARCHIVE" \
    | tar -xf - -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"
to_container_path() {
    case "$1" in
        "$THETA_REMOTE_ASSETS"/*)
            printf '/data/ubai-assets/%s' "${1#"$THETA_REMOTE_ASSETS"/}"
            ;;
        "$THETA_REMOTE_REPO"/*)
            printf '/data/delayed-temporal/%s' "${1#"$THETA_REMOTE_REPO"/}"
            ;;
        *)
            echo "Reducer path is outside mounted roots: $1" >&2
            return 2
            ;;
    esac
}

container_log_dir="$(to_container_path "$THETA_LOG_DIR")"
container_output_dir="$(to_container_path "$THETA_OUTPUT_DIR")"
IFS=',' read -r -a manifest_paths <<< "$THETA_MANIFESTS"
manifest_args=()
for manifest in "${manifest_paths[@]}"; do
    manifest_args+=(--manifest "$(to_container_path "$manifest")")
done

case "$THETA_REDUCE_MODE" in
    benchmark)
        : "${THETA_AVAILABILITY_FILE:?THETA_AVAILABILITY_FILE is required}"
        availability_args=()
        while IFS= read -r item; do
            [[ -n "$item" ]] && availability_args+=(--available "$item")
        done < "$THETA_AVAILABILITY_FILE"
        command=(
            scripts/analysis/select_ubai_gpu_family.py
            "${manifest_args[@]}"
            --log-dir "$container_log_dir"
            --output "$container_output_dir/gpu-selection.json"
            "${availability_args[@]}"
        )
        ;;
    selection|confirmation|full)
        command=(
            scripts/analysis/summarize_theta_selection.py
            "${manifest_args[@]}"
            --log-dir "$container_log_dir"
            --raw-csv "$container_output_dir/theta-selection-raw.csv"
            --summary-csv "$container_output_dir/theta-selection-summary.csv"
            --selection-json "$container_output_dir/selection.json"
            --figure-prefix "$container_output_dir/theta-selection"
        )
        if [[ "$THETA_REDUCE_MODE" == "confirmation" || "$THETA_REDUCE_MODE" == "full" ]]; then
            command+=(--confirm)
        fi
        if [[ "$THETA_REDUCE_MODE" == "full" ]]; then
            command+=(--require-full)
        fi
        ;;
    *)
        echo "Unknown reducer mode: $THETA_REDUCE_MODE" >&2
        exit 2
        ;;
esac

srun \
    --ntasks=1 \
    --container-image="$THETA_CONTAINER_IMAGE" \
    --container-mounts="$THETA_REMOTE_REPO:/data/delayed-temporal,$runtime_parent/dt:/opt/conda/envs/dt,$THETA_REMOTE_ASSETS:/data/ubai-assets" \
    --container-workdir=/data/delayed-temporal \
    /usr/bin/env \
    PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
    MPLBACKEND=Agg \
    /opt/conda/envs/dt/bin/python "${command[@]}"
