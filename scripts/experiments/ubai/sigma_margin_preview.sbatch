#!/bin/bash
#SBATCH --job-name=sigma-margin-preview
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

: "${SIGMA_MARGIN_MANIFEST:?SIGMA_MARGIN_MANIFEST is required}"
: "${SIGMA_MARGIN_LOG_DIR:?SIGMA_MARGIN_LOG_DIR is required}"
: "${SIGMA_MARGIN_PREVIEW_OUTPUT_DIR:?SIGMA_MARGIN_PREVIEW_OUTPUT_DIR is required}"
: "${SIGMA_MARGIN_PREVIEW_FIGURE_DIR:?SIGMA_MARGIN_PREVIEW_FIGURE_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

enroot_data="/enroot/$(id -u)/data"
runtime_parent="$enroot_data/sigma-margin-preview-${SLURM_JOB_ID}"
cleanup() {
    rm -rf -- "$runtime_parent"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
mkdir -p "$enroot_data" "$runtime_parent/scratch"
find "$enroot_data" -mindepth 1 -maxdepth 1 -type d \
    -name 'sigma-margin-preview-*' -uid "$(id -u)" -mmin +120 \
    -exec rm -rf -- {} +
zstd --decompress --stdout "$THETA_ENV_ARCHIVE" | tar -xf - -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"
mkdir -p "$SIGMA_MARGIN_PREVIEW_OUTPUT_DIR" "$SIGMA_MARGIN_PREVIEW_FIGURE_DIR"

to_container_path() {
    case "$1" in
        "$THETA_REMOTE_ASSETS"/*)
            printf '/data/ubai-assets/%s' "${1#"$THETA_REMOTE_ASSETS"/}"
            ;;
        "$THETA_REMOTE_REPO"/*)
            printf '/data/delayed-temporal/%s' "${1#"$THETA_REMOTE_REPO"/}"
            ;;
        *)
            echo "Preview path is outside mounted roots: $1" >&2
            return 2
            ;;
    esac
}

container_manifest="$(to_container_path "$SIGMA_MARGIN_MANIFEST")"
container_log_dir="$(to_container_path "$SIGMA_MARGIN_LOG_DIR")"
container_output_dir="$(to_container_path "$SIGMA_MARGIN_PREVIEW_OUTPUT_DIR")"
container_figure_dir="$(to_container_path "$SIGMA_MARGIN_PREVIEW_FIGURE_DIR")"

srun \
    --ntasks=1 \
    --container-image="$THETA_CONTAINER_IMAGE" \
    --container-mounts="$THETA_REMOTE_REPO:/data/delayed-temporal,$THETA_REMOTE_ASSETS/source-checkouts/transformers:/data/delayed-temporal/src/transformers,$THETA_REMOTE_ASSETS/source-checkouts/spikingjelly:/data/delayed-temporal/src/spikingjelly,$runtime_parent/dt:/opt/conda/envs/dt,$runtime_parent/scratch:/work-tmp,$THETA_REMOTE_ASSETS:/data/ubai-assets" \
    --container-workdir=/data/delayed-temporal \
    /usr/bin/env \
    TMPDIR=/work-tmp \
    XDG_CACHE_HOME=/work-tmp/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/work-tmp/matplotlib \
    /opt/conda/envs/dt/bin/python scripts/analysis/preview_sigma_margin_sweep.py \
    --manifest "$container_manifest" \
    --log-dir "$container_log_dir" \
    --raw-csv "$container_output_dir/raw_runs.csv" \
    --summary-csv "$container_output_dir/summary.csv" \
    --site-csv "$container_output_dir/site_summary.csv" \
    --snapshot-json "$container_output_dir/snapshot.json" \
    --figure-prefix "$container_figure_dir/noise_sigma_margin_partial"

: > "$SIGMA_MARGIN_PREVIEW_OUTPUT_DIR/COMPLETE"
