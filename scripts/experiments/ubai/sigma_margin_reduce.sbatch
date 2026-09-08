#!/bin/bash
#SBATCH --job-name=sigma-margin-reduce
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

: "${SIGMA_MARGIN_MANIFEST:?SIGMA_MARGIN_MANIFEST is required}"
: "${SIGMA_MARGIN_LOG_DIR:?SIGMA_MARGIN_LOG_DIR is required}"
: "${SIGMA_MARGIN_OUTPUT_DIR:?SIGMA_MARGIN_OUTPUT_DIR is required}"
: "${SIGMA_MARGIN_FIGURE_DIR:?SIGMA_MARGIN_FIGURE_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ROOT:?THETA_ENV_ROOT is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

test -x "$THETA_ENV_ROOT/bin/python"
test -s "$THETA_ENV_ROOT/.archive.sha256"
mkdir -p "$SIGMA_MARGIN_OUTPUT_DIR" "$SIGMA_MARGIN_FIGURE_DIR"

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

container_manifest="$(to_container_path "$SIGMA_MARGIN_MANIFEST")"
container_log_dir="$(to_container_path "$SIGMA_MARGIN_LOG_DIR")"
container_output_dir="$(to_container_path "$SIGMA_MARGIN_OUTPUT_DIR")"
container_figure_dir="$(to_container_path "$SIGMA_MARGIN_FIGURE_DIR")"
scratch_root="$SIGMA_MARGIN_LOG_DIR/reducer-tmp/${SLURM_JOB_ID}"
container_scratch_root="$(to_container_path "$scratch_root")"
cleanup() {
    rm -rf -- "$scratch_root"
}
trap cleanup EXIT
mkdir -p "$scratch_root"
manifest_commit="$(awk -F $'\t' 'NR == 2 {print $15}' "$SIGMA_MARGIN_MANIFEST")"
actual_commit="$(git -C "$THETA_REMOTE_REPO" rev-parse HEAD)"
if [[ -z "$manifest_commit" || "$actual_commit" != "$manifest_commit" ]]; then
    echo "Reducer source commit mismatch: $actual_commit != $manifest_commit" >&2
    exit 2
fi

srun \
    --ntasks=1 \
    --container-image="$THETA_CONTAINER_IMAGE" \
    --container-mounts="$THETA_REMOTE_REPO:/data/delayed-temporal,$THETA_REMOTE_ASSETS/source-checkouts/transformers:/data/delayed-temporal/src/transformers,$THETA_REMOTE_ASSETS/source-checkouts/spikingjelly:/data/delayed-temporal/src/spikingjelly,$THETA_ENV_ROOT:/opt/conda/envs/dt,$THETA_REMOTE_ASSETS:/data/ubai-assets" \
    --container-workdir=/data/delayed-temporal \
    /usr/bin/env \
    TMPDIR="$container_scratch_root" \
    XDG_CACHE_HOME="$container_scratch_root/cache" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/data/delayed-temporal:/data/delayed-temporal/src/transformers/src:/data/delayed-temporal/src/spikingjelly \
    MPLBACKEND=Agg \
    MPLCONFIGDIR="$container_scratch_root/matplotlib" \
    /opt/conda/envs/dt/bin/python scripts/analysis/summarize_sigma_margin_sweep.py \
    --manifest "$container_manifest" \
    --log-dir "$container_log_dir" \
    --raw-csv "$container_output_dir/raw_runs.csv" \
    --summary-csv "$container_output_dir/summary.csv" \
    --site-csv "$container_output_dir/site_summary.csv" \
    --frontier-json "$container_output_dir/frontier.json" \
    --provenance-json "$container_output_dir/provenance.json" \
    --figure-prefix "$container_figure_dir/noise_sigma_margin_vit_base_5k_float64_v1"

: > "$SIGMA_MARGIN_OUTPUT_DIR/COMPLETE"
