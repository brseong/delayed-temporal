#!/bin/bash

# Submit a CPU reducer after one or more manifest arrays have completed.

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 MODE COLON_SEPARATED_MANIFESTS DEPENDENCY OUTPUT_DIR" >&2
    exit 2
fi

mode="$1"
manifests="$2"
dependency="$3"
output_dir="$(realpath -m "$4")"

: "${THETA_LOG_DIR:?THETA_LOG_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

mkdir -p "$output_dir" "$THETA_LOG_DIR/slurm"
extra_exports=""
dependency_type="afterok"
if [[ "$mode" == "benchmark" ]]; then
    : "${THETA_AVAILABILITY_FILE:?THETA_AVAILABILITY_FILE is required}"
    extra_exports=",THETA_AVAILABILITY_FILE=$THETA_AVAILABILITY_FILE"
    dependency_type="afterany"
fi

sbatch \
    --parsable \
    --dependency="$dependency_type:$dependency" \
    --output="$THETA_LOG_DIR/slurm/%x-%j.out" \
    --error="$THETA_LOG_DIR/slurm/%x-%j.err" \
    --export="ALL,THETA_REDUCE_MODE=$mode,THETA_MANIFESTS=$manifests,THETA_OUTPUT_DIR=$output_dir$extra_exports" \
    "$THETA_REMOTE_REPO/scripts/experiments/ubai/theta_selection_reduce.sbatch"
