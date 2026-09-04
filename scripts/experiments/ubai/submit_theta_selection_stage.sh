#!/bin/bash

# Submit one manifest as a homogeneous UBAI Slurm array. Run this only on gate1/2.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 MANIFEST PARTITIONS TIME_LIMIT [DEPENDENCY]" >&2
    exit 2
fi

manifest="$(realpath "$1")"
partitions="$2"
time_limit="$3"
dependency="${4:-}"

: "${THETA_LOG_DIR:?THETA_LOG_DIR is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_REMOTE_ENV:?THETA_REMOTE_ENV is required}"
: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"

task_count="$(( $(wc -l < "$manifest") - 1 ))"
if [[ "$task_count" -le 0 ]]; then
    echo "Manifest has no task rows: $manifest" >&2
    exit 2
fi
array_end="$((task_count - 1))"
mkdir -p "$THETA_LOG_DIR/slurm"

dependency_args=()
if [[ -n "$dependency" ]]; then
    dependency_args+=(--dependency="afterok:$dependency")
fi

sbatch \
    --parsable \
    --partition="$partitions" \
    --time="$time_limit" \
    --array="0-${array_end}%8" \
    --output="$THETA_LOG_DIR/slurm/%x-%A_%a.out" \
    --error="$THETA_LOG_DIR/slurm/%x-%A_%a.err" \
    --export="ALL,THETA_MANIFEST=$manifest" \
    "${dependency_args[@]}" \
    "$THETA_REMOTE_REPO/scripts/experiments/ubai/theta_selection_task.sbatch"
