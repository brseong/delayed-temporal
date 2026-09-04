#!/bin/bash

# Prepare or submit the canonical 5k timing-sigma/deadline-margin sweep on UBAI.

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--submit" ) ]]; then
    echo "Usage: $0 [--submit]" >&2
    exit 2
fi
submit="0"
if [[ "${1:-}" == "--submit" ]]; then
    submit="1"
fi

tag="vit_base_sigma_margin_5k_float64_v1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_repo="$(cd -- "$script_dir/../../.." && pwd)"
remote_repo="${THETA_REMOTE_REPO:-$default_repo}"
remote_assets="${THETA_REMOTE_ASSETS:-/home1/sizz1997/myubai/delayed-temporal-assets/theta-selection-v1}"
theta_result_root="${THETA_RESULT_ROOT:-$remote_assets/results/vit_base_theta_selection_float64_v1}"
result_root="${SIGMA_MARGIN_RESULT_ROOT:-$remote_assets/results/$tag}"
manifest_dir="$result_root/manifests"
log_dir="$result_root/logs"
output_dir="$result_root/outputs"
figure_dir="$result_root/figures"
manifest="$manifest_dir/expected_runs.tsv"
pending_manifest="$manifest_dir/pending.tsv"
dataset_manifest="$remote_assets/datasets/imagenet_theta_selection_v1/manifest.json"
selection_json="$theta_result_root/outputs/selection.json"
theta_raw_csv="$theta_result_root/outputs/theta-selection-raw.csv"
theta_full_manifest="$theta_result_root/manifests/full.tsv"
gpu_selection="$theta_result_root/outputs/gpu-selection.json"
checkpoint_path="/data/ubai-assets/checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k"
checkpoint_sha256="${THETA_CHECKPOINT_SHA256:-596ea1f22f56761c30661c87310c670e4ff296729bc5de349af41ac6ef6286ff}"
environment_archive="${THETA_ENV_ARCHIVE:-$remote_assets/runtime/dt-environment.tar.zst}"
container_image="${THETA_CONTAINER_IMAGE:-$remote_assets/runtime/ubuntu-24.04.sqsh}"

for path in \
    "$selection_json" "$theta_raw_csv" "$theta_full_manifest" \
    "$gpu_selection" "$dataset_manifest" "$environment_archive" "$container_image"; do
    if [[ ! -f "$path" ]]; then
        echo "Required artifact is missing: $path" >&2
        exit 2
    fi
done

mkdir -p "$manifest_dir" "$log_dir/slurm" "$output_dir" "$figure_dir"
exec 9> "$result_root/submit.lock"
if ! flock -n 9; then
    echo "Another sigma-margin submission command is active" >&2
    exit 2
fi
source_commit="$(git -C "$remote_repo" rev-parse HEAD)"
python3 "$remote_repo/scripts/experiments/ubai/build_sigma_margin_manifest.py" \
    --output "$manifest" \
    --experiment-json "$manifest_dir/experiment.json" \
    --selection-json "$selection_json" \
    --theta-raw-csv "$theta_raw_csv" \
    --theta-full-manifest "$theta_full_manifest" \
    --dataset-manifest "$dataset_manifest" \
    --gpu-selection "$gpu_selection" \
    --source-commit "$source_commit" \
    --checkpoint-path "$checkpoint_path" \
    --checkpoint-sha256 "$checkpoint_sha256"

python3 "$remote_repo/scripts/analysis/summarize_sigma_margin_sweep.py" \
    --manifest "$manifest" \
    --log-dir "$log_dir" \
    --write-pending "$pending_manifest"
pending_count="$(( $(wc -l < "$pending_manifest") - 1 ))"
partition="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_partition"])' "$gpu_selection")"

echo "Tag: $tag"
echo "Source commit: $source_commit"
echo "Partition: $partition"
echo "Expected runs: 470"
echo "Pending runs: $pending_count"
echo "Manifest: $manifest"
if [[ "$submit" == "0" ]]; then
    echo "Dry preparation complete; pass --submit to enqueue pending runs."
    exit 0
fi
active_jobs="$(squeue -h -u "${USER:?USER is required}" -n sigma-margin,sigma-margin-reduce -o '%A' | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_jobs" ]]; then
    echo "Sigma-margin jobs are already active: $active_jobs" >&2
    exit 2
fi

export THETA_REMOTE_REPO="$remote_repo"
export THETA_REMOTE_ASSETS="$remote_assets"
export THETA_ENV_ARCHIVE="$environment_archive"
export THETA_CONTAINER_IMAGE="$container_image"
export SIGMA_MARGIN_MANIFEST="$manifest"
export SIGMA_MARGIN_TASK_MANIFEST="$pending_manifest"
export SIGMA_MARGIN_LOG_DIR="$log_dir"
export SIGMA_MARGIN_OUTPUT_DIR="$output_dir"
export SIGMA_MARGIN_FIGURE_DIR="$figure_dir"

dependency_args=()
if (( pending_count > 0 )); then
    array_end="$((pending_count - 1))"
    array_job="$(sbatch \
        --parsable \
        --partition="$partition" \
        --time=03:00:00 \
        --array="0-${array_end}%8" \
        --output="$log_dir/slurm/%x-%A_%a.out" \
        --error="$log_dir/slurm/%x-%A_%a.err" \
        --export=ALL \
        "$remote_repo/scripts/experiments/ubai/sigma_margin_task.sbatch")"
    array_job="${array_job%%;*}"
    # The reducer runs after success or failure so an incomplete array becomes a
    # terminal failed summary rather than an indefinitely pending dependency.
    dependency_args+=(--dependency="afterany:$array_job")
    echo "Array job: $array_job"
else
    array_job=""
    echo "All evaluator logs are already complete; submitting reducer only."
fi

reducer_job="$(sbatch \
    --parsable \
    "${dependency_args[@]}" \
    --output="$log_dir/slurm/%x-%j.out" \
    --error="$log_dir/slurm/%x-%j.err" \
    --export=ALL \
    "$remote_repo/scripts/experiments/ubai/sigma_margin_reduce.sbatch")"
reducer_job="${reducer_job%%;*}"
printf '%s\tarray=%s\treducer=%s\tpending=%s\n' \
    "$(date --iso-8601=seconds)" "$array_job" "$reducer_job" "$pending_count" \
    >> "$output_dir/submissions.tsv"
echo "Reducer job: $reducer_job"
