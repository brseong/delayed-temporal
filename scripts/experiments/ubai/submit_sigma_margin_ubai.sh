#!/bin/bash

# Prepare, pilot, or submit the canonical 5k timing-sigma/deadline-margin sweep.

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--pilot" && "$1" != "--submit" ) ]]; then
    echo "Usage: $0 [--pilot|--submit]" >&2
    exit 2
fi
mode="dry"
if [[ "${1:-}" == "--pilot" ]]; then
    mode="pilot"
elif [[ "${1:-}" == "--submit" ]]; then
    mode="full"
fi

tag="vit_base_sigma_margin_5k_float64_v1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_repo="$(cd -- "$script_dir/../../.." && pwd)"
remote_repo="${THETA_REMOTE_REPO:-$default_repo}"
remote_assets="${THETA_REMOTE_ASSETS:-/home1/sizz1997/myubai/delayed-temporal-assets/theta-selection-v1}"
theta_result_root="${THETA_RESULT_ROOT:-$remote_assets/results/vit_base_theta_selection_float64_v1}"
result_root="${SIGMA_MARGIN_RESULT_ROOT:-$remote_assets/results/$tag}"
control_python="${SIGMA_MARGIN_CONTROL_PYTHON:-/home1/sizz1997/miniconda3/bin/python}"
manifest_dir="$result_root/manifests"
log_dir="$result_root/logs"
output_dir="$result_root/outputs"
figure_dir="$result_root/figures"
wandb_dir="$result_root/wandb"
manifest="$manifest_dir/expected_runs.tsv"
pilot_manifest="$manifest_dir/pilot.tsv"
pending_manifest="$manifest_dir/pending.tsv"
dataset_manifest="$remote_assets/datasets/imagenet_theta_selection_v1/manifest.json"
selection_json="$theta_result_root/outputs/selection.json"
theta_raw_csv="$theta_result_root/outputs/theta-selection-raw.csv"
theta_confirmation_manifest="$theta_result_root/manifests/confirmation-lower.tsv"
gpu_selection="$theta_result_root/outputs/gpu-selection.json"
checkpoint_path="/data/ubai-assets/checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k"
checkpoint_sha256="${THETA_CHECKPOINT_SHA256:-596ea1f22f56761c30661c87310c670e4ff296729bc5de349af41ac6ef6286ff}"
environment_archive="${THETA_ENV_ARCHIVE:-$remote_assets/runtime/dt-environment.tar.zst}"
container_image="${THETA_CONTAINER_IMAGE:-$remote_assets/runtime/ubuntu-24.04.sqsh}"
storage_limit_bytes="${SIGMA_MARGIN_STORAGE_LIMIT_BYTES:-60000000000}"

if [[ ! -x "$control_python" ]]; then
    echo "UBAI control-plane Python is unavailable: $control_python" >&2
    exit 2
fi
for path in \
    "$selection_json" "$theta_raw_csv" "$theta_confirmation_manifest" \
    "$gpu_selection" "$dataset_manifest" "$environment_archive" "$container_image"; do
    if [[ ! -f "$path" ]]; then
        echo "Required artifact is missing: $path" >&2
        exit 2
    fi
done

mkdir -p "$manifest_dir" "$log_dir/slurm" "$output_dir" "$figure_dir" \
    "$wandb_dir/runs" "$wandb_dir/rejected"
exec 9> "$result_root/submit.lock"
if ! flock -n 9; then
    echo "Another sigma-margin submission command is active" >&2
    exit 2
fi
source_commit="$(git -C "$remote_repo" rev-parse HEAD)"
"$control_python" "$remote_repo/scripts/experiments/ubai/build_sigma_margin_manifest.py" \
    --output "$manifest" \
    --pilot-output "$pilot_manifest" \
    --experiment-json "$manifest_dir/experiment.json" \
    --selection-json "$selection_json" \
    --theta-raw-csv "$theta_raw_csv" \
    --theta-confirmation-manifest "$theta_confirmation_manifest" \
    --dataset-manifest "$dataset_manifest" \
    --gpu-selection "$gpu_selection" \
    --source-commit "$source_commit" \
    --checkpoint-path "$checkpoint_path" \
    --checkpoint-sha256 "$checkpoint_sha256"

"$control_python" "$remote_repo/scripts/analysis/summarize_sigma_margin_sweep.py" \
    --manifest "$manifest" \
    --log-dir "$log_dir" \
    --wandb-dir "$wandb_dir" \
    --write-pending "$pending_manifest"
pending_count="$(( $(wc -l < "$pending_manifest") - 1 ))"
partition="$("$control_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_partition"])' "$gpu_selection")"

echo "Tag: $tag"
echo "Source commit: $source_commit"
echo "Partition: $partition"
echo "Expected runs: 470"
echo "Pending runs: $pending_count"
echo "Manifest: $manifest"
if [[ "$mode" == "dry" ]]; then
    echo "Dry preparation complete; pass --pilot for the six checks, then --submit for the remaining grid."
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
export SIGMA_MARGIN_LOG_DIR="$log_dir"
export SIGMA_MARGIN_OUTPUT_DIR="$output_dir"
export SIGMA_MARGIN_FIGURE_DIR="$figure_dir"
export SIGMA_MARGIN_WANDB_DIR="$wandb_dir"

if [[ "$mode" == "pilot" ]]; then
    export SIGMA_MARGIN_TASK_MANIFEST="$pilot_manifest"
    pilot_job="$(sbatch \
        --parsable \
        --partition="$partition" \
        --time=03:00:00 \
        --array="0-5%6" \
        --output="$log_dir/slurm/%x-%A_%a.out" \
        --error="$log_dir/slurm/%x-%A_%a.err" \
        --export=ALL \
        "$remote_repo/scripts/experiments/ubai/sigma_margin_task.sbatch")"
    pilot_job="${pilot_job%%;*}"
    printf '%s\tmode=pilot\tarray=%s\tpending=6\n' \
        "$(date --iso-8601=seconds)" "$pilot_job" >> "$output_dir/submissions.tsv"
    echo "Pilot array job: $pilot_job"
    exit 0
fi

"$control_python" -c '
import csv, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.experiments.ubai.build_sigma_margin_manifest import PILOT_RUN_IDS
with Path(sys.argv[2]).open(newline="", encoding="utf-8") as handle:
    pending = {row["run_id"] for row in csv.DictReader(handle, dialect="excel-tab")}
missing = sorted(set(PILOT_RUN_IDS) & pending)
if missing:
    raise SystemExit("pilot is incomplete or invalid: " + ", ".join(missing))
' "$remote_repo" "$pending_manifest"

"$control_python" -c '
import json, math, os, sys
from pathlib import Path
result_root, assets_root, output_path = map(Path, sys.argv[1:4])
limit = int(sys.argv[4])
pilot_ids = sys.argv[5:]
def tree_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0
pilot_bytes = 0
for run_id in pilot_ids:
    pilot_bytes += tree_bytes(result_root / "wandb" / "runs" / run_id)
    log = result_root / "logs" / f"{run_id}.log"
    pilot_bytes += log.stat().st_size
if pilot_bytes <= 0:
    raise SystemExit("pilot size is zero")
current_assets = tree_bytes(assets_root)
projected_additional = math.ceil(pilot_bytes * (470 - len(pilot_ids)) / len(pilot_ids))
projected_total = current_assets + projected_additional
payload = {
    "format_version": 1,
    "pilot_runs": len(pilot_ids),
    "pilot_bytes": pilot_bytes,
    "current_assets_bytes": current_assets,
    "projected_additional_bytes": projected_additional,
    "projected_total_bytes": projected_total,
    "storage_limit_bytes": limit,
    "status": "accepted" if projected_total <= limit else "rejected",
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output_path)
print(json.dumps(payload, sort_keys=True))
if projected_total > limit:
    raise SystemExit("projected storage exceeds the configured limit")
' "$result_root" "$remote_assets" "$output_dir/pilot-storage.json" "$storage_limit_bytes" \
    clean_spiking_baseline dense_reference \
    sigma_1p000em10_margin_0_seed_0 sigma_3p162em10_margin_0_seed_0 \
    sigma_1p000em09_margin_0_seed_0 sigma_1p000em09_margin_12_seed_0

export SIGMA_MARGIN_TASK_MANIFEST="$pending_manifest"
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
    dependency_args+=(--dependency="afterany:$array_job")
    echo "Array job: $array_job"
else
    array_job=""
    echo "All evaluator logs and local W&B runs are complete; submitting reducer only."
fi

reducer_job="$(sbatch \
    --parsable \
    "${dependency_args[@]}" \
    --output="$log_dir/slurm/%x-%j.out" \
    --error="$log_dir/slurm/%x-%j.err" \
    --export=ALL \
    "$remote_repo/scripts/experiments/ubai/sigma_margin_reduce.sbatch")"
reducer_job="${reducer_job%%;*}"
printf '%s\tmode=full\tarray=%s\treducer=%s\tpending=%s\n' \
    "$(date --iso-8601=seconds)" "$array_job" "$reducer_job" "$pending_count" \
    >> "$output_dir/submissions.tsv"
echo "Reducer job: $reducer_job"
