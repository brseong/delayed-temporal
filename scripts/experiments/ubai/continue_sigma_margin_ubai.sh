#!/bin/bash
#SBATCH --job-name=sigma-margin-next
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

repo="${THETA_REMOTE_REPO:-/home1/sizz1997/myubai/delayed-temporal-sigma-margin}"
assets="${THETA_REMOTE_ASSETS:-/home1/sizz1997/myubai/delayed-temporal-assets/theta-selection-v1}"
root="${SIGMA_MARGIN_RESULT_ROOT:-$assets/results/vit_base_sigma_margin_5k_float64_v1}"
control_python="${SIGMA_MARGIN_CONTROL_PYTHON:-/home1/sizz1997/miniconda3/bin/python}"
manifest="$root/manifests/expected_runs.tsv"
pending="$root/manifests/pending.tsv"
log_dir="$root/logs"
batch_dir="$root/manifests/batches"
mkdir -p "$batch_dir" "$root/outputs" "$log_dir/slurm"
exec 9> "$root/continue.lock"
flock -n 9 || { echo "Another continuation controller is active" >&2; exit 2; }

"$control_python" "$repo/scripts/analysis/summarize_sigma_margin_sweep.py" \
    --manifest "$manifest" --log-dir "$log_dir" --write-pending "$pending"
pending_count="$(( $(wc -l < "$pending") - 1 ))"

export THETA_REMOTE_REPO="$repo"
export THETA_REMOTE_ASSETS="$assets"
export THETA_ENV_ARCHIVE="${THETA_ENV_ARCHIVE:-$assets/runtime/dt-environment.tar.zst}"
export THETA_CONTAINER_IMAGE="${THETA_CONTAINER_IMAGE:-$assets/runtime/ubuntu-24.04.sqsh}"
export SIGMA_MARGIN_MANIFEST="$manifest"
export SIGMA_MARGIN_LOG_DIR="$log_dir"
export SIGMA_MARGIN_OUTPUT_DIR="$root/outputs"
export SIGMA_MARGIN_FIGURE_DIR="$root/figures"

if (( pending_count == 0 )); then
    reducer="$(sbatch --parsable \
        --output="$log_dir/slurm/%x-%j.out" --error="$log_dir/slurm/%x-%j.err" \
        --export=ALL "$repo/scripts/experiments/ubai/sigma_margin_reduce.sbatch")"
    reducer="${reducer%%;*}"
    printf '%s\tmode=disabled-tracking-complete\treducer=%s\n' \
        "$(date --iso-8601=seconds)" "$reducer" >> "$root/outputs/submissions.tsv"
    echo "Submitted reducer $reducer"
    exit 0
fi

batch_manifest="$batch_dir/disabled-batch-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.tsv"
"$control_python" -c '
import csv, sys
from collections import Counter
from pathlib import Path
pending_path, batch_path, batch_dir = map(Path, sys.argv[1:4])
with pending_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, dialect="excel-tab")
    fields, rows = tuple(reader.fieldnames or ()), list(reader)
attempts = Counter()
for old in batch_dir.glob("disabled-batch-*.tsv"):
    with old.open(newline="", encoding="utf-8") as handle:
        attempts.update(row["run_id"] for row in csv.DictReader(handle, dialect="excel-tab"))
exhausted = [row["run_id"] for row in rows if attempts[row["run_id"]] >= 3]
if exhausted:
    raise SystemExit("retry limit reached: " + ", ".join(exhausted[:10]))
selected = rows[:16]
with batch_path.open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
    writer.writeheader(); writer.writerows(selected)
print(len(selected))
' "$pending" "$batch_manifest" "$batch_dir"
batch_count="$(( $(wc -l < "$batch_manifest") - 1 ))"
if (( batch_count < 1 || batch_count > 16 )); then
    echo "Invalid quota batch size: $batch_count" >&2
    exit 2
fi
partition="$("$control_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_partition"])' "$root/../vit_base_theta_selection_float64_v1/outputs/gpu-selection.json")"

export SIGMA_MARGIN_TASK_MANIFEST="$batch_manifest"
array_end="$((batch_count - 1))"
array_job="$(sbatch --parsable --partition="$partition" --time=03:00:00 \
    --array="0-${array_end}%8" \
    --output="$log_dir/slurm/%x-%A_%a.out" --error="$log_dir/slurm/%x-%A_%a.err" \
    --export=ALL "$repo/scripts/experiments/ubai/sigma_margin_task.sbatch")"
array_job="${array_job%%;*}"
next_job="$(sbatch --parsable --partition=cpu1 --time=00:10:00 \
    --dependency="afterany:$array_job" \
    --output="$log_dir/slurm/%x-%j.out" --error="$log_dir/slurm/%x-%j.err" \
    --export=ALL "$root/control/sigma_margin_continue.sh")"
next_job="${next_job%%;*}"
printf '%s\tmode=disabled-tracking-chain\tarray=%s\tnext=%s\tbatch=%s\tpending_before=%s\n' \
    "$(date --iso-8601=seconds)" "$array_job" "$next_job" \
    "$(basename "$batch_manifest")" "$pending_count" >> "$root/outputs/submissions.tsv"
echo "Submitted array $array_job ($batch_count runs), continuation $next_job"
