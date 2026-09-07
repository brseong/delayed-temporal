#!/bin/bash

# Validate and sync accepted UBAI offline runs from baekryun.

set -euo pipefail

result_root="${1:-artifacts/logs/noise_scan/vit_base_sigma_margin_5k_float64_v1}"
project_python="${PYTHON_BIN:-/opt/conda/envs/dt/bin/python}"
wandb_bin="${WANDB_BIN:-/opt/conda/envs/dt/bin/wandb}"
sync_manifest="$result_root/outputs/wandb-sync-manifest.csv"
raw_csv="$result_root/outputs/raw_runs.csv"
complete_marker="$result_root/outputs/COMPLETE"
status_file="$result_root/outputs/wandb-sync-status.tsv"

for path in "$project_python" "$wandb_bin"; do
    if [[ ! -x "$path" ]]; then
        echo "Required executable is unavailable: $path" >&2
        exit 2
    fi
done
for path in "$sync_manifest" "$raw_csv" "$complete_marker"; do
    if [[ ! -f "$path" ]]; then
        echo "Refusing to sync incomplete result tree; missing: $path" >&2
        exit 2
    fi
done

"$project_python" -c '
import csv, hashlib, sys
from pathlib import Path
root, manifest_path = map(Path, sys.argv[1:3])
with manifest_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 470 or len({row["run_id"] for row in rows}) != 470:
    raise SystemExit("sync manifest must contain exactly 470 unique runs")
for row in rows:
    offline = (root / "wandb" / row["offline_dir"]).resolve()
    wandb_root = (root / "wandb").resolve()
    if wandb_root not in offline.parents or not offline.is_dir():
        raise SystemExit(f"invalid offline directory: {offline}")
    log = root / "logs" / ("{}.log".format(row["run_id"]))
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    if digest != row["log_sha256"]:
        raise SystemExit("log hash mismatch: {}".format(row["run_id"]))
' "$result_root" "$sync_manifest"

if [[ ! -f "$status_file" ]]; then
    printf 'run_id\tlog_sha256\tstatus\ttimestamp\n' > "$status_file"
fi
while IFS=, read -r run_id offline_relative log_sha256; do
    [[ "$run_id" == "run_id" ]] && continue
    if awk -F $'\t' -v run="$run_id" -v digest="$log_sha256" \
        '$1 == run && $2 == digest && $3 == "synced" {found=1} END {exit !found}' \
        "$status_file"; then
        continue
    fi
    "$wandb_bin" sync \
        --entity CIDA \
        --project vit-evaluation-imagenet-1k \
        "$result_root/wandb/$offline_relative"
    printf '%s\t%s\tsynced\t%s\n' "$run_id" "$log_sha256" \
        "$(date --iso-8601=seconds)" >> "$status_file"
done < "$sync_manifest"

"$project_python" scripts/verification/verify_sigma_margin_wandb_sync.py \
    --raw-csv "$raw_csv" \
    --entity CIDA \
    --project vit-evaluation-imagenet-1k \
    --group vit_base_sigma_margin_5k_float64_v1
