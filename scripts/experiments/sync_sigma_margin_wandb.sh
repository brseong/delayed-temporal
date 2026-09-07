#!/bin/bash

# Sync accepted pre-migration W&B runs and audit the complete online mirror.

set -euo pipefail

result_root="${1:-artifacts/logs/noise_scan/vit_base_sigma_margin_5k_float64_v1}"
project_python="${PYTHON_BIN:-/opt/conda/envs/dt/bin/python}"
wandb_bin="${WANDB_BIN:-/opt/conda/envs/dt/bin/wandb}"
run_manifest="$result_root/outputs/wandb-run-manifest.csv"
raw_csv="$result_root/outputs/raw_runs.csv"
complete_marker="$result_root/outputs/COMPLETE"
status_file="$result_root/outputs/wandb-sync-status.tsv"
offline_pending="$result_root/outputs/wandb-offline-pending.tsv"

for path in "$project_python" "$wandb_bin"; do
    if [[ ! -x "$path" ]]; then
        echo "Required executable is unavailable: $path" >&2
        exit 2
    fi
done
for path in "$run_manifest" "$raw_csv" "$complete_marker"; do
    if [[ ! -f "$path" ]]; then
        echo "Refusing to sync incomplete result tree; missing: $path" >&2
        exit 2
    fi
done

"$project_python" -c '
import csv, hashlib, sys
from pathlib import Path
root, manifest_path, offline_path = map(Path, sys.argv[1:4])
with manifest_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 470 or len({row["run_id"] for row in rows}) != 470:
    raise SystemExit("run manifest must contain exactly 470 unique runs")
validated = []
for row in rows:
    mode = row.get("mode")
    if mode not in {"offline", "online"}:
        raise SystemExit("invalid W&B mode: {}".format(mode))
    local = (root / "wandb" / row["local_dir"]).resolve()
    wandb_root = (root / "wandb").resolve()
    if wandb_root not in local.parents or not local.is_dir():
        raise SystemExit("invalid local W&B directory: {}".format(local))
    log = root / "logs" / ("{}.log".format(row["run_id"]))
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    if digest != row["log_sha256"]:
        raise SystemExit("log hash mismatch: {}".format(row["run_id"]))
    if mode == "offline":
        validated.append((row["run_id"], row["local_dir"], row["log_sha256"]))
with offline_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, dialect="excel-tab", lineterminator="\n")
    writer.writerow(("run_id", "local_dir", "log_sha256"))
    writer.writerows(validated)
' "$result_root" "$run_manifest" "$offline_pending"

if [[ ! -f "$status_file" ]]; then
    printf 'run_id\tlog_sha256\tstatus\ttimestamp\n' > "$status_file"
fi
while IFS=$'\t' read -r run_id local_relative log_sha256; do
    [[ "$run_id" == "run_id" ]] && continue
    if awk -F $'\t' -v run="$run_id" -v digest="$log_sha256" \
        '$1 == run && $2 == digest && $3 == "synced" {found=1} END {exit !found}' \
        "$status_file"; then
        continue
    fi
    "$wandb_bin" sync \
        --entity CIDA \
        --project vit-evaluation-imagenet-1k \
        "$result_root/wandb/$local_relative"
    printf '%s\t%s\tsynced\t%s\n' "$run_id" "$log_sha256" \
        "$(date --iso-8601=seconds)" >> "$status_file"
done < "$offline_pending"

"$project_python" scripts/verification/verify_sigma_margin_wandb_sync.py \
    --raw-csv "$raw_csv" \
    --entity CIDA \
    --project vit-evaluation-imagenet-1k \
    --group vit_base_sigma_margin_5k_float64_v1
