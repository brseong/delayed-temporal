#!/bin/bash

# Wait for an allowed GPU pool, run the complete ViT-B noise campaign, validate
# every artifact, and publish only the two final manuscript PDFs.

set -Eeuo pipefail
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

readonly allowed_gpu_list="4 5 6 7"
python_bin="${PYTHON_BIN:-/opt/conda/envs/dt/bin/python}"
poll_seconds="${GPU_POLL_SECONDS:-60}"
idle_samples="${GPU_IDLE_SAMPLES:-2}"
campaign_dry_run="${NOISE_CAMPAIGN_DRY_RUN:-0}"
campaign_tag="vit_base_noise_campaign_float64_v2"
campaign_dir="$repo_root/artifacts/logs/noise_scan/$campaign_tag"
campaign_log="$campaign_dir/campaign.log"
smoke_log="$campaign_dir/clean_smoke.log"
status_file="$campaign_dir/status.tsv"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$idle_samples" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_IDLE_SAMPLES must be a positive integer" >&2
    exit 2
fi
if [[ "$campaign_dry_run" != "0" && "$campaign_dry_run" != "1" ]]; then
    echo "NOISE_CAMPAIGN_DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "PYTHON_BIN is not executable: $python_bin" >&2
    exit 2
fi

if [[ "$campaign_dry_run" == "1" ]]; then
    echo "Allowed GPUs: $allowed_gpu_list"
    echo "Idle rule: $idle_samples consecutive samples, ${poll_seconds}s apart"
    echo "Stages: smoke quick(70) full(19) theta(111) validate publish build"
    exit 0
fi

mkdir -p "$campaign_dir"
exec 9>"$campaign_dir/campaign.lock"
if ! flock -n 9; then
    echo "Another $campaign_tag supervisor is already running" >&2
    exit 2
fi

record_status() {
    local state="$1"
    local detail="$2"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$detail" \
        | tee -a "$status_file" "$campaign_log"
}

wait_for_idle_pool() {
    local -A streak=()
    local gpu pids
    local -a selected=()
    local -a status=()
    for gpu in $allowed_gpu_list; do
        streak[$gpu]=0
    done

    while true; do
        selected=()
        status=()
        for gpu in $allowed_gpu_list; do
            pids="$(
                nvidia-smi --id="$gpu" --query-compute-apps=pid \
                    --format=csv,noheader,nounits 2>/dev/null \
                    | sed '/^[[:space:]]*$/d'
            )"
            if [[ -z "$pids" ]]; then
                streak[$gpu]=$((streak[$gpu] + 1))
                status+=("$gpu:idle(${streak[$gpu]})")
                if (( streak[$gpu] >= idle_samples )); then
                    selected+=("$gpu")
                fi
            else
                streak[$gpu]=0
                status+=("$gpu:busy")
            fi
        done
        record_status "gpu-poll" "${status[*]}" >&2
        if (( ${#selected[@]} > 0 )); then
            printf '%s\n' "${selected[*]}"
            return 0
        fi
        sleep "$poll_seconds"
    done
}

smoke_complete() {
    [[ -f "$smoke_log" ]] \
        && grep -Eq '^Accuracy: [0-9]+([.][0-9]+)?$' "$smoke_log" \
        && grep -q '^Evaluation metadata — ' "$smoke_log" \
        && ! grep -q 'Traceback (most recent call last)' "$smoke_log"
}

run_clean_smoke() {
    if smoke_complete; then
        record_status "smoke-skip" "existing complete clean smoke"
        return 0
    fi

    local pool gpu temporary_log failed_log
    pool="$(wait_for_idle_pool)"
    read -r gpu _ <<< "$pool"
    temporary_log="$smoke_log.tmp.$$"
    record_status "smoke-start" "gpu=$gpu"
    if (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export WANDB_RUN_GROUP="$campaign_tag"
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        export NUMEXPR_NUM_THREADS=4
        "$python_bin" scripts/evaluation/error_analysis_vit.py \
            --experiment_name "$campaign_tag-clean-smoke" \
            --device cuda \
            --model_id /data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k \
            --dataset_id imagenet-1k \
            --batch_size 32 \
            --max_eval_batches 1 \
            --theta 2000 \
            --precision float64 \
            --spiking-layernorm \
            --spiking-mlp \
            --spiking-attention \
            --model_backend spiking
    ) >"$temporary_log" 2>&1 && grep -Eq '^Accuracy: [0-9]+([.][0-9]+)?$' "$temporary_log"; then
        mv "$temporary_log" "$smoke_log"
        record_status "smoke-complete" "gpu=$gpu"
    else
        failed_log="$campaign_dir/clean_smoke.failed.$(date -u +%Y%m%dT%H%M%SZ).log"
        mv "$temporary_log" "$failed_log"
        record_status "smoke-failed" "$failed_log"
        return 1
    fi
}

run_noise_stage() {
    local protocol="$1"
    local tag="$2"
    local figure_prefix="$3"
    local pool
    pool="$(wait_for_idle_pool)"
    record_status "$protocol-start" "gpus=$pool tag=$tag"
    GPUS="$pool" \
    PYTHON_BIN="$python_bin" \
    SCAN_PROTOCOL="$protocol" \
    SCAN_TAG="$tag" \
    SCAN_FIGURE_PREFIX="$figure_prefix" \
    PRECISION=float64 \
    BATCH_SIZE=32 \
    REPLICA_SEEDS="0 1 2" \
    REQUIRE_IDLE_GPUS=1 \
    PUBLISH_PAPER_FIGURE=0 \
        bash "$repo_root/scripts/experiments/noise_scan_vit.sh" \
        2>&1 | tee -a "$campaign_log"
    record_status "$protocol-complete" "gpus=$pool tag=$tag"
}

run_theta_stage() {
    local pool
    pool="$(wait_for_idle_pool)"
    record_status "theta-start" "gpus=$pool tag=vit_base_theta_noise_5k_float64_v2"
    GPUS="$pool" \
    PYTHON_BIN="$python_bin" \
    SCAN_ROOT_TAG=vit_base_theta_noise_5k_float64_v2 \
    THETA_FIGURE_PREFIX="$repo_root/artifacts/figures/noise_theta_vit_base_5k_float64_v2" \
    PRECISION=float64 \
    BATCH_SIZE=32 \
    REPLICA_SEEDS="0 1 2" \
    REQUIRE_IDLE_GPUS=1 \
    PUBLISH_PAPER_FIGURE=0 \
        bash "$repo_root/scripts/experiments/theta_jitter_analysis_vit.sh" \
        2>&1 | tee -a "$campaign_log"
    record_status "theta-complete" "gpus=$pool"
}

validate_numeric_artifacts() {
    "$python_bin" - <<'PY'
import csv
import math
from pathlib import Path

root = Path("artifacts/logs/noise_scan")
raw_paths = [
    root / "vit_base_noise_quick_float64_v2" / "raw_runs.csv",
    root / "vit_base_noise_full_float64_v2" / "raw_runs.csv",
]
raw_paths.extend(
    root / f"vit_base_theta_noise_5k_float64_v2_theta_{theta}" / "raw_runs.csv"
    for theta in (40, 400, 2000)
)
for path in raw_paths:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty raw table: {path}")
    for row in rows:
        accuracy = float(row["accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise SystemExit(f"invalid accuracy in {path}: {accuracy}")
        if row["axis"] == "gaussian":
            ratio = float(row["time_noise_std_to_identity_ulp"])
            if not math.isfinite(ratio) or ratio <= 1.0:
                raise SystemExit(f"unresolved Gaussian perturbation in {path}: {ratio}")

summary_paths = [path.with_name("summary.csv") for path in raw_paths]
for path in summary_paths:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("accuracy_mean", "accuracy_ci95_low", "accuracy_ci95_high"):
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(f"invalid {field} in {path}: {value}")
PY
}

validate_and_publish() {
    record_status "validation-start" "campaign artifacts"
    "$python_bin" scripts/verification/verify_gaussian_time_noise.py
    "$python_bin" scripts/verification/verify_noise_scan_summary.py
    "$python_bin" scripts/verification/verify_noise_scan_runner.py
    "$python_bin" scripts/verification/verify_theta_noise_summary.py
    validate_numeric_artifacts
    lat check

    install -m 0644 \
        "$repo_root/artifacts/figures/noise_robustness_vit_base_quick_float64_v2.pdf" \
        "$repo_root/paper/figures/noise-robustness-vit-base.pdf"
    install -m 0644 \
        "$repo_root/artifacts/figures/noise_theta_vit_base_5k_float64_v2.pdf" \
        "$repo_root/paper/figures/noise-theta-vit-base.pdf"
    (
        cd "$repo_root/paper"
        latexmk -pdf -interaction=nonstopmode -halt-on-error neurips_2026.tex
    ) 2>&1 | tee -a "$campaign_log"
    record_status "complete" "validated PDFs published and manuscript built"
}

record_status "campaign-start" "allowed_gpus=$allowed_gpu_list"
run_clean_smoke
run_noise_stage \
    quick \
    vit_base_noise_quick_float64_v2 \
    "$repo_root/artifacts/figures/noise_robustness_vit_base_quick_float64_v2"
run_noise_stage \
    full \
    vit_base_noise_full_float64_v2 \
    "$repo_root/artifacts/figures/noise_robustness_vit_base_full_float64_v2"
run_theta_stage
validate_and_publish
