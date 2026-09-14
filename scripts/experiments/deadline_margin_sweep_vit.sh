#!/bin/bash

# Adaptively find the smallest Gaussian deadline grace that restores ViT-B/16.

set -Eeuo pipefail
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

readonly allowed_gpu_list="4 5 6 7"
gpu="${GPU:-4}"
python_bin="${PYTHON_BIN:-/opt/conda/envs/dt/bin/python}"
scan_tag="${SCAN_TAG:-vit_base_deadline_margin_r1e-10_float64_v1}"
log_dir="$repo_root/artifacts/logs/noise_scan/$scan_tag"
figure_prefix="$repo_root/artifacts/figures/deadline_margin_recovery_vit_base_r1e-10_float64_v1"
baseline_log="$repo_root/artifacts/logs/noise_scan/vit_base_noise_quick_float64_v2/noise_off_baseline.log"
time_noise_std_frac="1.000e-10"
recovery_tolerance="${RECOVERY_TOLERANCE:-0.01}"
read -r -a margins <<< "${DEADLINE_MARGIN_STDS:-0 0.5 1 1.5 2 2.5 3 4 5 6 8 10 12}"

if [[ " $allowed_gpu_list " != *" $gpu "* ]]; then
    echo "GPU must be one physical device from 4, 5, 6, or 7" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "PYTHON_BIN is not executable: $python_bin" >&2
    exit 2
fi
if [[ ! -f "$baseline_log" ]]; then
    echo "Required clean 5k baseline is missing: $baseline_log" >&2
    exit 2
fi
active_pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
    echo "GPU $gpu is occupied by compute process(es): $active_pids" >&2
    exit 2
fi

mkdir -p "$log_dir"
exec 9>"$log_dir/sweep.lock"
if ! flock -n 9; then
    echo "Another deadline-margin sweep is already running" >&2
    exit 2
fi

baseline_accuracy="$($python_bin - "$baseline_log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
matches = re.findall(r"^Accuracy: ([0-9.eE+-]+)$", text, re.MULTILINE)
if not matches:
    raise SystemExit("baseline log has no final accuracy")
print(matches[-1])
PY
)"
threshold="$($python_bin - "$baseline_accuracy" "$recovery_tolerance" <<'PY'
import sys
print(float(sys.argv[1]) - float(sys.argv[2]))
PY
)"

record_status() {
    printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$log_dir/status.tsv"
}

margin_slug() {
    printf '%s' "${1//./p}"
}

log_accuracy() {
    "$python_bin" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
if "Traceback (most recent call last)" in text:
    raise SystemExit(1)
matches = re.findall(r"^Accuracy: ([0-9.eE+-]+)$", text, re.MULTILINE)
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

run_condition() {
    local margin="$1"
    local seed="$2"
    local slug log_path temporary_log
    slug="$(margin_slug "$margin")"
    log_path="$log_dir/margin_${slug}_seed_${seed}.log"
    if accuracy="$(log_accuracy "$log_path" 2>/dev/null)"; then
        record_status "skip margin_std=$margin seed=$seed accuracy=$accuracy"
        return 0
    fi
    temporary_log="$log_path.tmp.$$"
    record_status "start margin_std=$margin seed=$seed gpu=$gpu"
    if (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export WANDB_RUN_GROUP="$scan_tag"
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
        "$python_bin" scripts/evaluation/error_analysis_vit.py \
            --experiment_name "$scan_tag-margin_${slug}-seed_${seed}" \
            --device cuda \
            --model_id /data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k \
            --dataset_id imagenet-1k \
            --batch_size 32 \
            --theta 2000 \
            --precision float64 \
            --quick-test \
            --spiking-layernorm \
            --spiking-mlp \
            --spiking-attention \
            --model_backend spiking \
            --gaussian-time-noise \
            --time-noise-std-frac "$time_noise_std_frac" \
            --time-noise-mean 0 \
            --time-noise-deadline-margin-std "$margin" \
            --time-noise-seed "$seed"
    ) >"$temporary_log" 2>&1 && accuracy="$(log_accuracy "$temporary_log")"; then
        mv "$temporary_log" "$log_path"
        record_status "complete margin_std=$margin seed=$seed accuracy=$accuracy"
    else
        mv "$temporary_log" "$log_dir/margin_${slug}_seed_${seed}.failed.$(date -u +%Y%m%dT%H%M%SZ).log"
        record_status "failed margin_std=$margin seed=$seed"
        return 1
    fi
}

record_status "sweep-start baseline=$baseline_accuracy threshold=$threshold r_t=$time_noise_std_frac"
recovered_margin=""
for margin in "${margins[@]}"; do
    run_condition "$margin" 0
    accuracy="$(log_accuracy "$log_dir/margin_$(margin_slug "$margin")_seed_0.log")"
    if "$python_bin" - "$accuracy" "$threshold" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
    then
        run_condition "$margin" 1
        run_condition "$margin" 2
        replicated_mean="$($python_bin - "$log_dir" "$(margin_slug "$margin")" <<'PY'
import re, sys
from pathlib import Path
values=[]
for path in Path(sys.argv[1]).glob(f"margin_{sys.argv[2]}_seed_*.log"):
    matches=re.findall(r"^Accuracy: ([0-9.eE+-]+)$", path.read_text(errors="replace"), re.MULTILINE)
    if matches: values.append(float(matches[-1]))
if len(values) != 3: raise SystemExit("recovery confirmation requires three seeds")
print(sum(values)/len(values))
PY
)"
        if "$python_bin" - "$replicated_mean" "$threshold" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
        then
            recovered_margin="$margin"
            record_status "recovered margin_std=$margin mean_accuracy=$replicated_mean"
            break
        fi
        record_status "confirmation-below-threshold margin_std=$margin mean_accuracy=$replicated_mean"
    fi
done

"$python_bin" scripts/analysis/summarize_deadline_margin_sweep.py \
    --log-dir "$log_dir" \
    --baseline "$baseline_accuracy" \
    --recovery-tolerance "$recovery_tolerance" \
    --raw-csv "$log_dir/raw_runs.csv" \
    --summary-csv "$log_dir/summary.csv" \
    --figure-prefix "$figure_prefix"

if [[ -z "$recovered_margin" ]]; then
    record_status "not-recovered max_margin_std=${margins[-1]}"
    exit 3
fi
record_status "sweep-complete recovered_margin_std=$recovered_margin"
