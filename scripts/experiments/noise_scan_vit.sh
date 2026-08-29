#!/bin/bash

# Run the maintained ViT Gaussian timing-noise fine scan on one process per GPU.
# The legacy jitter/hazard logs remain in artifacts/logs/noise_scan; this script
# writes the replacement experiment into a tagged child directory instead.

set -u
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

source "$repo_root/scripts/lib/gpu_pool.sh"

# These environment controls make a long scan reproducible and safely resumable.
# PYTHON_BIN must name a Python executable; running this script through
# ``conda run -n dt`` supplies the repository's complete evaluation environment.
read -r -a cuda_devices <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a time_noise_seeds <<< "${TIME_NOISE_SEEDS:-0 1 2}"
python_bin="${PYTHON_BIN:-python}"
scan_tag="${SCAN_TAG:-gaussian_v1}"
scan_model_label="${SCAN_MODEL_LABEL:-ViT-S}"
force="${FORCE:-0}"
torch_num_threads="${TORCH_NUM_THREADS:-4}"

device="cuda"
# The defaults reproduce the canonical ViT-S sweep. A distinct SCAN_TAG and
# SCAN_FIGURE_PREFIX let a follow-up architecture reuse the exact protocol
# without overwriting either the ViT-S logs or its canonical figure.
model_id="${MODEL_ID:-/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k}"
dataset_id="imagenet-1k"
batch_size="${BATCH_SIZE:-32}"
theta=2000

scan_logdir="${SCAN_LOGDIR:-$repo_root/artifacts/logs/noise_scan/$scan_tag}"
manifest_path="$scan_logdir/expected_runs.tsv"
raw_csv_path="$scan_logdir/raw_runs.csv"
summary_csv_path="$scan_logdir/summary.csv"
figure_prefix="${SCAN_FIGURE_PREFIX:-$repo_root/artifacts/figures/noise_robustness}"
mkdir -p "$scan_logdir"

# The defaults retain the original eleven-point grids. Explicit environment
# lists let a tagged follow-up refine only the physically relevant axis without
# editing this script or rerunning an unrelated static-mismatch grid.
default_time_noise_std_fracs=(
    1.000e-06 1.900e-06 2.800e-06 3.700e-06 4.600e-06 5.500e-06
    6.400e-06 7.300e-06 8.200e-06 9.100e-06 1.000e-05
)
default_mismatch_theta_stds=(
    1.000e-05 1.400e-05 1.800e-05 2.200e-05 2.600e-05 3.000e-05
    3.400e-05 3.800e-05 4.200e-05 4.600e-05 5.000e-05
)
if [[ -v TIME_NOISE_STD_FRACS ]]; then
    read -r -a time_noise_std_fracs <<< "$TIME_NOISE_STD_FRACS"
else
    time_noise_std_fracs=("${default_time_noise_std_fracs[@]}")
fi
if [[ -v MISMATCH_THETA_STDS ]]; then
    read -r -a mismatch_theta_stds <<< "$MISMATCH_THETA_STDS"
else
    mismatch_theta_stds=("${default_mismatch_theta_stds[@]}")
fi

if [[ ${#cuda_devices[@]} -eq 0 ]]; then
    echo "GPUS must contain at least one CUDA device" >&2
    exit 2
fi
if [[ ${#time_noise_seeds[@]} -lt 2 ]]; then
    echo "TIME_NOISE_SEEDS must contain at least two seeds for a confidence interval" >&2
    exit 2
fi
if [[ ${#time_noise_std_fracs[@]} -eq 0 ]]; then
    echo "TIME_NOISE_STD_FRACS must contain at least one positive fraction" >&2
    exit 2
fi
if [[ "$force" != "0" && "$force" != "1" ]]; then
    echo "FORCE must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$torch_num_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "TORCH_NUM_THREADS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ "$model_id" = /* && ! -e "$model_id" ]]; then
    echo "MODEL_ID does not exist: $model_id" >&2
    exit 2
fi

# Fail before scheduling GPUs if the selected interpreter cannot import the model
# stack. This avoids producing 45 nearly identical dependency-failure logs.
if ! "$python_bin" -c "import torch, transformers, datasets, wandb"; then
    echo "PYTHON_BIN lacks evaluation dependencies; run with 'conda run -n dt bash ...'" >&2
    exit 2
fi

write_manifest() {
    "${python_bin}" - \
        "$manifest_path" \
        "$scan_tag" \
        "${time_noise_seeds[*]}" \
        "${time_noise_std_fracs[*]}" \
        "${mismatch_theta_stds[*]}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
scan_tag = sys.argv[2]
seeds = tuple(sys.argv[3].split())
fractions = tuple(sys.argv[4].split())
mismatches = tuple(sys.argv[5].split())
# Keeping all planned conditions in one manifest makes resume checks and
# aggregation share one source of truth instead of duplicating expected counts.
if not seeds:
    raise SystemExit("manifest generation received no timing-noise seeds")

rows = [{
    "axis": "baseline",
    "magnitude": "0",
    "seed": "",
    "experiment_name": f"{scan_tag}-noise_off_baseline",
    "log_file": "noise_off_baseline.log",
}]
for fraction in fractions:
    for seed in seeds:
        run_name = f"A_gaussian_frac_{fraction}_seed_{seed}"
        rows.append({
            "axis": "gaussian",
            "magnitude": fraction,
            "seed": seed,
            "experiment_name": f"{scan_tag}-{run_name}",
            "log_file": f"{run_name}.log",
        })
for mismatch in mismatches:
    run_name = f"B_mismatch_{mismatch}"
    rows.append({
        "axis": "mismatch",
        "magnitude": mismatch,
        "seed": "",
        "experiment_name": f"{scan_tag}-{run_name}",
        "log_file": f"{run_name}.log",
    })

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("axis", "magnitude", "seed", "experiment_name", "log_file"),
        dialect="excel-tab",
    )
    writer.writeheader()
    writer.writerows(rows)
PY
}

# Pass the configured seed list as one quoted argument without interpreting it as
# shell code. The resulting file describes one baseline, every Gaussian
# fraction/seed pair, and each optional static-mismatch magnitude.
write_manifest

is_complete_log() {
    local log_path="$1"
    local original_size
    local without_nul_size

    # A missing file is an ordinary pending condition, so return before the
    # structural checks attempt to open it.
    [[ -f "$log_path" ]] || return 1

    # Concurrent or interrupted writers can leave sparse NUL regions even when
    # a later writer reaches the final accuracy line. Treat such a file as
    # structurally incomplete so resume regenerates it instead of aggregating a
    # mixture of two process outputs.
    original_size="$(wc -c < "$log_path")"
    without_nul_size="$(tr -d '\000' < "$log_path" | wc -c)"

    # A final accuracy line is the evaluator's completion marker. Tracebacks take
    # precedence so a partially successful process is never reused as completed.
    [[ "$original_size" -eq "$without_nul_size" ]] \
        && grep -Eq '^Accuracy: [0-9]+([.][0-9]+)?$' "$log_path" \
        && ! grep -q 'Traceback (most recent call last)' "$log_path"
}

launch_run() {
    local run_name="$1"
    local experiment_name="$2"
    shift 2
    local log_path="$scan_logdir/$run_name.log"

    # Resume leaves every verified result untouched. FORCE=1 deliberately reruns
    # the condition and replaces only its new tagged log, never a legacy artifact.
    if [[ "$force" == "0" ]] && is_complete_log "$log_path"; then
        echo "Skipping completed run: $run_name"
        return
    fi

    gpu_pool_acquire
    local gpu="$GPU_POOL_ACQUIRED"
    echo "Running noise scan on GPU $gpu: $run_name"

    # Each child sees exactly one CUDA device, avoiding DataParallel with the
    # process-wide Gaussian generator. W&B groups all replicas under the scan tag.
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export WANDB_RUN_GROUP="$scan_tag"
        # Eight independent PyTorch processes must not each create a host-wide
        # 128-thread pool. Bound common BLAS/OpenMP runtimes per replica so GPU jobs
        # do not starve one another on image preprocessing and launch bookkeeping.
        export OMP_NUM_THREADS="$torch_num_threads"
        export MKL_NUM_THREADS="$torch_num_threads"
        export OPENBLAS_NUM_THREADS="$torch_num_threads"
        export NUMEXPR_NUM_THREADS="$torch_num_threads"
        "$python_bin" scripts/evaluation/error_analysis_vit.py \
            --experiment_name "$experiment_name" \
            --device "$device" \
            --model_id "$model_id" \
            --dataset_id "$dataset_id" \
            --batch_size "$batch_size" \
            --theta "$theta" \
            --quick-test \
            --spiking-layernorm \
            --spiking-mlp \
            --spiking-attention \
            --model_backend spiking \
            "$@" \
            > "$log_path" 2>&1
    ) &
    gpu_pool_register "$!" "$gpu"
}

gpu_pool_init "${cuda_devices[@]}"

# The deterministic baseline is evaluated once because model initialization,
# subset selection, and loader order all use the evaluator's fixed seed 42.
launch_run \
    "noise_off_baseline" \
    "$scan_tag-noise_off_baseline"

# Timing seeds alter only the dedicated Gaussian generator. The model, data subset,
# and ordering therefore remain fixed while the stochastic event replica changes.
for time_noise_std_frac in "${time_noise_std_fracs[@]}"; do
    for time_noise_seed in "${time_noise_seeds[@]}"; do
        run_name="A_gaussian_frac_${time_noise_std_frac}_seed_${time_noise_seed}"
        launch_run \
            "$run_name" \
            "$scan_tag-$run_name" \
            --gaussian-time-noise \
            --time-noise-std-frac "$time_noise_std_frac" \
            --time-noise-mean 0.0 \
            --time-noise-seed "$time_noise_seed"
    done
done

# Static threshold mismatch remains a separate fixed draw under torch seed 42.
# It is evaluated once per magnitude and never combined with timing noise.
for mismatch_theta_std in "${mismatch_theta_stds[@]}"; do
    run_name="B_mismatch_${mismatch_theta_std}"
    launch_run \
        "$run_name" \
        "$scan_tag-$run_name" \
        --mismatch-enabled \
        --mismatch-theta-std "$mismatch_theta_std"
done

# Some jobs have already been reaped by gpu_pool_acquire; this wait covers every
# still-running child. Completeness is validated from logs rather than wait order.
wait || true

"$python_bin" scripts/analysis/summarize_noise_scan.py \
    --log-dir "$scan_logdir" \
    --manifest "$manifest_path" \
    --raw-csv "$raw_csv_path" \
    --summary-csv "$summary_csv_path" \
    --figure-prefix "$figure_prefix" \
    --model-label "$scan_model_label"
