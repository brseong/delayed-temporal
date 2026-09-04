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
readonly allowed_gpu_list="4 5 6 7"
read -r -a cuda_devices <<< "${GPUS:-$allowed_gpu_list}"
read -r -a replica_seeds <<< "${REPLICA_SEEDS:-${TIME_NOISE_SEEDS:-0 1 2}}"
python_bin="${PYTHON_BIN:-python}"
scan_protocol="${SCAN_PROTOCOL:-quick}"
precision="${PRECISION:-float64}"
scan_tag="${SCAN_TAG:-vit_base_noise_${scan_protocol}_${precision}_v2}"
scan_model_label="${SCAN_MODEL_LABEL:-ViT-B/16}"
force="${FORCE:-0}"
torch_num_threads="${TORCH_NUM_THREADS:-4}"
publish_paper_figure="${PUBLISH_PAPER_FIGURE:-0}"
dry_run="${NOISE_SCAN_DRY_RUN:-0}"
require_idle_gpus="${REQUIRE_IDLE_GPUS:-1}"

device="cuda"
# The canonical manuscript protocol evaluates the local timm ViT-B/16 checkpoint.
model_id="${MODEL_ID:-/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k}"
dataset_id="imagenet-1k"
batch_size="${BATCH_SIZE:-32}"
theta="${THETA:-2000}"

scan_logdir="${SCAN_LOGDIR:-$repo_root/artifacts/logs/noise_scan/$scan_tag}"
manifest_path="$scan_logdir/expected_runs.tsv"
raw_csv_path="$scan_logdir/raw_runs.csv"
summary_csv_path="$scan_logdir/summary.csv"
figure_prefix="${SCAN_FIGURE_PREFIX:-$repo_root/artifacts/figures/noise_robustness_vit_base_${scan_protocol}_${precision}_v2}"
paper_figure="$repo_root/paper/figures/noise-robustness-vit-base.pdf"
mkdir -p "$scan_logdir"

# The quick protocol resolves both transition curves on the fixed 5,000-image
# subset. The full protocol confirms only three representative conditions per axis.
if [[ "$scan_protocol" == "quick" ]]; then
    default_time_noise_std_fracs=(
        1.000e-10 1.250e-10 1.500e-10 1.750e-10 2.000e-10 2.500e-10
        3.162e-10 4.000e-10 5.000e-10 6.300e-10 8.000e-10 1.000e-09
    )
    default_mismatch_theta_stds=(
        1.000e-05 1.400e-05 1.800e-05 2.200e-05 2.600e-05 3.000e-05
        3.400e-05 3.800e-05 4.200e-05 4.600e-05 5.000e-05
    )
    evaluation_scope_args=(--quick-test)
elif [[ "$scan_protocol" == "full" ]]; then
    default_time_noise_std_fracs=(1.000e-10 2.500e-10 4.000e-10)
    default_mismatch_theta_stds=(1.000e-05 3.000e-05 5.000e-05)
    evaluation_scope_args=()
else
    echo "SCAN_PROTOCOL must be quick or full" >&2
    exit 2
fi
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
declare -A seen_gpus=()
for gpu in "${cuda_devices[@]}"; do
    if [[ " $allowed_gpu_list " != *" $gpu "* ]]; then
        echo "GPUS may contain only physical GPUs 4, 5, 6, and 7" >&2
        exit 2
    fi
    if [[ -v "seen_gpus[$gpu]" ]]; then
        echo "GPUS must not contain duplicate device $gpu" >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
done
if [[ ${#replica_seeds[@]} -lt 2 ]]; then
    echo "REPLICA_SEEDS must contain at least two seeds for confidence intervals" >&2
    exit 2
fi
for seed in "${replica_seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "REPLICA_SEEDS must contain non-negative integers" >&2
        exit 2
    fi
done
if [[ ${#time_noise_std_fracs[@]} -eq 0 ]]; then
    echo "TIME_NOISE_STD_FRACS must contain at least one positive fraction" >&2
    exit 2
fi
if [[ "$force" != "0" && "$force" != "1" ]]; then
    echo "FORCE must be 0 or 1" >&2
    exit 2
fi
if [[ "$publish_paper_figure" != "0" && "$publish_paper_figure" != "1" ]]; then
    echo "PUBLISH_PAPER_FIGURE must be 0 or 1" >&2
    exit 2
fi
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
    echo "NOISE_SCAN_DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ "$require_idle_gpus" != "0" && "$require_idle_gpus" != "1" ]]; then
    echo "REQUIRE_IDLE_GPUS must be 0 or 1" >&2
    exit 2
fi
if [[ "$publish_paper_figure" == "1" && "$scan_protocol" != "quick" ]]; then
    echo "Only the quick transition scan may publish the manuscript figure" >&2
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
if [[ "$precision" != "float32" && "$precision" != "float64" ]]; then
    echo "PRECISION must be float32 or float64 for the manuscript noise scan" >&2
    exit 2
fi
if [[ ! "$theta" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$theta" == "0" ]]; then
    echo "THETA must be a positive decimal number" >&2
    exit 2
fi
if [[ "$model_id" = /* && ! -e "$model_id" ]]; then
    echo "MODEL_ID does not exist: $model_id" >&2
    exit 2
fi

write_manifest() {
    "${python_bin}" - \
        "$manifest_path" \
        "$scan_tag" \
        "${replica_seeds[*]}" \
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
    for seed in seeds:
        run_name = f"B_mismatch_{mismatch}_seed_{seed}"
        rows.append({
            "axis": "mismatch",
            "magnitude": mismatch,
            "seed": seed,
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

if [[ "$dry_run" == "1" ]]; then
    echo "Dry run validated GPUs: ${cuda_devices[*]}"
    echo "Dry run wrote manifest: $manifest_path"
    exit 0
fi

if [[ "$require_idle_gpus" == "1" ]]; then
    for gpu in "${cuda_devices[@]}"; do
        active_pids="$(
            nvidia-smi --id="$gpu" --query-compute-apps=pid \
                --format=csv,noheader,nounits 2>/dev/null
        )"
        if [[ -n "$active_pids" ]]; then
            echo "GPU $gpu is already occupied by compute process(es): $active_pids" >&2
            echo "Wait for GPUs 4-7 to become idle before starting this scan" >&2
            exit 2
        fi
    done
fi

# Fail before scheduling GPUs if the selected interpreter cannot import the model
# stack. This avoids producing many identical dependency-failure logs.
if ! "$python_bin" -c "import torch, transformers, datasets, wandb"; then
    echo "PYTHON_BIN lacks evaluation dependencies; run with 'conda run -n dt bash ...'" >&2
    exit 2
fi

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
        && grep -q '^Evaluation metadata — ' "$log_path" \
        && ! grep -q 'Traceback (most recent call last)' "$log_path" \
        || return 1
    case "$(basename "$log_path")" in
        A_gaussian_*) grep -q '^Gaussian\[' "$log_path" ;;
        B_mismatch_*) grep -q '^Static threshold mismatch — enabled: True' "$log_path" ;;
        *) return 0 ;;
    esac
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
            --precision "$precision" \
            "${evaluation_scope_args[@]}" \
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
    for time_noise_seed in "${replica_seeds[@]}"; do
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

# Static threshold mismatch uses its own dedicated seed and is never combined with
# timing noise. Each replica retains one frozen offset draw for the full evaluation.
for mismatch_theta_std in "${mismatch_theta_stds[@]}"; do
    for mismatch_seed in "${replica_seeds[@]}"; do
        run_name="B_mismatch_${mismatch_theta_std}_seed_${mismatch_seed}"
        launch_run \
            "$run_name" \
            "$scan_tag-$run_name" \
            --mismatch-enabled \
            --mismatch-theta-std "$mismatch_theta_std" \
            --mismatch-seed "$mismatch_seed"
    done
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

if [[ "$publish_paper_figure" == "1" ]]; then
    cp "$figure_prefix.pdf" "$paper_figure"
    echo "Published validated manuscript figure: $paper_figure"
fi
