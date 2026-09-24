#!/bin/bash
#SBATCH --job-name=vit-bss2-depth
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${DEPTH_RUNNER_SOURCE:?DEPTH_RUNNER_SOURCE is required}"
: "${DEPTH_EVALUATOR_SOURCE:?DEPTH_EVALUATOR_SOURCE is required}"
: "${DEPTH_ASSETS:?DEPTH_ASSETS is required}"
: "${DEPTH_EXPERIMENT:?DEPTH_EXPERIMENT is required}"
: "${DEPTH_ENV_ARCHIVE:?DEPTH_ENV_ARCHIVE is required}"
: "${DEPTH_CONTAINER_IMAGE:?DEPTH_CONTAINER_IMAGE is required}"
: "${SLURM_JOB_ID:?Slurm allocation is required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"

task_index="$SLURM_ARRAY_TASK_ID"
[[ "$task_index" =~ ^[0-9]+$ && "$task_index" -lt 30 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]
evaluation_samples="${DEPTH_EVALUATION_SAMPLES:-5000}"
[[ "$evaluation_samples" == 500 || "$evaluation_samples" == 5000 ]]

scales=(0.03 0.1)
prefixes=(1 2 4 8 12)
scale_index="$((task_index / 15))"
within_scale="$((task_index % 15))"
prefix_index="$((within_scale / 3))"
seed="$((within_scale % 3))"
scale="${scales[$scale_index]}"
prefix="${prefixes[$prefix_index]}"

task_uid="$(id -u)"
enroot_data="/enroot/$task_uid/data"
mkdir -p "$enroot_data"
[[ "$(findmnt -n -T "$enroot_data" -o FSTYPE)" != tmpfs ]]
runtime_parent="$(mktemp -d --tmpdir="$enroot_data" "vit-bss2-depth-${SLURM_JOB_ID}-${task_index}.XXXXXXXX")"
cleanup() {
    case "$runtime_parent" in
        "$enroot_data"/vit-bss2-depth-*) rm -rf -- "$runtime_parent" ;;
        *) echo "Refusing unexpected cleanup target: $runtime_parent" >&2 ;;
    esac
}
trap cleanup EXIT
trap 'exit 143' TERM INT

exec 9> "$enroot_data/vit-bss2-depth-extraction.lock"
flock 9
mkdir -p "$runtime_parent/scratch"
zstd --decompress --stdout "$DEPTH_ENV_ARCHIVE" | tar -xf - -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"
flock -u 9
exec 9>&-

runner_root=/data/depth-runner
evaluator_root=/data/depth-evaluator
assets_root=/data/depth-assets
experiment_root=/data/depth-experiment
mounts="$DEPTH_RUNNER_SOURCE:$runner_root:ro,$DEPTH_EVALUATOR_SOURCE:$evaluator_root:ro"
mounts+=",$DEPTH_ASSETS:$assets_root:ro,$DEPTH_EXPERIMENT:$experiment_root"
mounts+=",$DEPTH_ASSETS/source-checkouts/transformers:$runner_root/src/transformers:ro"
mounts+=",$DEPTH_ASSETS/source-checkouts/spikingjelly:$runner_root/src/spikingjelly:ro"
mounts+=",$DEPTH_ASSETS/source-checkouts/transformers:$evaluator_root/src/transformers:ro"
mounts+=",$DEPTH_ASSETS/source-checkouts/spikingjelly:$evaluator_root/src/spikingjelly:ro"
mounts+=",$runtime_parent/dt:/opt/conda/envs/dt,$runtime_parent/scratch:/work-tmp"

export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
srun --ntasks=1 --gres=gpu:1 \
    --container-image="$DEPTH_CONTAINER_IMAGE" \
    --container-mounts="$mounts" \
    --container-workdir="$runner_root" \
    /usr/bin/env -u WANDB_API_KEY \
    DELAYED_TEMPORAL_ARTIFACTS_ROOT="$experiment_root/artifacts" \
    TMPDIR=/work-tmp TEMP=/work-tmp TMP=/work-tmp \
    XDG_CACHE_HOME=/work-tmp/cache HF_HOME=/work-tmp/huggingface \
    TORCH_HOME=/work-tmp/torch TRITON_CACHE_DIR=/work-tmp/triton \
    CUDA_CACHE_PATH=/work-tmp/cuda MPLCONFIGDIR=/work-tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$evaluator_root:$evaluator_root/src/transformers/src:$evaluator_root/src/spikingjelly" \
    WANDB_MODE=disabled WANDB_SILENT=true WANDB_CONSOLE=off \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    /opt/conda/envs/dt/bin/python \
    "$runner_root/scripts/experiments/run_vit_bss2_depth_condition.py" \
    --execution-mode slurm --runtime-root /work-tmp \
    --source-root "$evaluator_root" \
    --expected-commit 52fd758dfee41c9eeb8f8124b6c1dde9ff16bc0f \
    --model-id "$assets_root/checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k" \
    --calibration-source "$experiment_root/input/calibration_source" \
    --calibration-dataset-path "$assets_root/datasets/imagenet_theta_selection_v1/train_seed0_5000" \
    --calibration-dataset-fingerprint cabf903d14d1b1ac \
    --evaluation-dataset-path "$assets_root/datasets/imagenet_theta_selection_v1/validation_50000" \
    --evaluation-dataset-fingerprint 746378cc7befed99 \
    --image-preprocessing-config "$evaluator_root/scripts/configs/vit_timm_preprocessing.json" \
    --hardware-summary "$experiment_root/input/20260924T_best_median_screen_summary.json" \
    --condition screening-median --measured-noise-scale "$scale" \
    --first-block-count "$prefix" --seed "$seed" \
    --evaluation-samples "$evaluation_samples" \
    --python-bin /opt/conda/envs/dt/bin/python
