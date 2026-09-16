#!/bin/bash
#SBATCH --job-name=calibrated-noise
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

: "${CALIBRATED_DEPLOYMENT:?CALIBRATED_DEPLOYMENT is required}"
: "${SLURM_JOB_ID:?Slurm allocation is required}"
check_only="${CALIBRATED_CHECK_ONLY:-0}"
[[ "$check_only" == 0 || "$check_only" == 1 ]]
task_index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$task_index" =~ ^[0-9]+$ ]]
if [[ "$check_only" == 0 ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]
    [[ "$CUDA_VISIBLE_DEVICES" != -1 && "$CUDA_VISIBLE_DEVICES" != all && "$CUDA_VISIBLE_DEVICES" != none ]]
fi

control_python=/home1/sizz1997/miniconda3/bin/python
deployment_dir="$(dirname "$CALIBRATED_DEPLOYMENT")"
worker="$deployment_dir/code/run_calibrated_noise_task.py"
runtime_values="$("$control_python" "$worker" --deployment "$CALIBRATED_DEPLOYMENT" --runtime-values)"
mapfile -t config <<< "$runtime_values"
[[ "${#config[@]}" -eq 11 ]]
host_source="${config[0]}"
host_repository="${config[1]}"
host_assets="${config[2]}"
host_experiment="${config[3]}"
host_deployment="${config[4]}"
env_archive="${config[5]}"
env_sha256="${config[6]}"
env_unpacked_bytes="${config[7]}"
container_image="${config[8]}"
container_sha256="${config[9]}"
minimum_scratch_bytes="${config[10]}"
[[ "$(readlink -f "$deployment_dir")" == "$(readlink -f "$host_deployment")" ]]

task_uid="$(id -u)"
enroot_data="/enroot/$task_uid/data"
mkdir -p "$enroot_data"
[[ "$(readlink -f "$enroot_data")" == "$enroot_data" ]]
filesystem="$(findmnt -n -T "$enroot_data" -o FSTYPE)"
case "$filesystem" in
    ''|tmpfs|ramfs)
        echo "Refusing a memory-backed runtime filesystem: $filesystem" >&2
        exit 2
        ;;
esac
runtime_parent="$(mktemp -d --tmpdir="$enroot_data" "calibrated-noise-${SLURM_JOB_ID}-${task_index}.XXXXXXXX")"
cleanup() {
    case "$runtime_parent" in
        "$enroot_data"/calibrated-noise-*)
            if [[ -d "$runtime_parent" && ! -L "$runtime_parent" && "$(stat -c %u "$runtime_parent")" == "$task_uid" ]]; then
                rm -rf -- "$runtime_parent"
            fi
            ;;
        *) echo "Refusing unexpected cleanup target: $runtime_parent" >&2 ;;
    esac
}
trap cleanup EXIT
trap 'exit 143' TERM INT

actual_sha256="$(sha256sum "$env_archive")"
[[ "${actual_sha256%% *}" == "$env_sha256" ]]
actual_sha256="$(sha256sum "$container_image")"
[[ "${actual_sha256%% *}" == "$container_sha256" ]]

# Serialize space admission and extraction; retained environments remain visible to df.
exec 9> "$enroot_data/calibrated-noise-extraction.lock"
flock 9
available_bytes="$(df -B1 --output=avail "$enroot_data" | tail -n 1)"
required_bytes="$((env_unpacked_bytes + minimum_scratch_bytes))"
if (( available_bytes < required_bytes )); then
    echo "Insufficient disk space for the environment and task cache" >&2
    exit 2
fi
mkdir -p "$runtime_parent/scratch"
zstd --decompress --stdout "$env_archive" | tar -xf - -C "$runtime_parent"
test -x "$runtime_parent/dt/bin/python"
actual_bytes="$(du -s -B1 "$runtime_parent" | cut -f1)"
if (( actual_bytes > env_unpacked_bytes )); then
    echo "Extracted environment exceeded its reserved disk space" >&2
    exit 2
fi
flock -u 9
exec 9>&-

source_root=/data/delayed-temporal-worktrees/gelu-timeconstant-noise
repository_root=/data/delayed-temporal
experiment_root="$repository_root/artifacts/logs/noise_scan/vit_base_noise_calibrated_theta40_float64_v1"
mounts="$host_repository:$repository_root:ro,$host_source:$source_root:ro"
mounts+=",$host_assets:$repository_root/artifacts/assets/theta-selection-v1:ro"
mounts+=",$host_assets/source-checkouts/transformers:$repository_root/src/transformers:ro"
mounts+=",$host_assets/source-checkouts/spikingjelly:$repository_root/src/spikingjelly:ro"
mounts+=",$host_assets/source-checkouts/transformers:$source_root/src/transformers:ro"
mounts+=",$host_assets/source-checkouts/spikingjelly:$source_root/src/spikingjelly:ro"
mounts+=",$host_experiment:$experiment_root,$host_deployment:/calibrated-deployment:ro"
mounts+=",$runtime_parent/dt:/opt/conda/envs/dt,$runtime_parent/scratch:/work-tmp"

worker_arguments=(--task-index "$task_index")
gpu_arguments=(--gres=gpu:1)
if [[ "$check_only" == 1 ]]; then
    worker_arguments=(--check-only)
    gpu_arguments=(--gres=none)
    export NVIDIA_VISIBLE_DEVICES=void
else
    export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
fi
srun --ntasks=1 "${gpu_arguments[@]}" \
    --container-image="$container_image" --container-mounts="$mounts" \
    --container-workdir="$source_root" \
    /usr/bin/env -u WANDB_API_KEY \
    PATH="/calibrated-deployment/tools:$PATH" \
    TMPDIR=/work-tmp TEMP=/work-tmp TMP=/work-tmp \
    XDG_CACHE_HOME=/work-tmp/cache HF_HOME=/work-tmp/huggingface \
    TORCH_HOME=/work-tmp/torch TRITON_CACHE_DIR=/work-tmp/triton \
    CUDA_CACHE_PATH=/work-tmp/cuda MPLCONFIGDIR=/work-tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$source_root:$source_root/src/transformers/src:$source_root/src/spikingjelly" \
    WANDB_MODE=disabled WANDB_SILENT=true WANDB_CONSOLE=off \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    /opt/conda/envs/dt/bin/python /calibrated-deployment/code/run_calibrated_noise_task.py \
    --deployment /calibrated-deployment/deployment.json \
    "${worker_arguments[@]}"
