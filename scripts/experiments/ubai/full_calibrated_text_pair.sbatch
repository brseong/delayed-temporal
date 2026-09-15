#!/bin/bash
#SBATCH --job-name=full-text-comparison
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?a Slurm allocation is required}"
: "${CAMPAIGN_SOURCE:?CAMPAIGN_SOURCE is required}"
: "${CAMPAIGN_COMMIT:?CAMPAIGN_COMMIT is required}"
: "${CAMPAIGN_ASSETS:?CAMPAIGN_ASSETS is required}"
: "${CAMPAIGN_OUTPUT:?CAMPAIGN_OUTPUT is required}"
: "${TRANSFORMERS_SOURCE:?TRANSFORMERS_SOURCE is required}"
: "${SPIKINGJELLY_SOURCE:?SPIKINGJELLY_SOURCE is required}"
: "${PORTABLE_ENV_ARCHIVE:?PORTABLE_ENV_ARCHIVE is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"

tag=conversion_comparison_theta40_calibrated_float64_bounds3_v3
runtime="/enroot/${UID}/${tag}/${SLURM_JOB_ID}"
mkdir -p "$runtime/dt" "$runtime/tmp" "$runtime/bert" "$runtime/roberta"
filesystem=$(findmnt -n -o FSTYPE -T "$runtime")
if [[ "$filesystem" == tmpfs || "$filesystem" == ramfs ]]; then
    echo "refusing memory-backed runtime: $runtime ($filesystem)" >&2
    exit 2
fi
available=$(df --output=avail -B1 "$runtime" | tail -1 | tr -d ' ')
if (( available < 120 * 1024 * 1024 * 1024 )); then
    echo "less than 120 GiB is available below /enroot" >&2
    exit 2
fi

children=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for child in "${children[@]:-}"; do
        if kill -0 "$child" 2>/dev/null; then
            kill -TERM "$child" 2>/dev/null || true
        fi
    done
    wait || true
    if [[ -d "$runtime" && "$runtime" == /enroot/"${UID}"/"${tag}"/"${SLURM_JOB_ID}" ]]; then
        rm -rf -- "$runtime"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

tar --use-compress-program=unzstd -xf "$PORTABLE_ENV_ARCHIVE" -C "$runtime/dt"
python_root="$runtime/dt"
if [[ -x "$runtime/dt/bin/python" ]]; then
    python_root="$runtime/dt"
elif [[ -x "$runtime/dt/dt/bin/python" ]]; then
    python_root="$runtime/dt/dt"
else
    echo "portable Python was not found after extraction" >&2
    exit 2
fi

source_inside=/campaign-source
assets_inside=/campaign-assets
artifacts_inside=/campaign-artifacts
runtime_inside="$runtime"
mounts=(
    "$CAMPAIGN_SOURCE:$source_inside:ro"
    "$TRANSFORMERS_SOURCE:$source_inside/src/transformers/src:ro"
    "$SPIKINGJELLY_SOURCE:$source_inside/src/spikingjelly/spikingjelly:ro"
    "$CAMPAIGN_ASSETS:$assets_inside:ro"
    "$CAMPAIGN_OUTPUT:$artifacts_inside/logs/conversion_comparison/$tag"
    "$python_root:/opt/conda/envs/dt:ro"
    "$runtime:$runtime_inside"
    "$runtime/tmp:/tmp"
    "$runtime/tmp:/var/tmp"
)
mount_csv=$(IFS=,; echo "${mounts[*]}")

run_family() {
    family=$1
    model=$2
    train=$3
    train_fingerprint=$4
    evaluation=$5
    evaluation_fingerprint=$6
    srun --exclusive --exact --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$mount_csv" \
        --container-workdir="$source_inside" \
        /usr/bin/env \
        PATH=/opt/conda/envs/dt/bin:/usr/bin:/bin \
        PYTHONPATH="$source_inside:$source_inside/src/transformers/src:$source_inside/src/spikingjelly" \
        DELAYED_TEMPORAL_ARTIFACTS_ROOT="$artifacts_inside" \
        WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        TMPDIR="$runtime_inside/tmp" TMP="$runtime_inside/tmp" TEMP="$runtime_inside/tmp" \
        /opt/conda/envs/dt/bin/python -u \
        "$source_inside/scripts/experiments/run_full_calibrated_text_comparison.py" \
        --family "$family" --source-root "$source_inside" --expected-commit "$CAMPAIGN_COMMIT" \
        --model-id "$assets_inside/checkpoints/$model" \
        --calibration-dataset-path "$assets_inside/datasets/$train" \
        --calibration-dataset-fingerprint "$train_fingerprint" \
        --evaluation-dataset-path "$assets_inside/datasets/$evaluation" \
        --evaluation-dataset-fingerprint "$evaluation_fingerprint" \
        --gpu 0 --host-label ubai --runtime-root "$runtime_inside" \
        --output-root "$artifacts_inside/logs/conversion_comparison/$tag/text/$family"
}

run_family bert bert sst2/train_seed0_5000 "$SST2_TRAIN_FINGERPRINT" \
    sst2/validation_872 "$SST2_VALIDATION_FINGERPRINT" &
children+=("$!")
run_family roberta roberta sst2/train_seed0_5000 "$SST2_TRAIN_FINGERPRINT" \
    sst2/validation_872 "$SST2_VALIDATION_FINGERPRINT" &
children+=("$!")

failure=0
for child in "${children[@]}"; do
    wait "$child" || failure=1
done
exit "$failure"
