#!/bin/bash

# Build and transfer only the immutable assets needed by the UBAI theta sweep.

set -euo pipefail

if [[ "$(hostname)" != "baekryun-cuda129" ]]; then
    echo "Asset staging must run on baekryun-cuda129" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

stage_root="${THETA_STAGE_ROOT:-/data/nas/ubai_stage/theta-selection-v1}"
remote_host="${THETA_REMOTE_HOST:-gate1}"
remote_assets="${THETA_REMOTE_ASSETS:-/home1/sizz1997/myubai/delayed-temporal-assets/theta-selection-v1}"
python_bin="${PYTHON_BIN:-/opt/conda/envs/dt/bin/python}"
environment_root="${THETA_ENV_ROOT:-/opt/conda/envs/dt}"
checkpoint_source="${THETA_CHECKPOINT_SOURCE:-/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k}"
dataset_cache="${THETA_DATASET_CACHE:-/data/nas/datasets}"

mkdir -p \
    "$stage_root/datasets" \
    "$stage_root/checkpoints" \
    "$stage_root/runtime" \
    "$stage_root/source-checkouts"
dataset_root="$stage_root/datasets/imagenet_theta_selection_v1"
if [[ ! -f "$dataset_root/manifest.json" ]]; then
    if [[ -e "$dataset_root" ]]; then
        echo "Incomplete dataset artifact exists: $dataset_root" >&2
        exit 2
    fi
    "$python_bin" scripts/setup/prepare_imagenet_theta_selection.py \
        --output-root "$dataset_root" \
        --cache-dir "$dataset_cache"
fi

checkpoint_name="$(basename "$checkpoint_source")"
checkpoint_target="$stage_root/checkpoints/$checkpoint_name"
mkdir -p "$checkpoint_target"
if [[ ! -f "$stage_root/checkpoint-manifest.json" ]]; then
    if find "$checkpoint_target" -mindepth 1 -print -quit | grep -q .; then
        echo "Incomplete checkpoint artifact exists: $checkpoint_target" >&2
        exit 2
    fi
    cp -a "$checkpoint_source/." "$checkpoint_target/"
fi
"$python_bin" scripts/setup/hash_artifact.py \
    --path "$checkpoint_target" \
    --output "$stage_root/checkpoint-manifest.json" >/dev/null

environment_archive="$stage_root/runtime/dt-environment.tar.zst"
if [[ ! -f "$environment_archive" ]]; then
    legacy_archive="$stage_root/runtime/dt-environment.tar"
    if [[ -f "$legacy_archive" ]]; then
        /opt/conda/bin/zstd -T0 -3 "$legacy_archive" -o "$environment_archive"
        rm -f "$legacy_archive"
    else
        tar -C "$(dirname "$environment_root")" -cf - \
            "$(basename "$environment_root")" \
            | /opt/conda/bin/zstd -T0 -3 -o "$environment_archive"
    fi
fi

for checkout in transformers spikingjelly; do
    source_checkout="$repo_root/src/$checkout"
    target_checkout="$stage_root/source-checkouts/$checkout"
    if [[ ! -d "$target_checkout" ]]; then
        mkdir -p "$target_checkout"
        (
            cd "$source_checkout"
            tar --exclude=.git -cf - .
        ) | (
            cd "$target_checkout"
            tar -xf -
        )
    fi
done

(
    cd "$stage_root"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

stage_bytes="$(du -sb "$stage_root" | awk '{print $1}')"
if (( stage_bytes > 30 * 1024 * 1024 * 1024 )); then
    echo "Staged assets exceed the 30 GiB transfer budget: $stage_bytes bytes" >&2
    exit 2
fi

if [[ "${THETA_SKIP_TRANSFER:-0}" == "1" ]]; then
    echo "$stage_root"
    exit 0
fi

ssh "$remote_host" "mkdir -p '$remote_assets'"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial --info=progress2 "$stage_root/" "$remote_host:$remote_assets/"
else
    scp -rp \
        "$stage_root/SHA256SUMS" \
        "$stage_root/checkpoint-manifest.json" \
        "$stage_root/checkpoints" \
        "$stage_root/datasets" \
        "$stage_root/runtime" \
        "$stage_root/source-checkouts" \
        "$remote_host:$remote_assets/"
fi
ssh "$remote_host" "cd '$remote_assets' && sha256sum --check SHA256SUMS"
echo "$remote_host:$remote_assets"
