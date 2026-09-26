#!/bin/bash
#SBATCH --job-name=llama-clean
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00

set -euo pipefail

: "${LLAMA_WORKROOT:?}"
: "${LLAMA_MODEL_PATH:?}"
: "${LLAMA_PYTHON_BIN:?}"
: "${LLAMA_EXTRA_SITE_PACKAGES:?}"
: "${LLAMA_CALIBRATION_PATH:?}"
: "${LLAMA_LOGICAL_MODEL_ID:?}"
: "${LLAMA_OUTPUT_ROOT:?}"
: "${LLAMA_SHARD_COUNT:?}"

shard_index="${LLAMA_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
max_calibration_examples="${LLAMA_MAX_CALIBRATION_EXAMPLES:-5000}"
max_eval_examples="${LLAMA_MAX_EVAL_EXAMPLES:-2891}"
microbatch_size="${LLAMA_MICROBATCH_SIZE:-1}"
output_dir="${LLAMA_OUTPUT_ROOT}/shards/shard_${shard_index}"
mkdir -p "$output_dir"

export PYTHONPATH="${LLAMA_EXTRA_SITE_PACKAGES}:${LLAMA_WORKROOT}/repo/src/transformers/src:${LLAMA_WORKROOT}/repo/src/spikingjelly:${LLAMA_WORKROOT}/repo"
python_root="$(dirname "$(dirname "$LLAMA_PYTHON_BIN")")"
export LD_LIBRARY_PATH="${python_root}/lib:${LD_LIBRARY_PATH:-}"
for library_dir in "${python_root}"/lib/python3.12/site-packages/nvidia/*/lib "${LLAMA_EXTRA_SITE_PACKAGES}"/nvidia/*/lib; do
  if [[ -d "$library_dir" ]]; then
    export LD_LIBRARY_PATH="${library_dir}:${LD_LIBRARY_PATH}"
  fi
done
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${LLAMA_WORKROOT}/repo"
srun --ntasks=1 --kill-on-bad-exit=1 \
  "$LLAMA_PYTHON_BIN" -u scripts/evaluation/error_analysis_llama.py \
  --model-id "$LLAMA_MODEL_PATH" \
  --calibration-path "$LLAMA_CALIBRATION_PATH" \
  --calibration-model-id "$LLAMA_LOGICAL_MODEL_ID" \
  --output-dir "$output_dir" \
  --mode clean --evaluation-dataset wikitext2 \
  --max-calibration-examples "$max_calibration_examples" \
  --max-eval-examples "$max_eval_examples" \
  --batch-size 8 --microbatch-size "$microbatch_size" --max-length 128 \
  --eval-shard-index "$shard_index" --eval-shard-count "$LLAMA_SHARD_COUNT" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
