#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
gpu="${GPU:-0}"
batch_size="${BATCH_SIZE:-16}"
cache_dir="${CACHE_DIR:-/root/.cache/huggingface/datasets}"
output_dir="${OUTPUT_DIR:-artifacts/precision_gpt2}"
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

common=(
    --device cuda
    --batch_size "$batch_size"
    --max_length 128
    --task wikitext2
    --cache-dir "$cache_dir"
    --calibration-mode none
)

is_complete() {
    local log_file="$1"
    [[ -f "$log_file" ]] \
        && ! grep -q "Traceback (most recent call last)" "$log_file" \
        && grep -q '^Average Loss:' "$log_file" \
        && grep -q '^Perplexity:' "$log_file"
}

run_condition() {
    local name="$1"
    shift
    local log_file="$output_dir/${name}.log"
    if is_complete "$log_file"; then
        echo "Skipping completed condition: $name"
        return
    fi
    echo "Running condition on GPU $gpu: $name"
    "$python_bin" scripts/evaluation/error_analysis_gpt2.py \
        --experiment_name "appendix-${name}" \
        "${common[@]}" "$@" >"$log_file" 2>&1
}

# Dense reference and the local GPT-2 adapter with all temporal paths disabled
# separate wrapper fidelity from conversion error.
run_condition hf_float32 \
    --model_backend hf --dtype float32 \
    --theta 2000 --attention-theta 2000
run_condition wrapper_float32 \
    --model_backend spiking --dtype float32 \
    --no-spiking-layernorm --no-spiking-attention --no-spiking-mlp \
    --theta 2000 --attention-theta 2000
# Every float32 point below has the same representability-capped softmin score rail at
# max_length=128. Only the attention timestamp window and its subtraction ULP vary.
for attention_theta in 50 100 200 500 1000 2000; do
    run_condition "float32_attn${attention_theta}" \
        --model_backend spiking --dtype float32 \
        --spiking-layernorm --spiking-attention --spiking-mlp \
        --theta 2000 --attention-theta "$attention_theta" \
        --report-clamp-stats
done

# This endpoint is a complete numerical-reference control: float64 both resolves
# temporal subtraction and raises the softmin exponent representability ceiling.
run_condition float64_attn2000 \
    --model_backend spiking --dtype float64 \
    --spiking-layernorm --spiking-attention --spiking-mlp \
    --theta 2000 --attention-theta 2000 \
    --report-clamp-stats

"$python_bin" scripts/analysis/summarize_gpt2_precision.py \
    --log-dir "$output_dir" \
    --csv "$output_dir/results.csv" \
    --markdown "$output_dir/results.md"
