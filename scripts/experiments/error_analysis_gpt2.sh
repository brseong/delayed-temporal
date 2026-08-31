#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

source ./venv/bin/activate
device="cuda"
theta=2000
attention_theta=${ATTENTION_THETA:-100}
task="wikitext2" # wikitext2 | wikitext103

expr_names=(
    # "spiking_attn"
    # "sln"
    # "smlp"
    "spiking"
    # "baseline"
)

# Ablation flags
flags=(
    # "--no-spiking-ln-mul --no-spiking-ln-log --model_backend spiking" # spiking_attn
    # "--no-spiking-ln-mul --no-spiking-ln-expdiff --model_backend spiking" # spiking_attn
    # "--no-spiking-ln-log --no-spiking-ln-expdiff --model_backend spiking" # spiking_attn
    # "--no-spiking-attention --spiking-layernorm --no-spiking-mlp --model_backend spiking" # sln
    # "--no-spiking-attention --no-spiking-layernorm --spiking-mlp --model_backend spiking" # smlp
    "--spiking-attention --spiking-layernorm --spiking-mlp --model_backend spiking"       # all
    "--no-spiking-attention --no-spiking-layernorm --no-spiking-mlp --activation gelu --model_backend hf" # baseline (ANN only)
)

cuda_devices=(${GPUS:-0 1 2 3}) # override with e.g. GPUS="4 5 6 7"
source "$repo_root/scripts/lib/gpu_pool.sh"

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running error analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_gpt2.py \
        --experiment_name ${expr_names[$index]} --device ${device} \
        --batch_size 8 \
        ${flags[$index]} --theta ${theta} --attention-theta ${attention_theta} \
        --task ${task}"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
