#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-1 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
# model_id="WinKawaks/vit-small-patch16-224"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32

flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 2e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 4e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 5e-5 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-4 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 3e-4 --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --noise-std 1e-3 --model_backend ${backend}"
)
expr_names=(
    "std_1e-5"
    "std_2e-5"
    "std_3e-5"
    "std_4e-5"
    "std_5e-5"
)

gpu_pool_init "${cuda_devices[@]}"
for index in "${!expr_names[@]}"; do
    gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
    echo "Running error analysis on GPU ${gpu}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device}\
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta 400.0"
    echo $script
    eval $script &
    gpu_pool_register $! "$gpu"
done

wait
