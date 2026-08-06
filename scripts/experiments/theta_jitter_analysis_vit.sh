#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

cuda_devices=(${GPUS:-0 1 2 3 4 5 6 7})   # override with e.g. GPUS="4 5 6 7"
source ./venv/bin/activate 2>/dev/null
source "$repo_root/scripts/lib/gpu_pool.sh"
device="cuda"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=32   # per-GPU (was 32*8 for 8-way DataParallel; now one job per GPU via the pool)

expr_names=(
    "std_0"
    "std_1e-5"
    "std_2e-5"
    "std_3e-5"
    "std_4e-5"
)

noise_stds=(0 1e-5 2e-5 3e-5 4e-5)
thetas=(1000 500 200 100 50 20 10 5 2 1)

gpu_pool_init "${cuda_devices[@]}"
for theta in "${thetas[@]}"; do
    for index in "${!noise_stds[@]}"; do
        noise_std=${noise_stds[$index]}
        gpu_pool_acquire; gpu=$GPU_POOL_ACQUIRED
        echo "Running error analysis on GPU ${gpu}: ${expr_names[$index]} with noise std ${noise_std} and theta ${theta}"
        script="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/evaluation/error_analysis_vit.py \
            --experiment_name std_${noise_stds[$index]}_theta_${theta} --device ${device}\
            --model_id ${model_id} --dataset_id ${dataset_id} \
            --batch_size ${batch_size} ${flags[$index]} \
            --spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend} \
            --noise-std ${noise_std} --theta ${theta} --quick-test"
        echo $script
        eval $script &
        gpu_pool_register $! "$gpu"
    done
done

wait
