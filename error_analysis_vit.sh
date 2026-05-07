#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices="0,1,2,3"
cuda_count=4
source ./venv/bin/activate
device="cuda"
batch_sizes=($((16 * ${cuda_count})) $((32 * ${cuda_count})) $((8 * ${cuda_count})))
model_backend="spiking"

# Task and Model selection
dataset_id="imagenet-1k" # "cifar10" or "imagenet-1k"
# model_id="google/vit-base-patch16-224"
# model_id="WinKawaks/vit-small-patch16-224"
# model_id="google/vit-large-patch16-224"
model_ids=(
    # "google/vit-base-patch16-224"
    # "WinKawaks/vit-small-patch16-224"
    # "google/vit-large-patch16-224"
    "/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
    "/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"
    "/data/nas/vit_large_patch16_224.augreg_in21k_ft_in1k"
)

# Stage flags per experiment (mul=off; isolating log and expdiff):
# GPU 0: standard only (baseline for LN stages)
# GPU 1: log only
# GPU 2: log + expdiff (full SNN LN without mul)
ln_flags=(
    # "--no-spiking-ln-mul --no-spiking-ln-log --no-spiking-ln-expdiff"
    # "--no-spiking-ln-mul --spiking-ln-log --no-spiking-ln-expdiff"
    # "--no-spiking-ln-mul --spiking-ln-log --spiking-ln-expdiff"
    ""
    ""
    ""
)
flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    # "--no-spiking-layernorm --no-spiking-mlp --spiking-attention --model_backend ${model_backend}"
    # "--no-spiking-layernorm --spiking-mlp --no-spiking-attention --model_backend ${model_backend}"
    # "--spiking-layernorm --no-spiking-mlp --no-spiking-attention --model_backend ${model_backend}"
    # "--no-spiking-layernorm --no-spiking-mlp --no-spiking-attention --model_backend ${model_backend}"
)
expr_names=(
    ""
    ""
    ""
    # "full-snn"
    # "full-snn"
    # "full-snn"
    # "attn-only"
    # "mlp-only"
    # "ln-only"
)

for index in "${!expr_names[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=$cuda_devices python3 error_analysis_vit.py \
        --experiment_name ${model_backend}-${expr_names[$index]} --device ${device} \
        --batch_size ${batch_sizes[$index]} \
        --model_id ${model_ids[$index]} --dataset_id ${dataset_id} \
        ${flags[$index]} ${ln_flags[$index]} --theta 2000"
    echo $script
    eval $script
done

wait
