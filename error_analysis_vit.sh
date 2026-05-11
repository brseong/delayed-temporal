#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices="0,1,2,3"
cuda_count=4
device="cuda"
batch_sizes=($((16 * ${cuda_count})) $((32 * ${cuda_count})) $((8 * ${cuda_count})))
model_backend="spiking"

# Task and Model selection
dataset_id="imagenet-1k" # "imagenet-1k"
model_ids=(
    "/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
    "/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"
    "/data/nas/vit_large_patch16_224.augreg_in21k_ft_in1k"
)

flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${model_backend} --precision float32"
)
expr_names=(
    ""
    ""
    ""
)

for index in "${!expr_names[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=$cuda_devices python3 error_analysis_vit.py \
        --experiment_name ${model_backend}-${expr_names[$index]} --device ${device} \
        --batch_size ${batch_sizes[$index]} \
        --model_id ${model_ids[$index]} --dataset_id ${dataset_id} \
        ${flags[$index]} --theta 2000"
    echo $script
    eval $script
done

wait
