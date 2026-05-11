#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices=(1 4 5 6 7)
source ./venv/bin/activate
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

for index in "${!expr_names[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]}: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py \
        --experiment_name ${expr_names[$index]} --device ${device}\
        --model_id ${model_id} --dataset_id ${dataset_id} \
        --batch_size ${batch_size} ${flags[$index]} --theta 400.0"
    echo $script
    eval $script &
done

wait