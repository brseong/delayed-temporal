#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices="0,1,2,3,4,5,6,7"
source ./venv/bin/activate
device="cuda"
# model_id="WinKawaks/vit-small-patch16-224"
model_id="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"
dataset_id="imagenet-1k"
backend="spiking"
batch_size=$((32 * 8))

expr_names=(
    "std_0"
    "std_1e-5"
    "std_2e-5"
    "std_3e-5"
    "std_4e-5"
)

noise_stds=(0 1e-5 2e-5 3e-5 4e-5)
thetas=(1000 500 200 100 50 20 10 5 2 1)

for theta in "${thetas[@]}"; do
    for index in "${!noise_stds[@]}"; do
        noise_std=${noise_stds[$index]}
        echo "Running error analysis on GPU $cuda_devices: ${expr_names[$index]} with noise std ${noise_std} and theta ${theta}"
        script="CUDA_VISIBLE_DEVICES=${cuda_devices} python3 error_analysis_vit.py \
            --experiment_name std_${noise_stds[$index]}_theta_${theta} --device ${device}\
            --model_id ${model_id} --dataset_id ${dataset_id} \
            --batch_size ${batch_size} ${flags[$index]} \
            --spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend} \
            --noise-std ${noise_std} --theta ${theta} --quick-test"
        echo $script
        eval $script
    done
done

wait