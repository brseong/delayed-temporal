#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

cuda_devices="0,1,2,3,4,5,6,7"
source ./venv/bin/activate
device="cuda"
model_id="Bhumika/roberta-base-finetuned-sst2"
backend="spiking"
batch_size=$((16 * 8))
dataset_name="glue"
dataset_config_name="sst2"
dataset_split="validation"
task="sst2"

flags=(
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"
    "--spiking-layernorm --spiking-mlp --spiking-attention --model_backend ${backend}"
)
expr_names=(
    "std_0"
    "std_1e-5"
    "std_2e-5"
    "std_3e-5"
    "std_4e-5"
)

noise_stds=(0 1e-5 2e-5 3e-5 4e-5)
thetas=(2000 1000 500 250)

for theta in "${thetas[@]}"; do
    for index in "${!noise_stds[@]}"; do
        noise_std=${noise_stds[$index]}
        echo "Running error analysis: ${expr_names[$index]}"
        script="CUDA_VISIBLE_DEVICES=${cuda_devices} python3 error_analysis_roberta.py \
            --experiment_name roberta_${expr_names[$index]}_${task} --device ${device} \
            --task ${task} --batch_size ${batch_size} --spike_time_noise_std ${noise_std}\
            --model_id ${model_id} \
            --dataset_name ${dataset_name} --dataset_split ${dataset_split} \
            ${flags[$index]} --theta $theta"
        if [[ -n "${dataset_config_name}" ]]; then
            script+=" --dataset_config_name ${dataset_config_name}"
        fi
        echo $script
        eval $script
    done
done

wait