#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

# source ./venv/bin/activate
device="cuda"
theta=2000
task="${1:-sst2-b}"  # sst2-b | sst2-l | agnews | imdb

case "${task}" in
    sst2-b)
        model_id="Bhumika/roberta-base-finetuned-sst2"
        dataset_name="glue"
        dataset_config_name="sst2"
        dataset_split="validation"
        task="sst2"
        ;;
    sst2-l)
        model_id="philschmid/roberta-large-sst2"
        dataset_name="glue"
        dataset_config_name="sst2"
        dataset_split="validation"
        task="sst2"
        ;;
    agnews)
        model_id="textattack/roberta-base-ag-news"
        dataset_name="ag_news"
        dataset_config_name=""
        dataset_split="test"
        ;;
    imdb)
        model_id="textattack/roberta-base-imdb"
        dataset_name="imdb"
        dataset_config_name=""
        dataset_split="test"
        ;;
    *)
        echo "Unsupported task '${task}'. Use one of: sst2-b, sst2-l, agnews, imdb"
        exit 1
        ;;
esac

expr_names=(
    "all"
    "control"
)

# Ablation flags
flags=(
    "--spiking-attention --spiking-layernorm --spiking-mlp --model_backend spiking"       # all
    "--no-spiking-attention --no-spiking-layernorm --no-spiking-mlp --activation gelu --model_backend hf" # control (ANN only)
)

cuda_devices=(0 1) # Adjust if you want to run on different GPUs

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_roberta.py \
        --experiment_name roberta_${expr_names[$index]}_${task} --device ${device} \
        --task ${task} --batch_size 16 \
        --model_id ${model_id} \
        --dataset_name ${dataset_name} --dataset_split ${dataset_split} \
        ${flags[$index]} --theta ${theta}"
    if [[ -n "${dataset_config_name}" ]]; then
        script+=" --dataset_config_name ${dataset_config_name}"
    fi
    echo $script
    eval $script &
done

wait
