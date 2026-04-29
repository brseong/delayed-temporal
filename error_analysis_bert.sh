#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

source ./venv/bin/activate
device="cuda"
theta=400
model_backend="spiking"
task="${1:-sst2}"  # sst2 | agnews | imdb

case "${task}" in
    sst2)
        model_id="textattack/bert-base-uncased-SST-2"
        dataset_name="glue"
        dataset_config_name="sst2"
        dataset_split="validation"
        ;;
    agnews)
        model_id="textattack/bert-base-uncased-ag-news"
        dataset_name="ag_news"
        dataset_config_name=""
        dataset_split="test"
        ;;
    imdb)
        model_id="textattack/bert-base-uncased-imdb"
        dataset_name="imdb"
        dataset_config_name=""
        dataset_split="test"
        ;;
    *)
        echo "Unsupported task '${task}'. Use one of: sst2, agnews, imdb"
        exit 1
        ;;
esac

expr_names=(
    # "spiking_attn"
    # "sln"
    # "smlp"
    "all"
    # "control"
)

# Ablation flags
flags=(
    # "--spiking-attention --no-spiking-layernorm --no-spiking-mlp" # spiking_attn
    # "--no-spiking-attention --spiking-layernorm --no-spiking-mlp" # sln
    # "--no-spiking-attention --no-spiking-layernorm --spiking-mlp" # smlp
    "--spiking-attention --spiking-layernorm --spiking-mlp"       # all
    # "--no-spiking-attention --no-spiking-layernorm --no-spiking-mlp --activation gelu" # control (ANN only)
)

cuda_devices=(0) # Adjust if you want to run on different GPUs

for index in "${!expr_names[@]}"; do
    echo "Running error analysis: ${expr_names[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_bert.py \
        --experiment_name ${expr_names[$index]}_${task} --device ${device} \
        --model_backend ${model_backend} \
        --task ${task} \
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
