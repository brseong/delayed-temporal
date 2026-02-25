#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(0)
expr_name="ANSA\\ [-8,8]"
cuda_devices=(5)
l2_hex=(520d27e2126911f1a4530242ac11000e)
source ./venv/bin/activate

for index in "${indices[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]} with model ${l2_hex[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py --experiment_name ${expr_name} --device cuda --model_hex ${l2_hex[$index]}"
    echo $script
    eval $script
done

wait