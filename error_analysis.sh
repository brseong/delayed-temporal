#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(0 1 2 3)
expr_name="ASSDPA\\ [-7,7]"
cuda_devices=(4 5 6 7)
l2_hex=(adfce21604e211f18e1c0242ac11000f 7dedc37804e311f18e1b0242ac11000f e3ef1d5204e311f1b87b0242ac11000f f69a01ce04e311f1af950242ac11000f)
source ./venv/bin/activate

for index in "${indices[@]}"; do
    echo "Running error analysis on GPU ${cuda_devices[$index]} with model ${l2_hex[$index]}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 error_analysis_vit.py --experiment_name ${expr_name} --device cuda --model_hex ${l2_hex[$index]} &"
    echo $script
    eval $script
done

wait