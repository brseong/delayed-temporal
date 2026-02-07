#!/bin/bash

cuda_devices=(0 1 2 3)
l2_hex=(6d579cd8045611f1b9910242ac11000f 8da8c372045611f195670242ac11000f 2a9cd64c045611f18af80242ac11000f c1d8c106045611f18f2e0242ac11000f)
source ./venv/bin/activate

for device in "${cuda_devices[@]}"; do
    echo "Running error analysis on GPU $device"
    script="CUDA_VISIBLE_DEVICES=${device} python3 error_analysis_vit.py --device cuda --model_hex ${l2_hex[$device]} &"
    echo $script
    eval $script
done