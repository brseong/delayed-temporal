#!/bin/bash
trap 'kill -- -$$' SIGINT SIGTERM

indices=(0)
cuda_devices=(3)
device="cuda"
dataset="expsub" # "l2_square" or "expsub"
time_steps=40
hex="ab658c34213811f197a80242ac11000e"
source ./venv/bin/activate

for index in "${indices[@]}"; do
    echo "Running Jeffress train on GPU ${cuda_devices[$index]} with model ${hex}"
    script="CUDA_VISIBLE_DEVICES=${cuda_devices[$index]} python3 jeffress.py --lr 3e-2 --device ${device} --hex ${hex} --time-steps ${time_steps} ${dataset}"
    echo $script
    eval $script
done

wait