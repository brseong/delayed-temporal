#!/bin/bash
# Comprehensive ablation: isolates LN stages and attention independently.
# GPU layout:
#   0  control           – pure eager attention + nn.LayerNorm (true baseline)
#   1  sln_baseline      – SpikingLN all-off  → nn.functional.layer_norm shortcut
#   2  sln_log           – log stage only (no mul, no expdiff)
#   3  sln_log+expdiff   – log + expdiff (no mul)  — expected target
#   4  attn_only         – spiking attention, standard LN
#   5  sln+attn          – spiking LN (log+expdiff) + spiking attention

trap 'kill -- -$$' SIGINT SIGTERM
source ./venv/bin/activate

device="cuda"
mkdir -p ablation_logs

declare -A cfg
# GPU 0: true baseline
cfg[0]="--experiment_name control          --no-spiking-layernorm --no-spiking-attention"
# GPU 1: SpikingLN all-off → shortcut to nn.functional.layer_norm
cfg[1]="--experiment_name sln_baseline     --spiking-layernorm    --no-spiking-attention --no-spiking-ln-mul --no-spiking-ln-log --no-spiking-ln-expdiff"
# GPU 2: log stage only
cfg[2]="--experiment_name sln_log          --spiking-layernorm    --no-spiking-attention --no-spiking-ln-mul --spiking-ln-log    --no-spiking-ln-expdiff"
# GPU 3: log + expdiff (no mul) — primary SNN LN candidate
cfg[3]="--experiment_name sln_log+expdiff  --spiking-layernorm    --no-spiking-attention --no-spiking-ln-mul --spiking-ln-log    --spiking-ln-expdiff"
# GPU 4: spiking attention only, standard LN
cfg[4]="--experiment_name attn_only        --no-spiking-layernorm --spiking-attention"
# GPU 5: full pipeline (log+expdiff LN + spiking attention)
cfg[5]="--experiment_name sln+attn         --spiking-layernorm    --spiking-attention    --no-spiking-ln-mul --spiking-ln-log    --spiking-ln-expdiff"

for gpu in 0 1 2 3 4 5; do
    echo "Launching GPU ${gpu}: ${cfg[$gpu]}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 error_analysis_vit.py \
        --device ${device} ${cfg[$gpu]} \
        > ablation_logs/gpu${gpu}.log 2>&1 &
done

wait
echo "All done."
