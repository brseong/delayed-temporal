# Biologically Plausible Dual Operators for TTFS-Coded Analog Spiking Transformers

This repository is the official implementation of *Biologically Plausible Dual Operators for TTFS-Coded
Analog Spiking Transformers*.

## Requirements
To set up the environment for reproducing the results of this paper, please follow the instructions below:
```bash
conda create -n myenv python=3.12
conda activate myenv
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Before running the error analysis script, download the pre-trained ViT models:
```bash
bash scripts/setup/convert_vits.sh
```
Make sure to place the downloaded models in the appropriate directory as specified in `scripts/experiments/error_analysis_vit.sh`.

## Evaluation (ViT)

To evaluate the model on ImageNet, run:

```bash
bash scripts/experiments/error_analysis_vit.sh
```

## Threshold-Jitter Analysis (ViT)

To perform threshold-jitter analysis on the ViT model, run:

```bash
bash scripts/experiments/theta_jitter_analysis_vit.sh
```

## BrainScaleS-2 toy ANN2SNN robustness

The attention-free hardware-in-the-loop path first trains and freezes a
`4-30-3` Yin-Yang ANN, then converts it into the Hagen UInt5/int6/Int8 contract.
Local validation requires no EBRAINS packages:

```bash
python scripts/evaluation/brainscales2_toy_hil.py \
  --phase train --task yinyang --architecture yy-30 \
  --output-dir artifacts/brainscales2-toy/checkpoint
python scripts/evaluation/brainscales2_toy_hil.py \
  --phase convert --task yinyang --architecture yy-30 \
  --checkpoint artifacts/brainscales2-toy/checkpoint/checkpoint.pt \
  --output-dir artifacts/brainscales2-toy/checkpoint
python scripts/evaluation/brainscales2_toy_hil.py \
  --phase local-eval --task yinyang --architecture yy-30 \
  --checkpoint artifacts/brainscales2-toy/checkpoint/checkpoint.pt \
  --converted-checkpoint artifacts/brainscales2-toy/checkpoint/converted_checkpoint.pt \
  --pwm-backend torch --pool-backend replay \
  --output-dir artifacts/brainscales2-toy/replay
```

Hardware execution uses the thin
`scripts/notebooks/ebrains_brainscales2_toy_hil.ipynb` launcher in the
`EBRAINS-experimental` kernel. Formal runs require separate explicit Hagen and
spiking calibration files; `hxtorch` remains a lazy EBRAINS-only dependency.
