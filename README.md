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
