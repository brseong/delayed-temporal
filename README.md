# Biologically Plausible Dual Operators for TTFS-Coded Analog Spiking Transformers

This repository converts pretrained Transformer models to models implemented with TTFS operators and evaluates conversion accuracy, operation cost, and robustness. The main workflow evaluates existing checkpoints; it does not train them.

## Start here

The maintained evaluation interface is intentionally small:

- `scripts/evaluation/error_analysis_vit.py`
- `scripts/evaluation/error_analysis_bert.py`
- `scripts/evaluation/error_analysis_roberta.py`
- `scripts/evaluation/error_analysis_gpt2.py`

These programs own model and dataset loading, backend selection, calibration binding, metrics, and diagnostics. Experiment controllers call them as subprocesses; the evaluators do not depend on a particular campaign.

See [`scripts/README.md`](scripts/README.md) before adding an experiment. It distinguishes reusable evaluation code from campaign orchestration, result analysis, setup, and verification.

## Environment

Use Python 3.12 and install the pinned dependencies:

```bash
conda create -n dt python=3.12
conda activate dt
pip install -r requirements.txt
```

Pretrained checkpoints and datasets are external assets. Existing experiment manifests record their paths and hashes; do not silently substitute another checkpoint or dataset cache.

## Basic evaluation

Use the maintained Python campaign controllers listed in `scripts/experiments/README.md`. They fix their artifact layout and validate source and calibration identity before execution.

For a direct smoke evaluation, invoke an evaluator rather than copying a campaign controller:

```bash
CUDA_VISIBLE_DEVICES=4 python3 scripts/evaluation/error_analysis_vit.py \
  --experiment_name smoke --model_backend spiking \
  --model_id /data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k \
  --dataset_id imagenet-1k --batch_size 32 \
  --spiking-layernorm --spiking-mlp --spiking-attention \
  --max_eval_batches 5
```

Every temporal operator uses its declared analytic or frozen calibrated
potential range. There is no shared range setting for the complete model.

## Verification

Run focused checks for the code you changed. The layout boundary itself is checked with:

```bash
python3 scripts/verification/verify_script_layout.py
```

After operator or operation-count changes, also run:

```bash
python3 scripts/verification/verify_sop.py
```

Generated logs, tables, and figures belong under `artifacts/`; publication-ready copies belong under the venue-specific `paper/` tree.
