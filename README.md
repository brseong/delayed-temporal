# Biologically Plausible Dual Operators for TTFS-Coded Analog Spiking Transformers

This repository contains the evaluation code for converting pretrained Transformer models into compositions of time-to-first-spike (TTFS) operators. It measures deterministic conversion error, task accuracy, operation count, and sensitivity to timing noise. The workflow evaluates pretrained checkpoints; it does not train models.

This anonymous release contains source code only. Model checkpoints, datasets, completed experiment artifacts, and the ICLR manuscript are distributed separately.

## Repository structure

- `utils/transforms/`: bounded values, TTFS encoding and decoding, temporal primitives, composed operators, calibration, and noise injection.
- `utils/transformers/`: spiking layers and adapters for ViT, CCT, BERT, RoBERTa, GPT-2, and Llama.
- `scripts/evaluation/`: direct model-family evaluation entry points.
- `scripts/experiments/`: reproducible campaign controllers and resume logic.
- `scripts/analysis/`: result aggregation and figure generation.
- `scripts/verification/`: focused checks for operators, model integration, experiment contracts, and operation counts.

See [`scripts/README.md`](scripts/README.md) for the dependency boundaries and [`scripts/experiments/README.md`](scripts/experiments/README.md) for the maintained campaign map.

## Installation

Python 3.12 is required. Create an isolated environment and install the dependencies from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Evaluation requires the dependencies listed in `requirements.txt`. CUDA runs additionally require a compatible NVIDIA driver and enough GPU memory for the selected model and batch size.

Checkpoints and datasets are external inputs. For a reproducible run, record their exact identities and reuse the same preprocessing, calibration data, and evaluation split.

## Quick evaluation

The evaluation programs support dense Hugging Face models through `--model_backend hf` and converted models through `--model_backend spiking`. The following command runs a short ViT evaluation with placeholder asset paths:

```bash
export MODEL_ID=/path/to/vit-checkpoint
export EVAL_DATASET=/path/to/saved-evaluation-dataset

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/error_analysis_vit.py \
  --experiment_name smoke \
  --model_backend spiking \
  --model_id "$MODEL_ID" \
  --dataset_id imagenet-1k \
  --evaluation-dataset-path "$EVAL_DATASET" \
  --evaluation-split validation \
  --batch_size 8 \
  --max_eval_batches 5 \
  --device cuda \
  --spiking-layernorm \
  --spiking-attention \
  --spiking-mlp
```

Use `--model_backend hf` with the same checkpoint, dataset, and numerical settings for the dense reference. Run `python scripts/evaluation/error_analysis_vit.py --help` for calibration, precision, timing-noise, and ablation options.

Additional entry points are available for CCT, BERT, RoBERTa, GPT-2, and Llama under `scripts/evaluation/`.

## Calibration and experiment contracts

Converted evaluations use analytic ranges or a frozen local calibration table for each temporal operator. Values outside a declared range are clamped to that range. Calibration and evaluation data must remain separate, and the same frozen calibration must be used when comparing clean and noisy conditions.

The campaign controllers under `scripts/experiments/` bind source, checkpoint, dataset, preprocessing, calibration, and condition identities before reusing a completed result. Generated logs, tables, and figures belong under `artifacts/`, which is intentionally excluded from this source snapshot.

## Reproducing the ICLR 2027 results

The top-level reproduction driver inventories the result owners, validates completed evidence, regenerates figures, and builds the manuscript:

```bash
python scripts/experiments/reproduce_iclr_2027.py inventory \
  --artifacts-root /path/to/artifacts \
  --paper-root /path/to/iclr_2027

python scripts/experiments/reproduce_iclr_2027.py check \
  --artifacts-root /path/to/artifacts \
  --paper-root /path/to/iclr_2027

python scripts/experiments/reproduce_iclr_2027.py all \
  --artifacts-root /path/to/artifacts \
  --paper-root /path/to/iclr_2027 \
  --python-bin "$(command -v python)"
```

These commands require the separately supplied artifact and manuscript trees. The driver validates their recorded identities and fails when required evidence is missing or inconsistent. Raw experimental reruns remain under their individual campaign controllers because the reported results span several models and execution environments.

The principal campaign owners are:

| Result family | Controller | Result processing |
|---|---|---|
| Vision and language conversion | `scripts/experiments/run_poseidon_local_range_paper_campaign.py` | `scripts/analysis/summarize_local_range_paper_campaign.py` |
| Discrete-time simulation | `scripts/experiments/run_clock_driven_vit.py` | `scripts/analysis/plot_clock_discretization.py` |
| Cross-model timing-noise evaluation | `scripts/experiments/run_poseidon_screening_median_model_sweep.py` and `run_screening_median_text_sweep.py` | `scripts/analysis/summarize_screening_median_text_sweep.py` |
| ViT-B encoder-block timing-noise evaluation | `scripts/experiments/run_vit_bss2_depth_campaign.py` | `scripts/analysis/summarize_vit_bss2_depth.py` |
| Appendix ViT-B timing-noise evaluation | `scripts/experiments/run_appendix_vit_noise_campaign.py` | `scripts/analysis/summarize_local_range_paper_campaign.py --noise-only` |
| Llama appendix diagnostic | `scripts/experiments/run_llama_appendix_noise.py` | validated JSON summary and TeX table |

Each controller documents its required assets and arguments through `--help`. Use a new artifact directory for a new scientific rerun; do not combine outputs produced from different source revisions under one campaign identity.

## Verification

Run focused checks for the component you change. Common checks include:

```bash
python scripts/verification/verify_sop.py
python scripts/verification/verify_no_global_theta.py
python scripts/verification/verify_gaussian_time_noise.py
python scripts/verification/verify_calibration.py
```

`verify_sop.py` checks the arithmetic of the stated operation-count model. It does not validate circuit energy or physical runtime. Some model and campaign checks require external checkpoints, datasets, or authenticated experiment artifacts.

## Output policy

Keep generated datasets, logs, calibration tables, summaries, and figures under `artifacts/`. Do not commit credentials, local machine paths, or downloaded checkpoints. Publication-ready files should be produced from verified evidence rather than copied from partial logs.
