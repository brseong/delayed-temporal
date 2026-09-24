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

## Reproducing the ICLR 2027 experiments and figures

This runbook covers every executable experiment used by the current ICLR manuscript except the pending BrainScaleS-2 measurements. It distinguishes rerunning an experiment from rendering a figure from verified artifacts. Run commands from the repository root unless a command changes directory explicitly.

### Scope

| Manuscript item | Experimental population | Maintained owner | Publication output |
|---|---|---|---|
| Table 3, ViT conversion | CIFAR-10 test 10,000; fixed ImageNet-1k validation 5,000 | `scripts/experiments/run_poseidon_local_range_paper_campaign.py` | `table_results.csv` plus the generated manuscript values |
| Table 4, language-model conversion | SST-2 validation 872; WikiText-2 test 2,891 nonempty texts | same campaign | `table_results.csv` plus the generated manuscript values |
| Table 3, ViT operation cost | checkpoint configurations for ViT-S/16, ViT-B/16, and ViT-L/16 | `scripts/analysis/vit_comparison_costs.py` | SOP totals checked against the manuscript |
| Discrete-time simulation figure | fixed first 500 ImageNet-1k validation images | verified historical sweep artifacts and `scripts/analysis/plot_clock_discretization.py` | `ViT-clock-discretization.pdf` and `.png` |
| Appendix simulated timing-noise figure | fixed first 5,000 ImageNet-1k validation images; 21 cells and three seeds | the Poseidon campaign and `scripts/analysis/summarize_local_range_paper_campaign.py` | `ViT-noise-eval.pdf` and `.png` |
| Framework schematic | no experiment | `paper/iclr_2027/figures/potential-time-framework.tex` | `potential-time-framework.pdf` and `.png` |
| BrainScaleS-2 primitive-level error figure | pending measurements | not included in this runbook | no result may be substituted for the placeholder |

The conversion campaign uses double precision, continuous spike times, frozen layer-wise calibration, W&B disabled, and TensorBoard disabled. Each model collects its own calibration from 5,000 training examples using two passes, 2,048 histogram bins, observed extrema, and 5% interval-width expansion on each side. Validation and test labels do not enter calibration.

### Freeze the source and select artifact storage

Use the recorded paper source when reproducing the current numerical results. A new scientific rerun must instead use a reviewed clean commit and a new artifact root; never mix commits in one campaign directory.

```bash
export PYTHON_BIN=/opt/conda/envs/dt/bin/python
export PAPER_SOURCE_COMMIT=b1a6bf8f7baa89250201c9af96d05b6154249de5
export SOURCE_ROOT=/data/delayed-temporal-worktrees/paper-reproduction-b1a6bf8
export RUN_ARTIFACTS=/data/delayed-temporal/artifacts

test -e "$SOURCE_ROOT/.git" || \
  git worktree add --detach "$SOURCE_ROOT" "$PAPER_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$PAPER_SOURCE_COMMIT"
```

The existing `artifacts/` tree contains the immutable self-contained datasets and the verified completed results. To force a clean rerun without overwriting them, choose an unused disk-backed root and expose only the immutable assets there:

```bash
export RUN_ARTIFACTS=/data/delayed-temporal-reruns/iclr2027-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ARTIFACTS"
ln -s /data/delayed-temporal/artifacts/assets "$RUN_ARTIFACTS/assets"
test "$(findmnt -n -o FSTYPE -T "$RUN_ARTIFACTS")" != tmpfs
test "$(findmnt -n -o FSTYPE -T "$RUN_ARTIFACTS")" != ramfs
```

The campaign also requires the three ImageNet checkpoints at:

- `/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k`
- `/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k`
- `/data/nas/vit_large_patch16_224.augreg_in21k_ft_in1k`

Its remaining checkpoints and datasets come from `artifacts/assets/vit-conversion-comparison-v1`, `artifacts/assets/conversion-comparison-text-v1`, and `artifacts/assets/theta-selection-v1`. The manifests and fixed dataset fingerprints are authoritative. Do not replace an unavailable asset with a freshly downloaded cache under the same tag.

### Run Tables 3 and 4 and the simulated timing-noise sweep

The maintained full controller runs only on `poseidon1`, owns GPUs 0--3 there, writes all runtime files below the selected artifact root, and rejects tmpfs and ramfs. It evaluates four ViTs, RoBERTa-B, RoBERTa-L, and GPT-2, then runs the 63 ViT-B timing-noise replicas after the matching ViT-B calibration is complete.

```bash
export DELAYED_TEMPORAL_ARTIFACTS_ROOT="$RUN_ARTIFACTS"

"$PYTHON_BIN" -u \
  "$SOURCE_ROOT/scripts/experiments/run_poseidon_local_range_paper_campaign.py" \
  --source-root "$SOURCE_ROOT" \
  --expected-commit "$PAPER_SOURCE_COMMIT" \
  --gpus 0 1 2 3 \
  --python-bin "$PYTHON_BIN"
```

The controller is resumable: rerun the identical command after a failure. It reuses only complete results whose manifests pass the experiment-specific identity checks. Inspect progress without modifying the campaign:

```bash
"$PYTHON_BIN" -m json.tool \
  "$RUN_ARTIFACTS/logs/paper_end_to_end_local_range_poseidon_v1/status.json"
tail -F \
  "$RUN_ARTIFACTS/logs/paper_end_to_end_local_range_poseidon_v1/controller-logs/"*.log
```

The simulated timing-noise grid uses ViT-B/16, batch size 32, the fixed ImageNet validation 5k, and the calibration frozen by the matching clean evaluation. For every local spike-time window $T$, $\sigma_t=r_tT$. It evaluates $r_t=10^{-5}10^{i/8}$ for $i=0,\ldots,8$ at deadline margin $4\sigma_t$, and ratios $k\in\{0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12\}$ at $r_t=10^{-5}$ with margin $k\sigma_t$. Each unique cell uses seeds 0, 1, and 2.

After all tasks finish, validate and aggregate the results:

```bash
"$PYTHON_BIN" scripts/analysis/summarize_local_range_paper_campaign.py \
  --artifacts-root "$RUN_ARTIFACTS"
"$PYTHON_BIN" scripts/verification/verify_local_range_paper_summary.py
```

The summary command writes:

- `$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/table_results.csv`
- `$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/noise_raw_runs.csv`
- `$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/noise_summary.csv`
- `$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/summary.json`
- `$RUN_ARTIFACTS/figures/ViT-noise-eval-end-to-end.pdf` and `.png`

Update the manuscript's ANN, SNN, and difference values only from `table_results.csv`; do not copy progress logs or partial results into the paper.

### Recompute the ViT SOP column

The SOP estimator consumes each actual checkpoint configuration. It models the declared circuit mapping rather than Python or CUDA instructions.

```bash
mkdir -p "$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/sop"

"$PYTHON_BIN" scripts/analysis/vit_comparison_costs.py \
  "$RUN_ARTIFACTS/assets/vit-conversion-comparison-v1/checkpoints/vit_small_patch16_224_cifar10/config.json" \
  > "$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/sop/cifar10_vit_small.json"
"$PYTHON_BIN" scripts/analysis/vit_comparison_costs.py \
  /data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k/config.json \
  > "$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/sop/imagenet_vit_small.json"
"$PYTHON_BIN" scripts/analysis/vit_comparison_costs.py \
  /data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k/config.json \
  > "$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/sop/imagenet_vit_base.json"
"$PYTHON_BIN" scripts/analysis/vit_comparison_costs.py \
  /data/nas/vit_large_patch16_224.augreg_in21k_ft_in1k/config.json \
  > "$RUN_ARTIFACTS/results/paper_end_to_end_local_range_poseidon_v1/sop/imagenet_vit_large.json"

"$PYTHON_BIN" scripts/verification/verify_vit_comparison_costs.py
"$PYTHON_BIN" scripts/verification/verify_sop.py
```

`verify_vit_comparison_costs.py` checks the current Table 3 accounting. `verify_sop.py` is an independent checker for the archived operation-count model and must not be treated as physical energy or runtime validation.

### Recreate the discrete-time simulation figure

The current paper figure is backed by verified historical artifacts with $\theta=20$, double precision, no timing noise, and the first 500 ImageNet validation images. The time-step-width sweep records source commit `ec058e5707c1550077ef55a1a95bca1d347a8849`; the fixed-steps-per-window extension records `2869ef440f618515fe227dd6c9173a5b9ecae232`.

Current `main` no longer has the global $\theta$ execution contract. Therefore, `scripts/experiments/run_clock_driven_vit.py` on current `main` is a new local-range experiment, not an exact rerun of the paper's $\theta=20$ figure. To reproduce the published plot, verify and render the recorded artifacts:

```bash
"$PYTHON_BIN" scripts/verification/verify_clock_driven.py
"$PYTHON_BIN" scripts/analysis/plot_clock_discretization.py \
  --time-step-root \
    artifacts/logs/clock_driven/vit_base_clock_driven_imagenet500_theta20_float64_fine_v1 \
  --window-root \
    artifacts/logs/clock_driven/vit_base_clock_driven_window_steps_384_768_imagenet500_theta20_float64_v1 \
  --output-prefix artifacts/figures/ViT-clock-discretization
```

An exact experimental rerun requires detached worktrees at the two source commits and the contracts recorded in each artifact's `experiment.json` or `verification.json`. Do not present a current-main local-range rerun as the same experiment. The maintained current-main controller can be inspected with:

```bash
"$PYTHON_BIN" scripts/experiments/run_clock_driven_vit.py --help
```

### Build and promote figures

Generated figures stay in `artifacts/figures/` until their evidence passes verification. Promote only the final PDF and its PNG preview:

```bash
install -m 0644 artifacts/figures/ViT-clock-discretization.pdf \
  paper/iclr_2027/figures/ViT-clock-discretization.pdf
install -m 0644 artifacts/figures/ViT-clock-discretization.png \
  paper/iclr_2027/figures/ViT-clock-discretization.png
install -m 0644 "$RUN_ARTIFACTS/figures/ViT-noise-eval-end-to-end.pdf" \
  paper/iclr_2027/figures/ViT-noise-eval.pdf
install -m 0644 "$RUN_ARTIFACTS/figures/ViT-noise-eval-end-to-end.png" \
  paper/iclr_2027/figures/ViT-noise-eval.png
```

The framework figure is a TikZ schematic, not an experimental result. Build it in an artifact directory, then promote it:

```bash
mkdir -p artifacts/runtime/potential-time-framework
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=artifacts/runtime/potential-time-framework \
  paper/iclr_2027/figures/potential-time-framework.tex
pdftoppm -png -r 300 -singlefile \
  artifacts/runtime/potential-time-framework/potential-time-framework.pdf \
  artifacts/runtime/potential-time-framework/potential-time-framework
install -m 0644 artifacts/runtime/potential-time-framework/potential-time-framework.pdf \
  paper/iclr_2027/figures/potential-time-framework.pdf
install -m 0644 artifacts/runtime/potential-time-framework/potential-time-framework.png \
  paper/iclr_2027/figures/potential-time-framework.png
```

### Final verification and manuscript build

Run the evidence checks before compiling the paper:

```bash
"$PYTHON_BIN" scripts/verification/verify_local_range_paper_summary.py
"$PYTHON_BIN" scripts/verification/verify_full_calibrated_text_comparison.py
"$PYTHON_BIN" scripts/verification/verify_vit_comparison_costs.py
"$PYTHON_BIN" scripts/verification/verify_clock_driven.py
"$PYTHON_BIN" scripts/verification/verify_gaussian_time_noise.py
"$PYTHON_BIN" scripts/verification/verify_no_global_theta.py
"$PYTHON_BIN" scripts/verification/verify_script_layout.py
lat check

mkdir -p artifacts/runtime/iclr2027
(
  cd paper/iclr_2027
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir=/data/delayed-temporal/artifacts/runtime/iclr2027 \
    iclr2027_conference.tex
)
test -s artifacts/runtime/iclr2027/iclr2027_conference.pdf
! rg -n 'undefined references|Citation .* undefined|Reference .* undefined' \
  artifacts/runtime/iclr2027/iclr2027_conference.log
```

Inspect the final PDF, captions, legends, axes, table values, and page layout manually. The paper tree is an independent Git checkout, so review and commit its changes separately from the code repository.

### Deferred BrainScaleS-2 experiment

The main-text primitive-level error figure remains pending until the BrainScaleS-2 measurements are available. EBRAINS login, hardware job submission, data download, and hardware-derived plotting are intentionally outside this runbook. Preserve every existing BrainScaleS-2 artifact; do not delete it, replace it with the simulated Gaussian timing-noise sweep, or populate the placeholder from unverified measurements.

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
