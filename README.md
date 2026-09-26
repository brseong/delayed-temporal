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

The canonical entry point maps every active manuscript result to its experiment owner, verifies the completed evidence, rebuilds all publication figures, and compiles the paper. Run it from the repository root:

```bash
python3 scripts/experiments/reproduce_iclr_2027.py inventory
python3 scripts/experiments/reproduce_iclr_2027.py check
python3 scripts/experiments/reproduce_iclr_2027.py all
```

`check` does not modify files. `all` runs the focused verifiers, regenerates figures below `artifacts/reproduction/iclr_2027/`, promotes the verified PDF and PNG pairs to the paper tree, and writes the compiled manuscript below the same reproduction root. Use `--artifacts-root`, `--paper-root`, or `--python-bin` when the defaults do not apply.

The integrated path rebuilds the manuscript from completed, authenticated experiment evidence. Raw experimental reruns remain with their campaign controllers because the recorded results span different source revisions and execution environments. The `inventory` output is the authoritative mapping from manuscript results to those controllers.

### Scope

| Manuscript item | Experimental population | Maintained owner | Publication output |
|---|---|---|---|
| Table 3, ViT conversion | CIFAR-10 test 10,000; fixed ImageNet-1k validation 5,000 | `scripts/experiments/run_poseidon_local_range_paper_campaign.py` | `table_results.csv` plus the generated manuscript values |
| Table 4, language-model conversion | SST-2 validation 872; WikiText-2 test 2,891 nonempty texts | same campaign | `table_results.csv` plus the generated manuscript values |
| Table 3, ViT operation cost | checkpoint configurations for ViT-S/16, ViT-B/16, and ViT-L/16 | `scripts/analysis/vit_comparison_costs.py` | SOP totals checked against the manuscript |
| Discrete-time simulation figure | fixed first 500 ImageNet-1k validation images | verified historical sweep artifacts and `scripts/analysis/plot_clock_discretization.py` | `ViT-clock-discretization.pdf` and `.png` |
| Main timing-noise model comparison | CCT-7, ViT-S/16, ViT-B/16, RoBERTa-B, and GPT-2; three seeds | screening median model and text campaign controllers | `model-timing-noise-current.pdf` and `.png` |
| Main ViT-B encoder block timing-noise panel | fixed ImageNet-1k validation 5k; four scales, five block depths, and three seeds | `scripts/experiments/run_vit_bss2_depth_campaign.py` | `vit-b-depth-timing-noise.pdf` and `.png` |
| Appendix simulated timing-noise figure | fixed first 5,000 ImageNet-1k validation images; 21 cells and three seeds | the Poseidon campaign and `scripts/analysis/summarize_local_range_paper_campaign.py` | `ViT-noise-eval.pdf` and `.png` |
| Framework schematic | no experiment | `paper/iclr_2027/figures/potential-time-framework.tex` | `potential-time-framework.pdf` and `.png` |

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

The current paper figure is backed by the verified complete conversion sweeps in double precision on the first 500 ImageNet-1k validation images. Both sweeps retain the same frozen calibration and disable timing noise. To reproduce the published plot, verify and render the completed artifacts:

```bash
"$PYTHON_BIN" scripts/verification/verify_clock_driven.py
"$PYTHON_BIN" scripts/analysis/plot_clock_discretization.py \
  --time-step-root \
    artifacts/logs/clock_driven/vit_base_clock_driven_imagenet500_full_conversion_float64_fine_v3 \
  --window-root \
    artifacts/logs/clock_driven/vit_base_clock_driven_window_steps_384_768_imagenet500_full_conversion_float64_v3 \
  --output-prefix artifacts/figures/ViT-clock-discretization
```

An exact experimental rerun requires the source revision and contracts recorded in each artifact's `experiment.json`. The maintained controller can be inspected with:

```bash
"$PYTHON_BIN" scripts/experiments/run_clock_driven_vit.py --help
```

### Build and promote figures

The integrated driver rebuilds and promotes every active figure. For a manual promotion, copy only a verified PDF and its PNG preview from the evidence path recorded by `inventory`:

```bash
install -m 0644 artifacts/figures/ViT-clock-discretization.pdf \
  paper/iclr_2027/figures/ViT-clock-discretization.pdf
install -m 0644 artifacts/figures/ViT-clock-discretization.png \
  paper/iclr_2027/figures/ViT-clock-discretization.png
install -m 0644 artifacts/figures/appendix_vit_base_local_range_raw_timestamp_float64_v2/ViT-noise-eval.pdf \
  paper/iclr_2027/figures/ViT-noise-eval.pdf
install -m 0644 artifacts/figures/appendix_vit_base_local_range_raw_timestamp_float64_v2/ViT-noise-eval.png \
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

### BrainScaleS-2-derived timing-noise figures

The active main text panels use the authenticated screening median pair in `artifacts/brainscales2-primitives/20260924T_best_median_screen_summary.json`. The model comparison panel is reduced from the completed vision and text screening median campaigns; the ViT-B panel is reduced from the completed cumulative encoder block campaign. `reproduce_iclr_2027.py all` validates those inputs and regenerates both panels before compiling the manuscript.

These panels evaluate Gaussian timing noise parameterized by measured variation; they are not complete BrainScaleS-2 deployments. Preserve every BrainScaleS-2 or `bss2` artifact and never replace authenticated measurement input with simulated data.

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
