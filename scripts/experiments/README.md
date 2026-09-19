# Experiment campaigns

This directory orchestrates reproducible runs around the stable `scripts/evaluation/error_analysis_*` entry points.

## Maintained Python campaigns

The multi-file Python campaigns use a contract module, one or more workers or controllers, and a matching summarizer or verification program.

| Campaign | Contract and runners | Result processing |
|---|---|---|
| Calibrated ViT comparison | `vit_comparison.py`, `run_vit_comparison.py`, `vit_comparison_controller.py` | `scripts/analysis/summarize_vit_comparison.py` |
| Calibrated text comparison | `run_full_calibrated_text_comparison.py` | `scripts/analysis/summarize_full_calibrated_text_comparison.py` |
| Threshold, timing-noise, and deadline-margin sweeps | `calibrated_three_sweeps.py`, `run_calibrated_three_sweeps.py`, `run_calibrated_three_sweep_task.py` | `scripts/analysis/summarize_calibrated_three_sweeps.py` |
| Calibrated ViT noise comparison | `run_calibrated_noise_vit.py` | `scripts/analysis/plot_calibrated_noise_progress.py` |

The `ubai/` directory contains only cluster deployment and task wrappers for these campaigns. It must follow the cluster instructions and must not become a second implementation of evaluator semantics.

## Small checks

`quick_vit_check.py`, `quick_text_check.py`, and `quick_calibrated_text_check.py` are bounded diagnostic runs. They do not produce publication results and must identify their frozen evaluator source.

## Compatibility shell drivers

The flat `.sh` files predate the manifest-based controllers and remain for reproduction of the original workflows. They are compatibility entry points, not templates for new campaign code. New multi-condition work should use a manifest, an immutable condition identity, resumable results, and a separate analysis program.

## Rules for new work

1. Add model or metric behavior to an `error_analysis_*` evaluator or reusable `utils/` module.
2. Keep scientific conditions and resume rules in a campaign contract, separate from host scheduling.
3. Import generic host helpers from their single owner in `scripts/runtime/`, never from another campaign controller or cluster deployment module.
4. Write outputs only under `artifacts/` and keep generated files out of the source tree.
5. Add a focused verifier and document the behavior in `lat.md/`.

Source paths recorded in completed artifact manifests identify the historical commit. Current code does not retain duplicate implementations merely to reproduce an old path; use the recorded commit when reproducing that artifact.
