# Experiment campaigns

This directory orchestrates reproducible runs around the stable `scripts/evaluation/error_analysis_*` entry points.

## Maintained Python campaigns

The multi-file Python campaigns use a contract module, one or more workers or controllers, and a matching summarizer or verification program.

| Campaign | Contract and runners | Result processing |
|---|---|---|
| Calibrated ViT comparison | `run_full_calibrated_vit_comparison.py` | `scripts/analysis/summarize_local_range_paper_campaign.py` |
| Calibrated text comparison | `run_full_calibrated_text_comparison.py` | `scripts/analysis/summarize_full_calibrated_text_comparison.py` |
| Local-window timing-noise and deadline-margin sweeps | `run_vit_local_range_noise_condition.py`, `run_poseidon_local_range_paper_campaign.py` | `scripts/analysis/summarize_local_range_paper_campaign.py` |
| Discrete-time compatibility | `run_clock_driven_vit.py` | `scripts/analysis/plot_clock_discretization.py`, `scripts/analysis/plot_clock_time_step_sweep.py` |

Historical UBAI deployment scripts were removed with their global-range campaigns. Any future cluster wrapper must follow the cluster instructions and must not become a second implementation of evaluator semantics.

## Small checks

`quick_text_check.py` and `quick_calibrated_text_check.py` are bounded diagnostic runs. They do not produce publication results and must identify their frozen evaluator source.

## Retired execution paths

The flat shell drivers and cross-campaign calibration-reuse path were removed. Publication experiments must use the maintained Python campaign controllers, which fix their artifact tags and reject calibration reuse from earlier campaigns. Historical results remain evidence, not executable inputs to a current campaign.

## Rules for new work

1. Add model or metric behavior to an `error_analysis_*` evaluator or reusable `utils/` module.
2. Keep scientific conditions and resume rules in a campaign contract, separate from host scheduling.
3. Import generic host helpers from their single owner in `scripts/runtime/`, never from another campaign controller or cluster deployment module.
4. Write outputs only under `artifacts/` and keep generated files out of the source tree.
5. Add a focused verifier and document the behavior in `lat.md/`.

Source paths recorded in completed artifact manifests identify the historical commit. Current code does not retain duplicate implementations merely to reproduce an old path; use the recorded commit when reproducing that artifact.
