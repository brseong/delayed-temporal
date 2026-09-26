# Experiment campaigns

This directory orchestrates reproducible runs around the stable `scripts/evaluation/error_analysis_*` entry points.

## Maintained Python campaigns

The multi-file Python campaigns use a contract module, one or more workers or controllers, and a matching summarizer or verification program.

`reproduce_iclr_2027.py` is the paper reproduction entry point. It inventories the campaign owners below, verifies their completed evidence, regenerates every active publication figure, and compiles the manuscript. It does not duplicate evaluator or campaign semantics and keeps raw experiments on their required hosts.

| Campaign | Contract and runners | Result processing |
|---|---|---|
| Calibrated ViT comparison | `run_full_calibrated_vit_comparison.py` | `scripts/analysis/summarize_local_range_paper_campaign.py` |
| Calibrated text comparison | `run_full_calibrated_text_comparison.py` | `scripts/analysis/summarize_full_calibrated_text_comparison.py` |
| Local-window timing-noise and deadline-margin sweeps | `run_vit_local_range_noise_condition.py`, `run_poseidon_local_range_paper_campaign.py` | `scripts/analysis/summarize_local_range_paper_campaign.py` |
| Appendix raw-timestamp ViT-B rerun | `run_appendix_vit_noise_campaign.py`, `run_vit_local_range_noise_condition.py` | `scripts/analysis/summarize_local_range_paper_campaign.py --noise-only` |
| Discrete-time compatibility | `run_clock_driven_vit.py` | `scripts/analysis/plot_clock_discretization.py`, `scripts/analysis/plot_clock_time_step_sweep.py` |
| Llama 2 7B two-dataset timing-noise diagnostic | `run_llama_appendix_noise.py` | Validated JSON summary and TeX table under the diagnostic artifact root |

Historical UBAI deployment scripts were removed with their global-range campaigns. Any future cluster wrapper must follow the cluster instructions and must not become a second implementation of evaluator semantics.

## Small checks

`quick_text_check.py` and `quick_calibrated_text_check.py` are bounded diagnostic runs. They do not produce publication results and must identify their frozen evaluator source.

The Llama appendix diagnostic uses one frozen calibration collected from 32 WikiText-2 training texts, then evaluates the first 128 WikiText-2 and IMDb test texts with the same Llama 2 7B checkpoint. Its noise-free reference and three measured-noise multipliers (`0.00001`, `0.0001`, `0.001`) are distinct from the full Figure 3a population. Noisy conditions use seeds 0, 1, and 2; the summary and TeX table report mean perplexity and 95% Student-$t$ confidence intervals. `run_llama_appendix_noise.py` checks checkpoint, implementation, calibration, dataset, hardware summary, and condition identities before reusing a result. It writes a summary and TeX table only when every dataset, multiplier, and seed is complete. Use `--mode plan` to inspect missing cells, `--mode run --seeds 0 1 2 --gpus ...` to fill them, and `--mode summarize --seeds 0 1 2` after all cells finish. On UBAI, run it only inside a Slurm compute allocation, never on a gate node.

## Retired execution paths

The flat shell drivers and cross-campaign calibration-reuse path were removed. Publication experiments must use the maintained Python campaign controllers, which fix their artifact tags and reject calibration reuse from earlier campaigns. Historical results remain evidence, not executable inputs to a current campaign.

## Rules for new work

1. Add model or metric behavior to an `error_analysis_*` evaluator or reusable `utils/` module.
2. Keep scientific conditions and resume rules in a campaign contract, separate from host scheduling.
3. Import generic host helpers from their single owner in `scripts/runtime/`, never from another campaign controller or cluster deployment module.
4. Write outputs only under `artifacts/` and keep generated files out of the source tree.
5. Add a focused verifier and document the behavior in `lat.md/`.

Source paths recorded in completed artifact manifests identify the historical commit. Current code does not retain duplicate implementations merely to reproduce an old path; use the recorded commit when reproducing that artifact.
