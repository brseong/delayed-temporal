# Script Architecture

The script tree keeps model-family evaluation stable while isolating reusable runtime support, campaign orchestration, artifact analysis, and verification.

## Stable Evaluation Interface

The four `error_analysis_*` programs are the maintained command-line interface for ViT, BERT, RoBERTa, and GPT-2 evaluation.

They own model and dataset loading, preprocessing, backend selection, calibration attachment, progress records, metrics, and diagnostic counters. They do not import experiment controllers. Campaigns supply conditions and invoke evaluators without redefining evaluator semantics.

## Reusable Runtime Support

Generic host operations live under `scripts/runtime` and have no dependency on a model family, scientific condition, or campaign.

Atomic JSON replacement, local GPU telemetry and admission, and Slurm queue parsing are shared runtime concerns. A campaign may import these helpers, but another evaluator or campaign must not import a controller solely to reach them.

## Campaign and Artifact Boundary

Experiment modules own manifests, scientific condition grids, scheduling, retries, and evidence validation; analysis modules read completed artifacts and create aggregate outputs.

Completed manifests may record source paths and file hashes. Those paths remain stable until an explicit migration preserves the old interface and updates identity checks. Generated logs, tables, and figures remain under `artifacts/` and are not source modules.

## Dependency Boundary Verification

The layout check prevents stable evaluators and generic runtime helpers from depending on campaign controllers.

[[scripts/verification/verify_script_layout.py#verify_layout]] requires all four evaluator entry points, rejects imports from experiments into those evaluators, rejects higher-layer imports from runtime helpers, and prevents generic helpers from again being borrowed from the calibrated sweep controller.
