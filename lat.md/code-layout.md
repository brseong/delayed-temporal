# Script Architecture

The script tree keeps model-family evaluation stable while isolating reusable runtime support, campaign orchestration, artifact analysis, and verification.

## Stable Evaluation Interface

The four `error_analysis_*` programs are the maintained command-line interface for ViT, BERT, RoBERTa, and GPT-2 evaluation.

They own model and dataset loading, preprocessing, backend selection, calibration attachment, progress records, metrics, and diagnostic counters. They do not import experiment controllers. Campaigns supply conditions and invoke evaluators without redefining evaluator semantics.

## Reusable Runtime Support

Generic host operations live under `scripts/runtime` and have no dependency on a model family, scientific condition, or campaign.

Atomic status replacement, immutable evidence writes, artifact identity, worker environments, local device admission, and Slurm parsing each have one owner in this layer. Campaign modules consume these APIs and do not expose substitute implementations.

The owners are `files.py`, `identity.py`, `environment.py`, `local_gpu.py`, and `slurm.py`, respectively. Content and source checks use `identity.py`; mutable and immutable output paths use `files.py`.

## Campaign and Artifact Boundary

Experiment modules own manifests, scientific condition grids, scheduling, retries, and evidence validation; analysis modules read completed artifacts and create aggregate outputs.

Completed manifests identify their historical source commit and file hashes. Reproduction uses that commit; the current implementation does not keep duplicate policy or compatibility wrappers solely to preserve an old source path. Generated outputs remain under `artifacts/`.

## Dependency Boundary Verification

The layout check prevents stable evaluators and generic runtime helpers from depending on campaign controllers.

[[scripts/verification/verify_script_layout.py#verify_layout]] requires all four evaluator entry points, enforces the owner of every generic helper, rejects runtime imports from dependent layers, and prevents campaign or cluster modules from serving as generic helper libraries.
