# Script layout

The script tree separates stable model-family evaluation from reproducible campaign orchestration and post-run analysis.

## Evaluation entry points

`evaluation/` contains the four maintained `error_analysis_*` command-line programs. They are the public experiment interface and may import reusable model or runtime support, but they must not import a campaign controller.

See [`evaluation/README.md`](evaluation/README.md) for the exact responsibilities of each entry point.

## Experiment orchestration

`experiments/` contains condition grids, manifests, local and Slurm scheduling, resume logic, and campaign-specific evidence validation. A controller may invoke an evaluator; it must not become the only place where a metric or model-family behavior is implemented.

Several completed campaigns are source-hashed in their artifacts. Their recorded paths are retained for reproducibility even when the implementation is no longer a template for new work. See [`experiments/README.md`](experiments/README.md) for the campaign map.

## Reusable runtime support

`runtime/` contains host and process helpers that do not know about model families or scientific conditions. GPU admission, atomic status writes, and Slurm output parsing live here instead of being imported from an unrelated campaign controller.

## Analysis, setup, and verification

- `analysis/` reads completed artifacts, aggregates results, and creates figures or tables. It does not schedule evaluation work.
- `setup/` prepares or hashes external assets. It does not evaluate a model.
- `verification/` checks operator behavior, evaluator contracts, campaign evidence, and repository structure.
- `notebooks/` is exploratory and is not a maintained execution interface.
- `lib/` contains compatibility shell helpers. New reusable Python support belongs in `runtime/`.

## Dependency direction

The intended dependency direction is:

```text
experiments  ->  evaluation  ->  utils
     |               |
     +----------> runtime <---+

analysis     ->  artifacts
verification ->  any maintained layer
```

Do not import `scripts.experiments.run_*` merely to reuse a filesystem, GPU, or scheduler helper. Move that helper to `scripts/runtime/` and test it independently.

