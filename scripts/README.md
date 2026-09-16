# Script layout

The script tree separates stable model-family evaluation from reproducible campaign orchestration and post-run analysis.

## Evaluation entry points

`evaluation/` contains the four maintained `error_analysis_*` command-line programs. They are the public experiment interface and may import reusable model or runtime support, but they must not import a campaign controller.

See [`evaluation/README.md`](evaluation/README.md) for the exact responsibilities of each entry point.

## Experiment orchestration

`experiments/` contains condition grids, manifests, local and Slurm scheduling, resume logic, and campaign-specific evidence validation. A controller may invoke an evaluator; it must not become the only place where a metric or model-family behavior is implemented.

Several completed campaigns are source-hashed in their artifacts. Their recorded paths are retained for reproducibility even when the implementation is no longer a template for new work. See [`experiments/README.md`](experiments/README.md) for the campaign map.

## Reusable runtime support

`runtime/` contains host and process helpers that do not know about model families or scientific conditions. It is the only owner of filesystem writes, artifact identity, device admission, worker environments, and Slurm output parsing.

- `files.py` owns safe paths and durable mutable or immutable writes.
- `identity.py` owns file, structured record, artifact, package source, and source checkout identities.
- `environment.py` owns worker scratch and cache environment variables.
- `local_gpu.py` owns device discovery, idle admission, and checks for one allocated device.
- `slurm.py` owns scheduler queue parsing.

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

Do not import a campaign module to reuse a filesystem, identity, environment, device, or scheduler helper. Move the behavior and all callers to `scripts/runtime/`, remove the superseded definition, and test the public invariant there.
