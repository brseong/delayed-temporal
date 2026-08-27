# BrainScaleS-2 Hardware Validation

This path connects bounded project TTFS codewords to a physical neuron-pooling experiment without presenting the tensor operator simulator as an on-chip Transformer backend.

## System Boundary

Hardware pooling is an independent validation path for first-spike variability, not a third `model_backend` and not an end-to-end deployment of ViT, BERT, RoBERTa, or GPT-2.

The maintained Transformer adapters still evaluate algebraic tensor operators. The hardware path accepts a [[utils/transforms/types.py#Potential]], emits nominal input events, runs a small physical Synapse-to-LIF graph, and returns raw observations for statistical analysis.

## Hardware Encoding

The bridge reuses the project encoders while keeping simulated Gaussian timing noise separate from physical noise.

[[utils/hardware/brainscales2/encoding.py#encode_potential_for_brainscales2]] clamps against the declared potential bounds, invokes the identity or logarithmic encoder, rescales its fixed time domain into the configured physical input window, and quantizes it onto the hxtorch input grid.

The encoding result preserves ideal and injected event times, source domains, input shape, and a clamp mask. Broadcast routing creates one input channel; independent routing creates one identical input channel per pool neuron.

If process-wide Gaussian timing noise is enabled, encoding fails instead of composing software noise with hardware variability. The physical experiment therefore measures only events produced by the selected chip and calibration.

## Configuration and Result Contract

One immutable configuration records both project-time conversion and the physical neuron operating point.

[[utils/hardware/brainscales2/config.py#BrainScaleS2PoolConfig]] validates time windows, LIF targets, pool sizes, placement and routing modes, calibration provenance, and mock-noise parameters. Formal hardware runs require an explicit calibration file unless an environment calibration is deliberately allowed for a smoke test.

[[utils/hardware/brainscales2/config.py#PoolRunResult]] stores tensors shaped `[trial, sample, neuron]` for first-spike time, delivery, and spike count. It also records injected input times, physical neuron indices, the original input shape, and backend metadata.

## Execution Backends

Physical and synthetic executions implement one pool-level interface while keeping EBRAINS-only imports optional.

[[utils/hardware/brainscales2/backend.py#BrainScaleS2PoolBackend]] lazy-loads hxtorch, constructs `Synapse -> LIF`, pins the requested logical neurons, reads the underlying sparse SpikeHandle data, converts FPGA ticks to seconds, and releases hardware in a `finally` block.

Dense spike grids are not accepted as a fallback output because their configured `dt` can hide the jitter being measured. Integer FPGA timestamps use the grenade clock constant unless a release-specific scale is explicitly supplied. A configurable 50-microsecond inter-batch guard prevents residual membrane state from leaking between samples and trials.

[[utils/hardware/brainscales2/backend.py#MockPoolBackend]] provides deterministic seeded static offsets, trial-shared disturbances, neuron-local residuals, and misses. It validates the complete local pipeline without claiming physical calibration.

## Placement and Routing Ablations

Pooling conditions distinguish spatial and input-path sharing rather than assuming independent and identically distributed neuron noise.

[[utils/hardware/brainscales2/backend.py#resolve_physical_neuron_indices]] selects either contiguous neurons in one quadrant or a round-robin set across all four quadrants. Broadcast routing shares one input source, while independent routing supplies a diagonal one-source-per-neuron projection.

Every result and manifest records the chosen atomic-neuron indices. Repeated trials of `M=1` form the temporal-repeatability baseline, and larger pools expose the spatial covariance structure.

## Pooling Analysis and Artifacts

Analysis separates calibration-only offsets from held-out pooling statistics and retains missing-event behavior.

[[utils/hardware/brainscales2/analysis.py#calibrate_pool]] estimates a sample trajectory and persistent neuron offsets on the first trial half. [[utils/hardware/brainscales2/analysis.py#pool_first_spikes]] evaluates corrected mean, uncorrected mean, median, and earliest-event estimators on held-out trials while leaving all-miss pools undefined.

[[utils/hardware/brainscales2/analysis.py#fit_variance_floor]] fits `Var(pool)=a/M+c` by valid-count weighted least squares. The bootstrap resamples held-out trials within each pool size to provide confidence intervals for the reducible component `a` and shared floor `c`.

[[utils/hardware/brainscales2/artifacts.py#write_experiment_artifacts]] writes a manifest, long-form raw event CSV, tensor archive, summary table, variance fit, and plot. The manifest includes configuration, calibration hash, chip and software metadata, placement, routing, and input-domain provenance.

## Experiment Entry Point and Verification

The CLI owns operating-point selection, condition orchestration, and reproducible output while the notebook remains a thin launcher.

[[scripts/evaluation/brainscales2_pooling.py#main]] supports mock and hardware runs, quick smoke settings, an operating-point calibration phase, fixed potential sweeps, all configured pooling conditions, and artifact generation. Hardware execution is intended for the EBRAINS experimental kernel and does not add hxtorch to the base requirements.

The launcher targets the EBRAINS experimental kernel's Python 3.11 runtime. [[utils/transforms/types.py#NeuralTransform]] and [[utils/transforms/noise.py#inject_spike_time_noise]] use legacy `TypeVar` and `ParamSpec` declarations so the reused project encoders import there. `scripts/notebooks/ebrains_brainscales2_pooling.ipynb` installs only `jaxtyping` and `matplotlib` with kernel-scoped `%pip`, invokes the CLI through `sys.executable`, disables hardware stages by default, and loads `setup_hardware_client` from a writable `/tmp` checkout of the official demos.

[[scripts/verification/verify_brainscales2_pooling.py#main]] checks affine and logarithmic endpoints, routing shapes, invalid domains, the software-noise guard, placement, raw-event reduction, mock reproducibility, variance-floor recovery, artifacts, and the safe EBRAINS notebook launcher contract. Notebook metadata must identify Python 3.11, while Jupyter-written patch suffixes such as `3.11.10` remain valid.
