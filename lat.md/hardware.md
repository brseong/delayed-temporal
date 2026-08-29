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

[[utils/hardware/brainscales2/config.py#CADCDiagnosticResult]] stores paired no-input and single-input membrane/spike traces shaped `[trial, time, neuron]`. CADC values remain hardware ADC units and are never interpreted directly as threshold parameter codes.

## Execution Backends

Physical and synthetic executions implement one pool-level interface while keeping EBRAINS-only imports optional.

[[utils/hardware/brainscales2/backend.py#BrainScaleS2PoolBackend]] lazy-loads hxtorch, constructs `Synapse -> LIF`, pins the requested logical neurons, reads the underlying sparse SpikeHandle data, converts FPGA ticks to seconds, and releases hardware in a `finally` block. Current releases expose raw observables on the LIF module; pre-13 EBRAINS releases retain them in the experiment's hardware-data extractor keyed by the LIF population descriptor.

[[utils/hardware/brainscales2/backend.py#_configure_experiment_calibration]] detects the hxtorch execution model. Legacy releases load a pinned `.pbin` through `default_execution_instance.load_calib`; modern releases use `hxtorch.core` or a public/private grenade `FixtureCalibration` binding. Result metadata records the selected path.

Legacy hxtorch treats a portable-binary calibration as a fixed chip configuration: requested threshold, time constants, and synaptic gain are not recalibrated. The backend marks those requests as unapplied and tunes delivery with `input_fan_in`, using shared lanes for broadcast and neuron-dedicated lanes for independent routing.

Dense spike grids are not accepted as a fallback output because their configured `dt` can hide the jitter being measured. Integer FPGA timestamps use the grenade clock constant unless a release-specific scale is explicitly supplied. A configurable 50-microsecond inter-batch guard prevents residual membrane state from leaking between samples and trials.

[[utils/hardware/brainscales2/backend.py#MockPoolBackend]] provides deterministic seeded static offsets, trial-shared disturbances, neuron-local residuals, and misses. It validates the complete local pipeline without claiming physical calibration.

[[utils/hardware/brainscales2/backend.py#BrainScaleS2PoolBackend#diagnose_cadc]] is the sole dense-observable exception. It records fixed-neuron baseline and one-PSP CADC traces before a run; accepted jitter data still comes only from sparse raw spike timestamps.

## Placement and Routing Ablations

Pooling conditions distinguish spatial and input-path sharing rather than assuming independent and identically distributed neuron noise.

[[utils/hardware/brainscales2/backend.py#resolve_physical_neuron_indices]] selects either contiguous neurons in one quadrant or a round-robin set across all four quadrants. Broadcast routing shares one input source, while independent routing supplies a diagonal one-source-per-neuron projection.

Every result and manifest records the chosen atomic-neuron indices. Repeated trials of `M=1` form the temporal-repeatability baseline, and larger pools expose the spatial covariance structure.

## Pooling Analysis and Artifacts

Analysis separates calibration-only offsets from held-out pooling statistics and retains missing-event behavior.

[[utils/hardware/brainscales2/analysis.py#calibrate_pool]] estimates a sample trajectory and persistent neuron offsets on the first trial half. [[utils/hardware/brainscales2/analysis.py#pool_first_spikes]] evaluates corrected mean, uncorrected mean, median, and earliest-event estimators on held-out trials while leaving all-miss pools undefined.

[[utils/hardware/brainscales2/analysis.py#fit_variance_floor]] fits `Var(pool)=a/M+c` by valid-count weighted least squares. The bootstrap resamples held-out trials within each pool size to provide confidence intervals for the reducible component `a` and shared floor `c`.

[[utils/hardware/brainscales2/artifacts.py#write_experiment_artifacts]] writes a manifest, long-form raw event CSV, tensor archive, summary table, variance fit, and plot. The manifest includes configuration, calibration hash, chip and software metadata, placement, routing, and input-domain provenance.

Generated artifacts are ignored by default. The repository allowlist admits only `artifacts/brainscales2/20260829T084007Z/`, preserving this accepted full-run bundle without making later local or EBRAINS outputs implicitly trackable.

[[utils/hardware/brainscales2/analysis.py#analyze_cadc_diagnostic]] compares a paired one-PSP response against no-input excursions. [[utils/hardware/brainscales2/artifacts.py#write_cadc_diagnostic_artifacts]] writes the traces, per-neuron summary, plot, and recommendation without converting CADC amplitudes into threshold codes.

## Experiment Entry Point and Verification

The CLI owns operating-point selection, condition orchestration, and reproducible output while the notebook remains a thin launcher.

[[scripts/evaluation/brainscales2_pooling.py#main]] supports CADC diagnosis, mock and hardware runs, raw-spike calibration, fixed potential sweeps, and artifact generation. Calibration can sweep synaptic fan-in as a digital delivery control and penalizes misses, multiple spikes, and spikes before their nominal input.

The launcher targets the EBRAINS experimental kernel's Python 3.11 runtime. [[utils/transforms/types.py#NeuralTransform]] and [[utils/transforms/noise.py#inject_spike_time_noise]] use legacy `TypeVar` and `ParamSpec` declarations so the reused project encoders import there. `scripts/notebooks/ebrains_brainscales2_pooling.ipynb` installs only `jaxtyping` and `matplotlib` with kernel-scoped `%pip`, invokes the CLI through `sys.executable`, and runs CADC diagnosis, a bounded raw-spike sweep, hardware smoke, then the full condition grid by default. It loads the official demo helpers from a writable `/tmp` checkout and saves the nightly `.pbin` in the run directory. Every hardware stage receives either an explicit user override or that same run-local file, preventing an implicit full-chip calibration while preserving a checksum-addressable input.

[[scripts/verification/verify_brainscales2_pooling.py#main]] checks affine and logarithmic endpoints, routing shapes, invalid domains, the software-noise guard, placement, raw-event reduction, mock reproducibility, synthetic CADC separation, variance-floor recovery, artifacts, and the safe EBRAINS notebook launcher contract. Notebook metadata must identify Python 3.11, while Jupyter-written patch suffixes such as `3.11.10` remain valid. Saved run flags may be either false or true because users enable stages in place; verification requires each mutable flag and its execution guard rather than its current value.

## Toy ANN2SNN Hardware-in-the-Loop

The toy path measures whether physical hidden-neuron pooling recovers classification accuracy after deterministic ANN-to-TTFS conversion, without claiming a host-free Transformer deployment.

### Frozen ANN and conversion boundary

Float training ends before conversion; parameter hashes prove that range calibration, quantization, and hardware execution never update the ANN weights.

[[utils/hardware/brainscales2/toy.py#ToyMLP]] defines one-hidden-layer Yin-Yang and MNIST classifiers. Yin-Yang uses `4-30-3`; MNIST uses `784-30-10` for dedicated pools and `784-128-10` for time-multiplexed pools.

[[utils/hardware/brainscales2/toy.py#convert_float_model]] maps inputs and hidden activations to UInt5, affine coefficients to signed int6, and readout values to Int8. Biases become weights on a constant UInt5 lane, and unlabeled calibration activations select the integer shifts.

[[utils/hardware/brainscales2/toy.py#ConvertedToyModel]] is the pure-PyTorch integer reference. It exposes the hidden UInt5 tensor as the only physical TTFS insertion point and consumes the pooled UInt5 tensor in the unchanged second affine layer.

### Hybrid Hagen and TTFS execution

The host switches between Hagen PWM execution and spiking LIF execution, so measured accuracy includes both physical stages while latency and energy are not end-to-end hardware claims.

[[utils/hardware/brainscales2/hagen.py#HagenPWMBackend]] lazy-loads `hxtorch.perceptron`, executes bias-free physical affine layers with the converted constant lane, invokes `ConvertingReLU` before releasing hardware, and records calibration, chip, tiling, and timing metadata.

The integer reference shift applies to its int32 accumulator, whereas physical Hagen output is already Int8. A separate Hagen hidden shift defaults to one and the probe recommends it from unlabeled calibration activations; physical output logits receive no second software shift.

For inputs wider than one signed Hagen array, the adapter first probes the high-level `Linear` path. Its explicit `host-128` fallback runs 128-input analog MAC tiles, sums partial Int8 values on the host, and records that host accumulation rather than presenting it as one on-chip matrix operation.

TTFS-domain pooling runs Hagen with `avg=1` and assigns `M` LIF replicas to each logical hidden unit. Potential-domain pooling is an alternative method that runs Hagen with `avg=M` and one downstream LIF; it is not mixed into the primary TTFS estimator.

An all-miss logical pool decodes to zero after ReLU. Artifacts retain the all-miss mask and also report accuracy on samples without any all-miss hidden activation, preventing silent sample deletion or fabricated deadline spikes.

### Network pool placement

Dedicated mapping preserves persistent physical identity, while time-multiplexed mapping deliberately reuses a small pool and is reported as a different hardware method.

[[utils/hardware/brainscales2/toy_pooling.py#resolve_grouped_physical_coordinates]] allocates `30 * M` unique neurons for the 30-hidden-unit models. `local-pool` keeps every logical pool inside one quadrant while distributing pools across quadrants; `cross-quadrant` distributes every pool's replicas across quadrants.

At `M=16`, dedicated placement uses 480 of 512 neuron circuits. The 128-hidden-unit MNIST model therefore requires time multiplexing, and a local pool is repeated in the coordinate result to make reuse explicit.

[[utils/hardware/brainscales2/toy_pooling.py#GroupedHardwarePoolBackend]] constructs one grouped-broadcast `Synapse -> LIF` graph for dedicated mapping. Every logical input source projects only to its own replica block; the network path never substitutes the primitive experiment's unreliable independent-input routing condition.

### Local mock and replay evidence

Synthetic and artifact-replay backends validate accuracy propagation before hardware use but are not promoted to new physical evidence.

[[utils/hardware/brainscales2/toy_pooling.py#MockToyPoolBackend]] generates coordinate-static, trial-shared, replica-local, and missing-event terms. Calibration events estimate response delay and persistent offsets before inference events are decoded.

[[utils/hardware/brainscales2/toy_pooling.py#ReplayToyPoolBackend]] splits the allowlisted primitive artifact by trial, estimates timing calibration only from the first half, and samples residuals and misses only from the held-out half. It reuses the primitive distribution across logical units and therefore declares `rough-model-only` scope.

### Network result contract

Network artifacts join float, ideal-converted, and physical predictions with complete intermediate tensors and bounded human-readable event extracts.

[[utils/hardware/brainscales2/toy_artifacts.py#write_toy_artifacts]] writes conversion and run manifests, predictions, accuracy and NLL drops, paired recovery intervals, miss metrics, raw timing tensors, and accuracy, confusion, and variance figures. `intermediates.pt` remains lossless; `events.csv` records a deterministic sample/trial subset to avoid duplicating millions of tensor entries.

[[scripts/evaluation/brainscales2_toy_hil.py#main]] separates train, convert, local evaluation, Hagen probe, hardware smoke, and full hardware phases. MNIST hardware evaluation defaults to a 128-sample runtime benchmark until the caller explicitly sets a formal sample count.

The EBRAINS notebook keeps all run flags disabled by default, requires distinct explicit Hagen and spiking calibration paths, configures the shared client from a writable `/tmp` demo checkout, and delegates every experiment to the CLI.

## Toy ANN2SNN Verification

These test specifications protect the conversion and network-level hardware boundary without requiring hxtorch locally.

### Deterministic datasets and frozen conversion

Yin-Yang splits must reproduce from their registered seeds, converted ranges must match the Hagen integer contract, and conversion must leave the float parameter hash unchanged.

### Physical pool allocation

Dedicated 30-by-16 placements must contain 480 unique in-range coordinates with the requested quadrant topology, while oversized dedicated mappings must fail.

### Miss-aware temporal decoding

Seeded mock observations must reproduce exactly, normal timing must decode near its UInt5 source, and an all-miss pool must become zero while retaining its miss mask.

### Held-out hardware replay

Replay must derive calibration from the first trial half, sample only the held-out half, reproduce from its seed, and identify itself as rough modeling rather than physical network evidence.

### Hagen tiling contract

Host tiling must cover every input column once and agree with an untiled integer accumulation when partial values do not saturate.

### Network artifact contract

Network evaluation must emit the stable manifest, runtime, metrics, predictions, event extract, and complete intermediate tensor files with consistent shapes.

### EBRAINS launcher contract

New source files must parse as Python 3.11, and the notebook must remain a thin opt-in launcher with explicit hardware flags and MNIST sample limits.
