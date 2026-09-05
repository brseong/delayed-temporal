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

`ToyMLP` supports a separately trained `relu` primary model and a separately trained bounded-positive `sigmoid` control. Checkpoints and conversion manifests store that activation, so changing `--activation` cannot silently reinterpret a ReLU checkpoint as sigmoid.

[[utils/hardware/brainscales2/toy.py#convert_float_model]] maps inputs and hidden activations to UInt5, affine coefficients to signed int6, and readout values to Int8. Biases become weights on a constant UInt5 lane; ReLU uses an unlabeled-calibration integer shift, while sigmoid has its fixed `[0,1] -> [0,31]` decoding scale.

[[utils/hardware/brainscales2/toy.py#ConvertedToyModel]] is the pure-PyTorch integer reference. It exposes the hidden UInt5 tensor as the only physical TTFS insertion point and consumes the pooled UInt5 tensor in the unchanged second affine layer.

### Hybrid Hagen and TTFS execution

The host switches between Hagen PWM execution and spiking LIF execution, so measured accuracy includes both physical stages while latency and energy are not end-to-end hardware claims.

[[utils/hardware/brainscales2/hagen.py#HagenPWMBackend]] lazy-loads `hxtorch.perceptron`, executes bias-free physical affine layers with the converted constant lane, and records calibration, chip, tiling, activation-boundary, and timing metadata.

The formal default does not invoke Hagen `ConvertingReLU`. It scales the cached raw first PWM preactivation into a [[utils/transforms/types.py#Potential]] with $V_{lb}=0$ and upper UInt5 bound 31, applies that declared bound in the adapter, and passes the resulting UInt5 values to the reused TTFS encoder. This is an explicit host-mediated Hagen-to-spiking representation boundary because the two modes are released and reinitialized separately; it must not be described as a continuous on-chip lower clamp. `--relu-boundary hagen-converting-relu` remains only as an explicit-Hagen baseline, and both choices record their clamp counts and provenance in manifests.

The sigmoid control runs `Hagen raw affine -> host sigmoid -> UInt5 Potential -> TTFS pool`. It intentionally does not claim that the public hxtorch graph realizes the paper's $\phi_{\mathrm{NL}}$ and constant-reference $\psi_{\mathrm{ED}}$ circuit: `sigmoid_physical_subcircuit=false` and host-adapter scale/range metadata are required in every artifact. The same pooled UInt5 rail and physical LIF graph are therefore exercised, while activation-circuit evidence remains out of scope.

The integer reference shift applies to its int32 accumulator, whereas physical Hagen output is already Int8. A separate Hagen hidden shift defaults to one and the probe recommends it from unlabeled calibration activations; physical output logits receive no second software shift.

For inputs wider than one signed Hagen array, the adapter first probes the high-level `Linear` path. Its explicit `host-128` fallback runs 128-input analog MAC tiles, sums partial Int8 values on the host, and records that host accumulation rather than presenting it as one on-chip matrix operation.

TTFS-domain pooling runs Hagen with `avg=1` and assigns `M` LIF replicas to each logical hidden unit. Potential-domain pooling runs Hagen with `avg=M` and one downstream LIF. The EBRAINS acceptance launcher selects TTFS replicas with finite-$M$ analytic corrected-max decoding; Hagen `avg=M` remains a separately labeled potential-domain comparison and is never conflated with neuron pooling.

An all-miss logical pool decodes to zero on the positive UInt5 rail. Artifacts retain the all-miss mask and also report accuracy on samples without any all-miss hidden activation, preventing silent sample deletion or fabricated deadline spikes.

Every condition retains the actual Hagen UInt5 tensor presented to the LIF stage as its nominal activation. Miss rates are split between nominal code zero and positive support, and non-miss decoded error is reported as UInt5 bias and MAE both overall and for each code from 0 through 31.

Two paired causal controls operate on the same pooled hidden tensor. The miss-repair oracle replaces only all-miss positions with the ideal converted hidden code before the selected readout, while the readout ablation sends the unmodified pooled tensor through the deterministic integer PyTorch second layer. A torch-readout oracle is also retained so miss repair can be compared without analog readout noise.

### Network pool placement

Dedicated mapping preserves persistent physical identity, while time-multiplexed mapping deliberately reuses a small pool and is reported as a different hardware method.

[[utils/hardware/brainscales2/toy_pooling.py#resolve_grouped_physical_coordinates]] allocates `30 * M` unique neurons for the 30-hidden-unit models. `local-pool` keeps every logical pool inside one quadrant while distributing pools across quadrants; `cross-quadrant` distributes every pool's replicas across quadrants.

At `M=16`, dedicated placement uses 480 of 512 neuron circuits. The 128-hidden-unit MNIST model therefore requires time multiplexing, and a local pool is repeated in the coordinate result to make reuse explicit.

[[utils/hardware/brainscales2/toy_pooling.py#GroupedHardwarePoolBackend]] constructs one grouped-broadcast `Synapse -> LIF` graph for dedicated mapping. Each logical source expands into the operating point's simultaneous fan-in lanes, which project only to that source's replica block; the network path never substitutes the primitive experiment's unreliable independent-input routing condition.

Full hardware evaluation slices samples and caps `pool_size * samples` at 128 replica-samples, so M=8 and M=16 use 16 and 8 samples. Every chunk runs in a disposable child process to release native hxtorch memory under the 2 GB EBRAINS limit, then [[utils/hardware/brainscales2/toy_pooling.py#concatenate_toy_pool_results]] restores order and provenance.

Timing calibration is acquired once per physical condition in disposable four-trial workers. Their raw events are concatenated before offset estimation across all 32 UInt5 codes, and every inference chunk reuses the same checksummed calibration; calibration and inference batches never coexist in one M=16 graph.

Before formal evaluation, [[scripts/evaluation/brainscales2_toy_hil.py#margin_calibration_phase]] measures unlabeled calibration activations at $M=1$ with a 100 microsecond diagnostic deadline. Code zero is excluded; for each 1 microsecond candidate extension from 0 to 40 microseconds, it bootstraps trials and samples and computes the upper confidence bound for the sample-level event that any positive hidden unit misses. The smallest common margin whose bound is at most 5% in both placements is selected, while nonfires at the diagnostic deadline form a structural floor and block the run when no candidate passes.

The selected margin extends only the observation deadline: the TTFS input window, UInt5 bounds, weights, and activation values are unchanged. Formal evaluation interleaves the selected margin and a zero-margin control over identical cached hidden inputs, every pool size, and both placements; the calibration context binds checkpoints, both calibration files, chip operating parameters, and the time grid before reuse.

After temporal decoding, the physical Hagen readout also slices the flattened trial-sample row axis before each PWM call. When any all-miss position exists, original and oracle-repaired rows are concatenated and tagged as separate segments; the logits retain row order, and every chunk records calibration, chip, shape, and elapsed time.

Formal multi-condition runs materialize each required physical Hagen hidden tensor once, then execute every placement and pool size in a fresh child process. Completed worker directories are resumable, and the parent rebuilds the combined artifact so process isolation does not change paired inputs or the result schema.

Each child process is retried only up to a configured bound with increasing backoff. A no-output watchdog terminates the isolated process group when a native RPC call does not return. Retries restart the condition; only complete artifacts are reused, and attempt logs and status enter the manifest.

### Local mock and replay evidence

Synthetic and artifact-replay backends validate accuracy propagation before hardware use but are not promoted to new physical evidence.

[[utils/hardware/brainscales2/toy_pooling.py#MockToyPoolBackend]] generates coordinate-static, trial-shared, replica-local, and missing-event terms. Calibration events estimate response delay and persistent offsets before inference events are decoded.

[[utils/hardware/brainscales2/toy_pooling.py#ReplayToyPoolBackend]] splits the allowlisted primitive artifact by trial, estimates timing calibration only from the first half, and samples residuals and misses only from the held-out half. It reuses the primitive distribution across logical units and therefore declares `rough-model-only` scope.

### Network result contract

Network artifacts join float, ideal-converted, and physical predictions with complete intermediate tensors and bounded human-readable event extracts.

[[utils/hardware/brainscales2/toy_artifacts.py#write_toy_artifacts]] writes conversion and run manifests, prediction variants, accuracy and NLL drops, paired recovery intervals, same-run zero-versus-selected-margin comparisons, support-stratified miss metrics, per-code UInt5 error, readout ablations, raw timing tensors, and figures. `intermediates.pt` remains lossless; `events.csv` records a deterministic sample/trial subset.

Accepted physical runs use a per-run Git allowlist. The committed bundle keeps checkpoints, calibration, manifests, metrics, predictions, figures, and compressed event extracts; oversized lossless tensors and worker chunks remain external and are represented by size and SHA-256 in `artifact_inventory.json`.

[[scripts/evaluation/brainscales2_toy_hil.py#main]] separates train, convert, local evaluation, Hagen probe, deadline-margin calibration, hardware smoke, and full hardware phases. MNIST hardware evaluation defaults to a 128-sample runtime benchmark until the caller explicitly sets a formal sample count.

The EBRAINS notebook defaults to a one-pass Yin-Yang acceptance pipeline: train, convert, Hagen probe, deadline-margin calibration, hardware smoke, and full evaluation. It configures the shared client from a writable `/tmp` checkout, pins both calibrations, applies the probe-selected shift, and blocks formal execution unless same-run smoke passes. Failures stop later stages and enter `pipeline_status.json`.

## Toy ANN2SNN Verification

These test specifications protect the conversion and network-level hardware boundary without requiring hxtorch locally.

### Host-mediated implicit ReLU boundary

The default hidden boundary must lower raw PWM values through the declared $V_{lb}=0$ Potential range without calling `ConvertingReLU`, retain UInt5 upper saturation, and label the result as host-mediated rather than continuous on-chip activation.

### Sigmoid host activation adapter

The bounded sigmoid control must require a separately labeled checkpoint, quantize its host sigmoid output to the same UInt5 rail, and record that no physical sigmoid subcircuit was executed.

### Deterministic datasets and frozen conversion

Yin-Yang splits must reproduce from their registered seeds, converted ranges must match the Hagen integer contract, and conversion must leave the float parameter hash unchanged.

### Physical pool allocation

Dedicated 30-by-16 placements must contain 480 unique in-range coordinates with the requested quadrant topology, while oversized dedicated mappings must fail.

### Grouped broadcast fan-in

Each logical hidden source must occupy its own simultaneous fan-in lane block and connect every lane only to that logical unit's physical replicas.

### Chunked pool aggregation

Hardware chunks must concatenate on samples without altering trial, neuron, replica, coordinate, or miss-mask semantics.

The effective hardware chunk must not exceed its configured replica-sample budget, and provenance must retain requested and effective sizes. Split calibration trials must preserve code order and coordinates, produce one shared offset estimate, and remain resumable independently of inference chunks.

### Pool-size-aware hardware chunk cap

The physical pool chunk must reduce with pool size so every grouped graph stays below its replica-sample memory budget.

A requested sample chunk remains an upper bound; the effective size is `min(requested, budget // pool_size)` with a minimum of one sample.

### Pool chunk process isolation

Every full-run hardware chunk must execute in a disposable child, persist its result and attempt status, and be reusable after a later chunk or outer condition is killed.

### Hagen output row chunking

Physical Hagen readout chunks must preserve flattened trial-sample row order, bound each PWM call, and retain per-chunk provenance in the aggregate metadata.

### Condition process isolation

Each hardware condition must run in a fresh process, reuse the matching shared first-hidden tensor, resume only matching completed worker configs, and aggregate into the standard paired artifact schema.

### Transient worker retry

A failed child must retry only to the configured bound, persist every attempt, and re-raise after exhaustion. A silent child must hit its idle timeout and terminate only its isolated process group.

Retries use increasing backoff, and a recovered worker returns normally to condition aggregation.

### Miss-aware temporal decoding

Seeded mock observations must reproduce exactly, normal timing must decode near its UInt5 source, and an all-miss pool must become zero while retaining its miss mask.

### Max estimator attribution

Temporal pooling separates drop-aware activation mean, raw maximum, finite-$M$ deadline-corrected maximum, and calibration-curve-corrected maximum before the frozen integer readout.

`mean` maps every missed replica to UInt5 zero before averaging. `raw-max` maps TTFS earliest-event selection to activation maximum, so its pooled miss probability decreases while its order-statistic bias remains.

`analytic-corrected-max` estimates the delivered residual scale and codewise deadline tail only from calibration events, then adds the finite-$M$ conditional earliest-time offset. `empirical-corrected-max` monotonically inverts the codewise calibration response without labels.

All four estimators consume the same raw-event tensor, retain all-miss-to-zero semantics, and run through the same frozen readout. Replay results remain rough model-selection evidence; hardware acceptance requires an independent calibration acquisition and evaluation events.

### Deadline margin selection

Margin calibration must exclude code zero and choose the smallest hardware-grid deadline extension whose hierarchical-bootstrap miss upper bound passes in every placement.

All candidates for a placement reuse the same bootstrap draws so the nested miss curve remains monotone. A persistent diagnostic-deadline floor must produce no selection rather than being hidden by a larger margin.

### Deadline margin provenance

A selected margin may be reused only with the exact unlabeled model, calibration files, timing grid, physical operating point, and complete 32-code timing correction that produced it.

Changed checkpoints, calibration checksums, neuron parameters, or corrected-max tables must invalidate reuse. The margin changes only the deadline and must not modify the encoded activation interval.

### Paired deadline comparison

Every formal margin run must report selected-versus-zero deadline changes from matched task, placement, mapping, pooling method, pool size, and cached hidden inputs.

The artifact records paired accuracy confidence intervals and positive-code sample miss-rate changes separately, preventing conditions at different margins from sharing an $M=1$ baseline.

### Held-out hardware replay

Replay must derive calibration from the first trial half, sample only the held-out half, reproduce from its seed, and identify itself as rough modeling rather than physical network evidence.

### Hagen tiling contract

Host tiling must cover every input column once and agree with an untiled integer accumulation when partial values do not saturate.

### Network artifact contract

Network evaluation must emit the stable manifest, runtime, metrics, predictions, event extract, and complete intermediate tensor files with consistent shapes.

### Hardware error attribution

Each condition must split zero/positive-code misses, report non-miss UInt5 error by code, repair only all-miss positions, and retain physical, torch-readout, and repaired logits.

### EBRAINS launcher contract

New source files must parse as Python 3.11, and the notebook must remain a thin launcher with an enabled Yin-Yang acceptance pipeline, explicit stage flags, and MNIST sample limits.

Training must precede hardware allocation, the probe-selected shift must feed smoke, and formal stages must be gated by a passing same-run smoke artifact.
