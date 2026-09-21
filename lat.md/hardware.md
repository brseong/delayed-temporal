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

The integer reference shift applies to its int32 accumulator, whereas physical Hagen output is already Int8. A separate Hagen hidden shift defaults to one and the probe recommends it from unlabeled calibration activations; physical output logits receive no second software shift. For the default host-mediated boundary, all shift candidates are scored from one physical preactivation tensor because shifting and clamping occur after Hagen. Small models also collapse duplicate 128-row and architecture-width probes.

The same `probe-hagen` phase measures affine fidelity on held-out unlabeled inputs after shift selection. It repeats the first physical affine, applies the frozen hidden boundary, and drives the second physical affine with the ideal hidden UInt5 tensor so the two stages remain separately attributable. `hagen_fidelity.pt` retains paired tensors; JSON and CSV outputs report trial mean error, repeated-trial variation, channelwise affine fits, saturation, correlation, and output argmax agreement.

For inputs wider than one signed Hagen array, the adapter first probes the high-level `Linear` path. Its explicit `host-128` fallback runs 128-input analog MAC tiles, sums partial Int8 values on the host, and records that host accumulation rather than presenting it as one on-chip matrix operation.

TTFS-domain pooling runs Hagen with `avg=1` and assigns `M` LIF replicas to each logical hidden unit. Potential-domain pooling runs Hagen with `avg=M` and one downstream LIF. The EBRAINS acceptance launcher selects TTFS replicas with finite-$M$ analytic corrected-max decoding; Hagen `avg=M` remains a separately labeled potential-domain comparison and is never conflated with neuron pooling.

An all-miss logical pool decodes to zero on the positive UInt5 rail. Artifacts retain the all-miss mask and also report accuracy on samples without any all-miss hidden activation, preventing silent sample deletion or fabricated deadline spikes.

Every condition retains the actual Hagen UInt5 tensor presented to the LIF stage as its nominal activation. Miss rates are split between nominal code zero and positive support, and non-miss decoded error is reported as UInt5 bias and MAE both overall and for each code from 0 through 31.

Two paired causal controls operate on the same pooled hidden tensor. The miss-repair oracle replaces only all-miss positions with the ideal converted hidden code before the selected readout, while the readout ablation sends the unmodified pooled tensor through the deterministic integer PyTorch second layer. A torch-readout oracle is also retained so miss repair can be compared without analog readout noise.

Hardware phases also permit `pwm-backend=torch` with `pool-backend=hardware` for the physical pooling control. It sends the deterministic converted hidden code through the physical LIF pool and frozen torch readout, while the manifest keeps this result separate from execution with two Hagen affine layers. Potential-domain pooling still requires Hagen. A development control may use an explicit observation deadline without a margin artifact; formal Hagen execution still requires a margin selected from calibration data.

### Hagen affine fidelity outcome

The 128-sample, eight-trial physical comparison shows that systematic affine error exceeds repeated-trial variation in both Hagen stages.

For hidden UInt5, the trial mean MAE was 1.23 codes, bias was -0.90 codes, RMSE was 2.07 codes, and repeated-trial standard deviation was 0.42 codes. The frozen integer reference reached 92.97% accuracy on this test subset, while the physical hidden tensor followed by the frozen torch readout averaged 67.87% across trials.

With ideal hidden UInt5 supplied directly to the physical output affine, the trial mean Int8 MAE was 15.23, bias was 3.70, RMSE was 21.98, and repeated-trial standard deviation was 2.10. Physical argmax agreed with the ideal output for 72.07% of individual trials and 82.81% after trial mean. Accuracy was 71.88% across trials and 82.03% after logit averaging.

The first raw affine retained high correlation with its integer accumulator but used a much smaller physical scale and channel-dependent offsets. These results identify first-affine transfer error as the largest measured bottleneck, with an additional output-affine contribution; neither is explained primarily by repeated-trial variation.

The fidelity probe can therefore fit one gain and offset per channel without labels. The first sample block is used only for calibration, and all reported correction errors and accuracies use the disjoint evaluation block. The first Hagen stage is corrected at its raw affine output before the frozen UInt5 adapter; correction after UInt5 quantization is retained only as a diagnostic control. The physical output logits receive an independent channelwise correction.

On the 64-sample evaluation block, raw first-affine correction reduced hidden UInt5 mean absolute error from 1.223 to 0.258 codes. Frozen torch-readout accuracy increased from 60.16% to 75.00% across trials and from 57.81% to 79.69% after averaging logits; the ideal integer accuracy on this block was 90.63%. Correcting the already quantized hidden code reached only 67.97% and 65.63%, confirming that correction belongs before the UInt5 boundary.

Output-affine correction reduced Int8 mean absolute error from 14.94 to 7.09, but accuracy changed from 71.48% to 71.29% across trials and from 79.69% to 78.13% after averaging logits. Channelwise ordinary least squares therefore improves numeric fidelity in both stages but provides an accuracy recovery only at the first affine in this split.

### Network pool placement

Dedicated mapping preserves persistent physical identity, while time-multiplexed mapping deliberately reuses a small pool and is reported as a different hardware method.

[[utils/hardware/brainscales2/toy_pooling.py#resolve_grouped_physical_coordinates]] allocates `30 * M` unique neurons for the 30-hidden-unit models. `local-pool` keeps every logical pool inside one quadrant while distributing pools across quadrants; `cross-quadrant` distributes every pool's replicas across quadrants.

At `M=16`, dedicated placement uses 480 of 512 neuron circuits. The 128-hidden-unit MNIST model therefore requires time multiplexing, and a local pool is repeated in the coordinate result to make reuse explicit.

[[utils/hardware/brainscales2/toy_pooling.py#GroupedHardwarePoolBackend]] constructs one grouped-broadcast `Synapse -> LIF` graph for dedicated mapping. Each logical source expands into the operating point's simultaneous fan-in lanes, which project only to that source's replica block; the network path never substitutes the primitive experiment's unreliable independent-input routing condition.

Full hardware evaluation slices samples and caps `pool_size * samples` at 128 replica-samples, so M=8 and M=16 use 16 and 8 samples. Every chunk runs in a disposable child process to release native hxtorch memory under the 2 GB EBRAINS limit, then [[utils/hardware/brainscales2/toy_pooling.py#concatenate_toy_pool_results]] restores order and provenance.

Timing calibration is acquired once per physical condition in disposable four-trial workers. Their raw events are concatenated before offset estimation across all 32 UInt5 codes, and every inference chunk reuses the same checksummed calibration; calibration and inference batches never coexist in one M=16 graph. Dedicated calibration gives each logical source a deterministic permutation of the 32 codes so simultaneous input times vary, then restores canonical code order before decoder calibration.

Before formal evaluation, [[scripts/evaluation/brainscales2_toy_hil.py#margin_calibration_phase]] measures unlabeled calibration activations at $M=1$ with a 100 microsecond diagnostic deadline. Code zero is excluded; for each 1 microsecond candidate extension from 0 to 40 microseconds, it bootstraps trials and samples and computes the upper confidence bound for the sample-level event that any positive hidden unit misses. The smallest common margin whose bound is at most 5% in both placements is selected, while nonfires at the diagnostic deadline form a structural floor and block the run when no candidate passes.

The selected margin extends only the observation deadline: the TTFS input window, UInt5 bounds, weights, and activation values are unchanged. Calibration acquires each declared pool size once at the diagnostic deadline, then censors the same raw events at every shorter candidate. Formal evaluation interleaves the selected margin and a zero-margin control over identical cached hidden inputs, every pool size, and both placements; the calibration context binds checkpoints, both calibration files, chip operating parameters, the selection pool size, and the time grid before reuse.

Digital synaptic weights can be calibrated for each physical neuron before timing calibration using [[scripts/evaluation/brainscales2_neuron_weights.py#main]]. The sweep uses the actual grouped graph and all 32 input codes without task labels. Separate validation trials test both varied and simultaneous input times, including quiet controls. Each candidate measurement runs in a disposable process with a timeout.

The default uses four input lanes per logical source. On the tested 30-neuron graph, six lanes could not be routed. Sequential diagnostic inputs exercise three code values per source to compare delivery with simultaneous traffic without changing the graph.

[[utils/hardware/brainscales2/neuron_weights.py#load_neuron_weights]] binds the selected weights to the chip, analog calibration checksum, input timing, fan-in, pool size, and exact physical coordinates. Failed validation prevents inference. The optional `--neuron-weight-calibration` file is checksummed in worker configurations and deadline calibration context. It changes the input synapses driving the LIF replicas; trained ANN weights remain fixed.

After temporal decoding, the physical Hagen readout also slices the flattened trial-sample row axis before each PWM call. When any all-miss position exists, original and oracle-repaired rows are concatenated and tagged as separate segments; the logits retain row order, and every chunk records calibration, chip, shape, and elapsed time.

Formal multi-condition runs materialize each required physical Hagen hidden tensor once, then execute every placement and pool size in a fresh child process. Completed worker directories are resumable, and the parent rebuilds the combined artifact so process isolation does not change paired inputs or the result schema.

Each child process is retried only up to a configured bound with increasing backoff. A no-output watchdog terminates the isolated process group when a native RPC call does not return. Retries restart the condition; only complete artifacts are reused, and attempt logs and status enter the manifest.

### Physical threshold calibration

Single input spikes use an explicitly calibrated analog threshold and a fixed validated placement; requested module parameters never replace physical calibration.

[[scripts/evaluation/brainscales2_thresholds.py#run_threshold_experiment]] creates a new base calibration, repeats CADC calibration, and refines leak, reset and threshold. Candidate copies preserve time constants and capacitance. Development observations select a common threshold without task labels; independent validation cannot select another candidate on failure.

The target grid is 125 down to 85 in steps of five, with unit steps between a passing candidate and the preceding failure. Each candidate records actual physical parameter values, native calibration, portable binary checksum and chip identifier. Initial calibration success does not establish successful threshold refinement: raw delivery validation is mandatory.

[[utils/hardware/brainscales2/thresholds.py#allocate_validated_coordinates]] may replace unreliable physical circuits while preserving all 30 logical neurons. Both placements must support 30 pools of 16; smaller pools reuse replica prefixes. Local placement requires enough complete pools within quadrants, and cross-quadrant placement requires 120 valid circuits per quadrant. Excluded coordinates remain in the report.

Development uses eight trials per code; fixed selection is validated using 64 trials per code and 512 quiet windows. Each physical neuron requires 99% single-spike delivery overall, at least 95% for each input time, and at most 1% quiet, premature or multiple-spike events. Wilson intervals accompany the rates but are not equated with the observed-rate acceptance cutoffs. Separate mixed and simultaneous inputs test the actual grouped graph at every pool size.

The default notebook executes physical threshold calibration before fresh deadline calibration, smoke and full evaluation. Input fan-in is one and synaptic weight is 63. The optional digital weight calibration is not combined with this selection. A separate two-by-two threshold and fan-in diagnostic retains the old four-input condition without changing the primary experiment.

[[utils/hardware/brainscales2/thresholds.py#selected_coordinates]] verifies the chip, calibration checksum and operating point before every physical execution. Deadline extensions are allowed only through the subsequent margin experiment; changed placement or calibration invalidates its context.

Full aggregation reads one condition at a time and streams prediction rows. Its small `intermediates.pt` index points to the existing condition archives; [[utils/hardware/brainscales2/toy_artifacts.py#iter_intermediate_conditions]] reads this format and the previous monolithic format. Raw events are not discarded to meet the 2 GB session limit.

The selected-threshold run also writes `estimator_controls.csv`: mean, raw-max and analytic-corrected-max decode identical physical events and use the frozen torch readout. These are explicitly readout controls, not additional physical Hagen measurements. The primary accuracy remains the physical Hagen result. Threshold workers record elapsed time and their maximum resident memory.

### Gain diagnostic

A diagnostic on sixteen neurons compares gain 500 with a fully recalibrated gain 700 using identical coordinates, input counts and threshold targets, without changing network acceptance.

[[scripts/evaluation/brainscales2_gain_diagnostic.py#run_diagnostic]] records paired quiet and stimulated CADC traces at 15 microseconds and raw delivery for all 32 input codes. It compares fan-in one and four at thresholds 125 and 100. Traces containing spikes are flagged because reset changes measured PSP amplitude. Every condition also runs with CADC recording disabled. Full calibration is mandatory when gain changes; potential refinement rejects a different gain. Separate workers and retained raw artifacts respect the 2 GB session limit. This diagnostic does not select a network operating point or start full inference.

### Local mock and replay evidence

Synthetic and artifact-replay backends validate accuracy propagation before hardware use but are not promoted to new physical evidence.

[[utils/hardware/brainscales2/toy_pooling.py#MockToyPoolBackend]] generates coordinate-static, trial-shared, replica-local, and missing-event terms. Calibration events estimate response delay and persistent offsets before inference events are decoded.

[[utils/hardware/brainscales2/toy_pooling.py#ReplayToyPoolBackend]] splits the allowlisted primitive artifact by trial, estimates timing calibration only from the first half, and samples residuals and misses only from the held-out half. It reuses the primitive distribution across logical units and therefore declares `rough-model-only` scope.

### Network result contract

Network artifacts join float, ideal-converted, and physical predictions with complete intermediate tensors and bounded human-readable event extracts.

[[utils/hardware/brainscales2/toy_artifacts.py#write_toy_artifacts]] writes conversion and run manifests, prediction variants, accuracy and NLL drops, paired recovery intervals, same-run zero-versus-selected-margin comparisons, support-stratified miss metrics, per-code UInt5 error, readout ablations, raw timing tensors, and figures. `intermediates.pt` remains lossless; `events.csv` records a deterministic sample/trial subset.

Accepted physical runs use a per-run Git allowlist. The committed bundle keeps checkpoints, calibration, manifests, metrics, predictions, figures, and compressed event extracts; oversized lossless tensors and worker chunks remain external and are represented by size and SHA-256 in `artifact_inventory.json`.

[[scripts/evaluation/brainscales2_toy_hil.py#main]] separates train, convert, local evaluation, Hagen probe, deadline-margin calibration, hardware smoke, and full hardware phases. MNIST hardware evaluation defaults to a 128-sample runtime benchmark until the caller explicitly sets a formal sample count.

The canonical EBRAINS notebook defaults to resuming the accepted seed-0 checkpoint and explicit calibration files at deadline-margin calibration. A fresh mode instead trains, converts, and probes the Hagen shift before the same smoke-gated full evaluation.

It configures the shared client from a writable `/tmp` checkout, isolates Hagen initialization and every CLI phase behind process-group watchdogs, and records failures in `pipeline_status.json`. Full evaluation requires same-run preflight, margin calibration, and hardware smoke.

## Toy ANN2SNN Verification

These test specifications protect the conversion and network-level hardware boundary without requiring hxtorch locally.

### Neuron synaptic weight calibration

Digital weight calibration must preserve grouped connectivity, reject invalid digital values, use separate validation observations, and reject a failed or mismatched physical calibration.

### Gain diagnostic checks

Tests verify simultaneous input counts, silent control windows, fixed quadrant coverage, rejection of gain changes during potential refinement, and raw event summaries for misses and repeated spikes.

### Physical threshold selection

Tests cover separate candidate copies, unchanged time constants, actual threshold updates, code coverage, independent seeds, spare circuit capacity, nested coordinates, and calibration checksum and chip mismatch rejection.

[[scripts/verification/verify_brainscales2_thresholds.py#verify_threshold_selection]] verifies these contracts without importing EBRAINS dependencies.

### Bounded artifact aggregation

Tests verify that both monolithic and indexed condition tensors can be read without changing events. Full aggregation reloads one condition at a time and writes prediction rows directly to disk.

[[scripts/verification/verify_brainscales2_thresholds.py#verify_shard_reader]] compares raw values across both formats; the existing condition aggregation check covers paired metrics.

Tests cover zero and maximum weights, source isolation, quiet activity, missing and multiple spikes, smallest passing candidate selection, reproducible input schedules, chip and coordinate identity, checksum changes, and failed validation.

### Host-mediated implicit ReLU boundary

The default hidden boundary must lower raw PWM values through the declared $V_{lb}=0$ Potential range without calling `ConvertingReLU`, retain UInt5 upper saturation, and label the result as host-mediated rather than continuous on-chip activation.

### Physical pooling with torch readout

Physical pooling may use deterministic converted hidden codes and the frozen torch readout, but it must reject mock Hagen execution and Hagen potential averaging in this control.

### Hagen shift probe consolidation

The implicit host-boundary shift sweep must reuse one physical preactivation, while native converting-ReLU candidates remain distinct hardware executions.

A model narrower than 128 input lanes must issue one unique shape probe rather than executing its architecture width twice. Candidate scores and metadata must disclose shared physical observations.

### Hagen affine fidelity

The Hagen probe must compare repeated physical affine outputs with the exact frozen integer reference on held-out unlabeled inputs.

First-affine raw output, hidden UInt5 output, and output Int8 must remain separate. The output affine receives ideal hidden codes, preventing TTFS pooling error from entering its fidelity measurement. Reports distinguish trial mean error from repeated-trial variation and retain per-channel values.

Channelwise affine correction must fit only the calibration sample block, without labels, and report all error and accuracy changes on the disjoint evaluation block. The primary hidden correction maps raw first-affine output back to the integer accumulator before applying the frozen UInt5 adapter; direct hidden-code correction is secondary.

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

### Varied timing calibration inputs

Every dedicated logical source must observe every UInt5 code once per calibration trial while simultaneous source times vary; time-multiplexed calibration retains identical code times.

### Pool-size-aware hardware chunk cap

The physical pool chunk must reduce with pool size so every grouped graph stays below its replica-sample memory budget.

A requested sample chunk remains an upper bound; the effective size is `min(requested, budget // pool_size)` with a minimum of one sample.

### Pool chunk process isolation

Every full-run hardware chunk must execute in a disposable child, persist its result and attempt status, and be reusable after a later chunk or outer condition is killed.

### Hagen output row chunking

Physical Hagen readout chunks must preserve flattened trial-sample row order, bound each PWM call, retain per-chunk provenance, and share one hxtorch initialization until the row group completes.

### Hagen first-layer row chunking

Physical Hagen hidden inference must split sample rows before each PWM call, preserve sample order, and retain per-chunk provenance; all chunks share one hxtorch initialization before one final release.

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

`analytic-corrected-max` estimates the delivered residual scale and codewise deadline tail only from calibration events, then adds the finite-$M$ conditional earliest-time offset. `empirical-corrected-max` inverts the codewise calibration response without labels.

The empirical corrected maximum groups UInt5 codes that share the same quantized input time. It selects the nearest measured response group, then resolves physically indistinguishable codes with the unlabeled calibration activation prior. The activation prior is recorded in the timing calibration and its checksum invalidates stale worker artifacts.

All four estimators consume the same raw-event tensor, retain all-miss-to-zero semantics, and run through the same frozen readout. Replay results remain rough model-selection evidence; hardware acceptance requires an independent calibration acquisition and evaluation events.

The artifact fields `logit_mean_accuracy` and `logit_mean_nll` report results after averaging logits across repeated physical trials. This secondary control does not replace the primary accuracy averaged over individual trial predictions when evaluating neuron pooling.

### Deadline margin selection

Margin calibration must exclude code zero and choose the smallest hardware-grid deadline extension whose hierarchical-bootstrap miss upper bound after pooling passes in every placement for the declared selection pool size.

Each logical pool is delivered when any physical replica fires by the candidate deadline. M=1 and M=16 diagnostic curves reuse their respective raw events acquired at the maximum observation deadline, while only the declared selection pool size controls the formal margin. All candidates reuse the same bootstrap draws so the nested miss curve remains monotone. A persistent diagnostic-deadline floor must produce no selection rather than being hidden by a larger margin.

### Deadline margin provenance

A selected margin may be reused only with the exact unlabeled model, calibration files, timing grid, physical operating point, selection pool size, and complete 32-code timing correction that produced it.

Changed checkpoints, calibration checksums, neuron parameters, or corrected-max tables must invalidate reuse. The margin changes only the deadline and must not modify the encoded activation interval.

### Deadline margin diagnostic outcome

At the tested operating point, no physical event arrived after the 60 us base deadline, so extending the observation deadline to 140 us cannot recover missed activations or accuracy.

The quick calibration artifact `20260916T025150Z_deadline_margin_M1_M16_quick` acquired $M=1$ and $M=16$ events at 140 us and applied the 60, 80, 100, 120, and 140 us deadlines to the same raw events. The maximum finite time was 29.82 us, with zero events between 60 and 140 us. For $M=16$, the positive UInt5 logical all-miss rate was 1.27% for `local-pool` and 0.106% for `cross-quadrant`; samples containing any positive logical all-miss position were 17.19% and 1.56%, respectively. The earlier 150-sample $M=16$ network run likewise had no event after 60 us and a maximum of 30.48 us.

The deadline margin remains a diagnostic and safety parameter, but it is not a robustness mechanism for this chip, calibration, and operating point. Placement and physical event delivery reliability are the relevant controls here. A changed threshold, gain, time constant, routing, or calibration requires a new margin diagnostic before reusing this conclusion.

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

The single canonical notebook remains a thin Python 3.11 launcher with explicit stage flags, bounded hardware calls, and MNIST sample limits.

Training must precede hardware allocation, the probe-selected shift must feed smoke, and formal stages must be gated by a passing same-run smoke artifact.

The default resume path reuses the accepted checkpoint and explicit Hagen calibration, skips redundant training and shift probing, then creates and validates a spiking threshold calibration before margin selection. Setting the resume source to `None` restores the fresh train-and-probe path.

Before either hardware path, the shared-client handshake has bounded retries, while Hagen initialization runs in a disposable process with a fixed watchdog and always attempts release. Every CLI phase also has a process-group timeout, stage state is recorded, and full Yin-Yang evaluation still requires the same run's smoke gate. The notebook contains no credentials.

## Independent Primitive Noise Characterization

The hardware characterization measures five primitive transfer functions independently and does not use logical neuron pooling.

### Physical acquisition boundary

Sixteen quadrant-balanced physical circuits expose repeated-trial variation within each circuit and fixed-pattern parameter differences between circuits.

The formal 16-circuit placement uses coordinates selected by qualification on the recorded chip: `(3, 11, 13, 1, 134, 141, 128, 130, 273, 261, 262, 274, 390, 386, 388, 392)`. The list is valid only with the recorded chip and calibration provenance; other chips require a new coordinate qualification.

Coordinate qualification permits missed spikes because they are part of the measured noise distribution. Every fitted code must retain calibration and validation observations. First spike fitting uses the first recorded spike whenever at least one spike is present, while the operating point requires a rate of trials with multiple spikes no greater than 0.1%. The hardware sweep preserves four selected circuits per quadrant.

The $\phi_{\mathrm{NP}}$ acquisition first measures configured initial membrane states under a synchronized constant-current ramp. Only a passing static stage enables synaptic precharge, CADC confirmation, and a second ramp acquisition. A static-only result is diagnostic and cannot validate the primitive.

The static ramp acquisition may use either one shared 32-entry reset lookup or one 32-entry lookup for each physical neuron. Every lookup is fitted only from calibration data; validation data cannot influence it. The selected reset codes and source artifact are recorded.

Dynamic precharge records spikes through the event channel and membrane potential through the dense recorder. It avoids the default analog recorder because the installed PyNN backend limits that recorder to two locations.

The dynamic operating point records the number of coincident precharge input channels. Increasing this multiplicity enlarges the measured potential span without changing the UInt5 code or synaptic weight range.

Every input code and physical neuron must retain a CADC precharge observation. Its rank statistic remains diagnostic because sparse absolute CADC samples can cross the signed readout boundary; dynamic acceptance uses the independently recorded first-spike transfer.

The hardware adapter resolves input spike sources and static synapses from the namespaces exported by the installed backend instead of assuming that every component is also exported at module level.

Each input point and physical circuit retains one measured precharge diagnostic from the first acquisition window. Unmeasured trial entries remain nonfinite and are never copied from the recorded diagnostic.

The $\phi_{\mathrm{NL}}$ acquisition reuses the validated synaptic precharge path and measures first-spike time after a fixed exponential synaptic input. The zero code is outside its positive fit domain.

The nonlinear primitive realizes its fixed exponential drive with eight coincident physical input channels at weight 63. The input multiplicity is recorded and remains constant across potential codes.

The $\psi_{\mathrm{Int}}$ acquisition measures bias-free Hagen PWM integration over every UInt5 duration and signed drives $\{-2,-1,1,2\}$. `Linear.avg` remains one, so hardware neurons are not averaged.

The Hagen operating point uses 1024 sends and maps 128 candidate outputs. Calibration residual normalized by fitted span selects 16 output indices; validation observations are not used for selection, and selected outputs are not averaged.

Formal Hagen collection partitions repetitions into bounded input chunks inside one hardware session. Candidate observations are concatenated in trial order before calibration selects output indices, and chunk boundaries are recorded.

The public Hagen `Linear` API does not expose an atomic neuron placement constraint. Integration artifacts therefore identify each circuit by its stable output index within the mapped layer and label that coordinate meaning explicitly; they do not reuse spiking neuron coordinates.

The $\psi_{\mathrm{NE}}$ acquisition records a non-spiking membrane CADC value at a fixed observation time after sweeping the input-spike time. Paired quiet trials provide the baseline; any output spike rejects the operating point.

The collector reads the membrane at $40\,\mu\mathrm{s}$ because the measured response peaked about $10\,\mu\mathrm{s}$ after the input event. This keeps every $5$--$25\,\mu\mathrm{s}$ input on the decay branch before the $60\,\mu\mathrm{s}$ deadline. The runner exposes this observation time so the operating point remains explicit.

The $\psi_{\mathrm{NE}}$ operating point uses two coincident input channels. Paired quiet observations are subtracted before fitting, while their raw values remain stored so saturation and baseline variation stay observable.

Formal $\psi_{\mathrm{NE}}$ collection partitions repetitions into disposable child processes because repeated hxtorch graphs exceed the notebook memory limit. Each child initializes and releases hardware, writes a fingerprinted retry cache, and returns at most four repetitions. The parent concatenates result tensors in trial order and records every chunk boundary.

The $\psi_{\mathrm{ED}}$ acquisition measures the exponential-difference response with the BrainScaleS-2 causal correlation sensor. It does not substitute a composed host calculation for the physical sensor response.

For each time difference $\delta\in[-10,10]$ microseconds, the pre-event separation is $15\,\mu\mathrm{s}-\delta$. The target event is scheduled at $30\,\mu\mathrm{s}$, the causal correlation code is read at $58\,\mu\mathrm{s}$, and target events after the $60\,\mu\mathrm{s}$ deadline are misses.

Alternating quiet and stimulated periods measure the difference between paired correlation codes. A one-millisecond guard lets the plasticity processor read and reset the sensor between periods without changing the event deadline.

The plastic synapse keeps weight zero so its routed pre-event reaches the correlation sensor without independently firing the target neuron. Eight weight-63 trigger channels produce the target event used by the sensor.

The $\psi_{\mathrm{ED}}$ worker preserves the target neuron's first-spike timestamp and total spike count. It applies the explicit correlation calibration, records one warmup period, and splits formal collection into bounded child processes with fingerprinted retry caches.

Each trial uses a seeded permutation of the time-difference inputs, then restores the canonical input order before analysis. This prevents a fixed input schedule from being confounded with correlation-sensor drift across acquisition time.

The installed `pynn_brainscales.brainscales2` timed constant-current playback owns the NP ramp. If that capability is unavailable, the runner stops instead of replacing the ramp with a spike train or a host-mediated Hagen transform.

PyNN schedules and reports wall-clock milliseconds. The acquisition converts the 5--25 us physical input window to milliseconds before playback and converts recorded timestamps back to seconds exactly once.

Each measured PyNN trial is preceded by a hardware reset pulse. One warmup trial is discarded so startup state does not enter the transfer fit, while low-level writes remain limited to backend reset control and preserve PyNN spike routing.

Formal PyNN collection partitions repeated trials for each input code into bounded acquisitions. Every acquisition discards its own warmup trial, and results are concatenated in trial order with recorded chunk boundaries.

Formal PyNN collection isolates at most 32 repetitions of one input code in a child process. Each child retains hardware acquisitions of eight repetitions and returns raw timestamps and spike counts. Process exit bounds native memory use within the notebook memory limit.

Each completed child process result is written to a fingerprinted cache before the next hardware call. A retry reuses only entries whose configuration, input code, and trial bounds match exactly.

PyNN acquisition uses a deterministic seeded trial assignment that distributes calibration and validation observations across acquisition time. Raw tensors still store the complete calibration split before the validation split, and metadata records the original acquisition indices.

### Fit and result boundary

Each circuit is fitted on 128 calibration repetitions and evaluated with fixed parameters on 128 held-out repetitions.

Per-circuit temporal sigma comes from residuals around that circuit's calibration transfer. Circuit offset, gain, slope, and effective time-constant differences remain separate fixed-pattern statistics. No circuit outputs are averaged.

Normalized root mean square error evaluates the mean at each validation input against the transfer fitted on the calibration split. Temporal sigma remains the standard deviation of individual calibration residuals, so variation across repeated trials is reported without being counted twice as transfer error.

The NP fit is linear in potential code. The NL calibration jointly searches $V_{\mathrm{lb}}$ and fits time linearly against $\log(V-V_{\mathrm{lb}})$, then holds both fixed for validation. The integration fit is linear in signed duration. The NE and ED fits search their effective time constants while fitting baseline and response scale.

The $\phi_{\mathrm{NP}}$ monotonicity gate uses Spearman rank correlation between input code and decreasing validation mean first spike time. The adjacent pair ordering fraction remains a diagnostic because temporal jitter can reverse neighboring measured means without breaking the global transfer order.

Missed spikes remain missing values and are not replaced by the deadline. Every fitted input and circuit must retain calibration and validation observations, and the aggregate deadline-miss rate must not exceed 1%. First spike fitting retains trials with additional spikes, reports their rate separately, and requires that rate to be no greater than 0.1%. CADC and Int8 saturation are separate flags. Monotonicity is an acceptance gate only for $\phi_{\mathrm{NP}}$, whose contract requires ordered conversion from potential to time. Other primitives report monotonicity as a diagnostic and apply their declared transfer, saturation, parameter drift, observation-availability, and output spike gates.

An observation with too few usable samples for one physical circuit remains writable as raw data. Its fit is marked unavailable and validation fails instead of aborting artifact creation.

[[scripts/evaluation/brainscales2_primitive_noise.py#run]] writes checksum-indexed raw split chunks, per-stage transfer, moments, and device statistics, figures, and one `primitive_noise_calibration.json`. Each `moments.csv` records calibration and validation sample count, mean, variance, miss rate, rate of trials containing more than one spike, and saturation rate for every input and circuit. The combined record validates only when NP passes both stages and the other four primitives pass held-out validation.

The resulting distributions are independent primitive marginals for later sensitivity analysis. They do not represent the joint error distribution of a composed BSS-2 circuit and do not include Transformer forward evaluation.

### Encoder Operating Point Search

The search minimizes the calibration timing noise ratio without changing the primitive equations or pooling physical outputs.

The search changes only the raw constant current code, threshold code, ramp stop time, precharge input count and weight, and exponential synaptic current input count and weight. The potential code, first spike readout, and fitted $\phi_{\mathrm{NP}}$ and $\phi_{\mathrm{NL}}$ equations remain unchanged.

For static encoder acquisition, the reset associated with each code remains installed for the full trial. The refractory period covering the full window suppresses later spikes, while retaining the reset value avoids rewriting an analog parameter after the initial membrane state is loaded.

All PyNN encoder acquisitions use the official Calix refractory period calculator with a target equal to the full repeated trial window. The observation deadline is followed by a quiet interval of the same duration. Calix resolves the backend clock scales and the selected circuit counters, so a circuit cannot emit a second spike within one trial; the next trial begins only when the configured refractory period has elapsed.

The search compatible with the operator assumptions keeps the synchronized ramp stop at 25 microseconds and the observation deadline at 60 microseconds. Longer ramps are not eligible merely because increasing the signal span can lower the normalized ratio.

After the coarse screen, separate refinements vary threshold and precharge input count around each encoder's selected current and ramp duration. The $\phi_{\mathrm{NL}}$ refinement additionally varies the input count and weight of its exponential synaptic current. The result records the best held out timing noise ratio for each encoder.

Coarse and refinement screens use three representative potential codes with 32 repetitions split equally between calibration and held out data. Only the selected configuration for each encoder receives the full 32 code grid with 128 calibration and 128 held out repetitions.

The calibration screen may record spike times without membrane potential observations. This mode is restricted to representative potential codes and cannot satisfy a declared $r_t$ threshold.

The physical circuit screen runs 64 circuits per hardware graph because a 512 circuit graph changes input delivery and produces widespread deadline misses. Eight disjoint batches cover all 512 coordinates. The globally selected circuit minimizes the calibration timing noise ratio across the selections from each batch, after which the coordinate is frozen for confirmation over all 32 potential codes with held out repetitions.

Transfer prerequisites are evaluated for each physical circuit. A circuit that fails remains in the diagnostic statistics but cannot invalidate a different circuit selected from calibration data.

The calibration-selected physical circuit is frozen before confirmation over all 32 potential codes. Confirmation restores membrane potential observations and uses an independent held out split before reporting whether an $r_t$ threshold is reached.

The representative code screen requires observed points, the declared transfer direction, bounded miss and repeated spike rates, and no saturation. Rank and normalized error remain diagnostics during screening; the full code confirmation must pass every original transfer gate before a target is reported as reached.

Each circuit's conditional timing deviation is divided by the $\phi_{\mathrm{NP}}$ signal span fitted on calibration repetitions. The same frozen span normalizes both encoders, so a candidate cannot improve its score only by redefining the denominator.

Candidates are ranked only with calibration repetitions. For each candidate, the physical circuit with the minimum calibration timing noise ratio is selected and its coordinate is frozen. Held out repetitions from that same circuit confirm whether the timing noise ratio reaches 0.001, 0.0001, or 0.00003. The median across measured circuits remains a fixed-pattern robustness diagnostic rather than the optimization objective.

Each encoder summary reports the validation timing noise ratio only if all required stages pass validation. $\phi_{\mathrm{NL}}$ requires static and dynamic $\phi_{\mathrm{NP}}$ plus its own transfer; $\phi_{\mathrm{NP}}$ requires its static and dynamic stages.

[[utils/hardware/brainscales2/primitive_optimization.py#score_encoder_operating_point]] scores one candidate, while [[scripts/evaluation/brainscales2_primitive_noise.py#optimize_encoder_operating_point]] owns enumeration and artifacts. The command writes a fixed search manifest, one standard primitive artifact per candidate, a result table, and a selected configuration. Finished candidates are reused only when the search manifest and candidate identity match.

### Provisional encoder operating point result

The current result selects one physical circuit on calibration repetitions and reports its held out timing noise ratio without refitting; it remains a representative code screen pending full code confirmation.

For $\phi_{\mathrm{NP}}$, calibration selects current code 1022, threshold code 550, ramp stop $25\,\mu\mathrm{s}$, precharge fan in 4 with weight 56, and physical coordinate 176. Its calibration timing noise ratio is 0.02132 and its held out ratio is 0.02387.

For $\phi_{\mathrm{NL}}$, calibration selects current code 1022, threshold code 550, ramp stop $25\,\mu\mathrm{s}$, precharge fan in 4 with weight 63, and physical coordinate 168. Its calibration timing noise ratio is 0.01062 and its held out ratio is 0.01152.

Both selected circuits pass their per-circuit transfer gates on the held out split. Selecting the minimum held out row after observation gives lower values 0.02178 and 0.01040, but those values are post hoc lower bounds and are not deployment estimates.

The target timing noise ratio 0.001 is not reached. The selected $\phi_{\mathrm{NP}}$ and $\phi_{\mathrm{NL}}$ ratios require 23.9-fold and 11.5-fold reductions, respectively. Extending the ramp stop to $850\,\mu\mathrm{s}$ did not help because the fitted signal span did not grow with runtime and conditional timing variation increased.

Full code confirmation remains pending because both the known operating point and the official single-neuron PyNN example timed out at the hardware run after a complete Jupyter server restart. The frozen candidate tables and simulator comparison are stored under `artifacts/brainscales2-primitives/20260920T223036Z_rt_search_summary/`.

### Formal physical result for five primitives

The formal run stores 128 calibration and 128 validation repetitions for each primitive without pooling outputs across physical circuits.

The consolidated result is `artifacts/brainscales2-primitives/20260918T020500Z_five_primitive_moments`. Its `moments.csv` contains 7,136 split, input, and circuit rows. The source raw files remain on the hardware server and their checksums are fixed in the manifest.

The aggregate deadline miss rates were 0.0046% for $\phi_{\mathrm{NP}}$, 0.0299% for $\phi_{\mathrm{NL}}$, and 0.0089% for $\psi_{\mathrm{ED}}$. Their rates of trials containing more than one spike were 0%, 0%, and 0.03995%, respectively, so first spike readout satisfied the 1% miss and 0.1% multiple spike operating limits. $\psi_{\mathrm{Int}}$ had no Int8 saturation, while $\psi_{\mathrm{NE}}$ had complete membrane readout delivery and no membrane readout saturation.

Distribution acquisition passed for all five primitives. This is separate from transfer function validation: $\phi_{\mathrm{NP}}$ and $\psi_{\mathrm{Int}}$ passed their validation gates, while $\phi_{\mathrm{NL}}$, $\psi_{\mathrm{NE}}$, and $\psi_{\mathrm{ED}}$ retain physical marginal distributions but are not promoted as validated transfer implementations.

## Independent Primitive Noise Verification

These tests protect calibration isolation, event semantics, artifact integrity, and the boundary between temporal and fixed-pattern variation without importing EBRAINS packages.

### Synthetic transfer recovery

Synthetic observations must recover the five transfer parameters and nonzero within-device noise scales within declared tolerances.

### Held-out validation isolation

Changing only held-out observations must not change calibration parameters, while held-out fit diagnostics and normalized error must respond.

### Temporal and fixed-pattern separation

Device offsets with no repeated-trial noise must produce fixed-pattern spread without inflating within-device temporal sigma.

### Miss and saturation semantics

Missing spikes remain non-finite, while first spike values remain usable when additional spikes occur.

Both data splits must retain observations at every fitted point, and the aggregate rate of trials with additional spikes must not exceed 0.1%. Any premature membrane-output spike fails its gate, while saturation remains distinct from missing delivery.

### Rank monotonicity gate

Global rank order determines whether the validation transfer retains the declared direction.

The adjacent pair ordering fraction remains diagnostic and may fall below the gate when temporal jitter reverses only neighboring measured means.

### Segmented PyNN recording decode

Segmented PyNN playback can return one spike train per physical neuron for each segment. The decoder groups timestamps by `source_index` before computing each trial's first spike time and count.

### Dynamic recording selection

The recording setup must keep spike recording independent from dense membrane recording and must select the dense recording device explicitly for every dynamic precharge acquisition.

### Segmented dense recording decode

Dense recording chunks must be grouped by source identifier before the sample nearest the requested diagnostic time is selected for each physical circuit.

### Sparse precharge evidence

Validation requires at least one finite precharge diagnostic for every input point and physical circuit. The precharge rank statistic remains a reported diagnostic rather than a transfer gate.

### Installed component resolution

Component resolution must accept the installed nested namespaces and fail clearly when either the spike source or static synapse type is absent.

### Correlation recording decode

The correlation decoder must preserve one causal sensor code per recorded period and physical circuit.

It rejects missing periods, unexpected plastic rows, and a circuit dimension that differs from the fixed placement.

### Correlation worker boundary

The correlation backend must concatenate bounded child-process results before applying the seeded calibration and validation split.

It preserves first-spike timestamps beyond the deadline as raw evidence, masks their primitive outputs as misses, and retains total spike counts without averaging circuits.

### Nonlinear drive configuration

The nonlinear drive configuration must reject a nonpositive input multiplicity and preserve its resolved value in the hardware metadata.

Verification also checks positive PyNN acquisition and process sizes, manifest preservation, deterministic trial assignment, and child process dispatch.

### Static reset separation

Verification checks scalar reset values and values for individual circuits. It requires static acquisition to retain the reset value associated with the input code while the ramp is active.

Verification checks selection of the three resolved refractory parameters for scalar circuit coordinates, rejects incomplete settings for the 512 circuits, confirms that the target covers the full repeated trial window, and confirms that the quiet trial interval and first spike configuration are installed before acquisition.

### Exponential response observation time

The runner must expose and preserve the $\psi_{\mathrm{NE}}$ membrane observation time, input multiplicity, and positive chunk size. Verification checks event construction with two input channels and paired baseline subtraction.

### Hagen output qualification

Verification checks that Hagen output selection uses only calibration, returns unique indices within the candidate range, and records candidate scores. It also checks that the Hagen chunk size is positive and preserved in the manifest.

### Insufficient fit preservation

An observation with too few usable samples for one physical circuit remains writable as raw data. Its fit is marked unavailable and validation fails instead of aborting artifact creation.

### NP stage gate

A static-only NP artifact is diagnostic-only and cannot enter the validated five-primitive calibration.

An operating point screen advances when at least one physical circuit passes the current stage; a failed circuit cannot suppress later acquisition for other circuits.

### Artifact integrity

Raw chunks must round-trip with their shapes and masks, reject checksum changes, and reject duplicate or out-of-range physical coordinates.

### Validated NP placement

The default 16-circuit placement must retain the coordinate order that passed the full-code hardware sweep and must allocate four unique circuits to each quadrant.

### Encoder operating point score

The score must use calibration repetitions for selection and reserve held out repetitions for confirmation.

Representative code screening may rank a candidate that fails a shape diagnostic, but the full code confirmation must retain the strict transfer gates.

Representative code screening must never mark an $r_t$ target as reached; only confirmation over all potential codes can do so.

The selected circuit is confirmed on its own held out observations; failures of unselected circuits do not change that confirmation.

### Resumable operating point search

The search must enumerate the complete requested grid, reject invalid raw controls, and reuse only matching completed candidates.

The notebook must scan all 512 physical circuits as eight 64 circuit graphs and select across batches using calibration repetitions only.

Every run may contribute scores to each encoder stage it measured, regardless of which encoder named that search pass.

A representative code circuit scan may omit precharge membrane observations. Any use outside that screening mode is rejected, and the immutable search manifest records the omission.
