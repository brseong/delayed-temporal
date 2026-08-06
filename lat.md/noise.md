# Noise Model

The current noise subsystem is a configurable computational robustness model applied at TTFS encoding boundaries, with separate static mismatch and parameter perturbations.

## Configuration

Encoder-level jitter and drop/insertion behavior are stored in a process-global configuration object.

[[utils/transforms/noise.py#NoiseConfig]] contains the distribution scale, potential- or time-referred mode, reference scale, and independent enable flags. [[utils/transforms/noise.py#set_spike_time_noise]] updates this singleton for subsequent encoder calls.

All encoder noise is off by default. The global state makes simple sweeps convenient but requires single-threaded discipline when hooks turn noise on and off around selected modules.

## Potential-Referred Jitter

The default jitter path adds Gaussian noise to a potential before its deterministic TTFS transform.

The voltage standard deviation is `std * potential_scale`, normally `std * theta`; when no reference is configured, the input domain range is used. The perturbed potential is clamped before encoding, so the encoder Jacobian determines the resulting time error.

This creates operating-point dependence naturally: an affine encoder has constant time sensitivity, while a log encoder amplifies voltage perturbations near its lower potential floor.

## Legacy Time-Referred Jitter

The legacy mode adds Gaussian noise directly to emitted spike times and reclamps them to the declared output window.

[[utils/transforms/noise.py#_emit_spike_time_core]] exists for reproducing older sweeps. It does not preserve an unclamped time or validity flag, so an overflow-induced deadline miss cannot be distinguished from a valid latest event.

The configuration includes a noise `kind`, but the implemented paths currently support Gaussian sampling only. The stored `eval_mode` flag is not consulted by the encoder wrapper, so callers should not infer training-versus-evaluation behavior from that field alone.

## Drop and Insertion Stress Test

The hazard path maps potential margin to a per-call firing probability, then samples drop and optional insertion events.

[[utils/transforms/noise.py#_apply_escape_hazard]] uses a soft-threshold width `delta_u` to make near-floor potentials less reliable. A dropped event is assigned the latest time; an independently sampled insertion is assigned the earliest time.

This is not a single first-passage draw shared with jitter. It is best described as a potential-dependent spike dropout/insertion stress test unless the implementation is replaced by a unified escape-process sampler.

## Unified Beta Helper

A helper ties jitter scale and drop width to one dimensionless sharpness parameter but still delegates to the separate sampling paths.

[[utils/transforms/noise.py#set_unified_noise]] derives Gaussian potential jitter and hazard width from `beta`. Its `jitter_only` and `drop_only` flags are ablation masks.

Because jitter and drop still use independent random draws, the helper supplies linked magnitudes rather than a mathematically unified first-passage event. The main ViT runner under `scripts/evaluation` currently configures the independent parameters directly.

## Static Threshold Mismatch

Device mismatch is represented as a frozen Gaussian potential offset attached to each supported spiking module.

[[utils/transforms/noise.py#install_device_mismatch]] samples one offset per normalized feature, linear input feature, or convolution input channel and installs it with forward pre-hooks. Offsets are sampled once, broadcast across batches and tokens or spatial positions, and excluded from checkpoints.

This is static parameter uncertainty, not trial-to-trial temporal noise. The model keeps a scalar saturation domain, so per-neuron saturation and calibration circuitry are outside the simulation.

## Static Weight and Bias Perturbation

The ViT evaluator also supports one-time perturbation of loaded synaptic parameters outside the shared noise module.

[[scripts/evaluation/error_analysis_vit.py#apply_parameter_noise]] applies multiplicative Gaussian weight perturbation and additive bias perturbation before evaluation. This path is separate from threshold mismatch and encoder noise and should be reported as static parameter uncertainty.

## Encoder Injection Boundary

Jitter and drop/insertion are injected by a decorator on both maintained potential-to-spike transforms.

[[utils/transforms/noise.py#inject_spike_time_noise]] perturbs potential before encoding in the default mode, or spike time after encoding in legacy mode, then applies the independent hazard path. [[utils/transforms/potential_to_spike.py#neg_linear_transform]] and [[utils/transforms/potential_to_spike.py#neg_log_transform]] both carry this decorator.

Noisy effective potentials are clamped to their declared input domain, and final spike times are always clamped to their declared output domain.

## Injection Scope and Compounding

Noise is applied per encoder invocation, including encoders reused inside composite arithmetic, rather than once per high-level Transformer neuron.

Multiplication, division, softmin, LayerNorm, and activations can call encoders several times. A global noise setting therefore compounds perturbations across many internal sites and represents a strong “noise at every operation” stress test.

[[utils/transforms/noise.py#install_noise_scope]] can bracket selected modules for attribution experiments, but it mutates global state and is explicitly unsuitable for nested scopes or `DataParallel` execution.

## Interpretation Limits

Noise results establish sensitivity to the implemented perturbations, not robustness to a calibrated physical substrate.

The simulator does not currently provide a unified deadline-miss state, first-passage escape sample, temporal correlations, per-device saturation, temperature dependence, routing errors, or SPICE-level validation. The latest spike time conflates valid, clamped, and dropped outcomes.

Experiments should name the exact component, scale, injection scope, random seeds or repeats, and empirical clamp or drop rate. [[decisions#Separate Deterministic Fidelity from Noise]] defines the intended evidence boundary.
