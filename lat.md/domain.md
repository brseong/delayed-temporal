# Domain Model

The project models analog values, TTFS spike times, and their valid ranges as explicit domain objects so composed operators can track finite-window assumptions.

## Potential and Declared Bounds

A `Potential` is a tensor paired with a declared `PotentialBounds` envelope, not a simulated membrane-neuron object.

[[utils/transforms/types.py#Potential]] is the carrier used by Transformer layers. [[utils/transforms/types.py#OpenBounds]] provides a range and clamping operation, while `PotentialBounds` and `TimeBounds` distinguish voltage-like and time-like quantities at the type level.

Bounds serve three roles:

- They determine the affine or logarithmic encoding window.
- They support interval arithmetic for composed output ranges.
- They provide explicit locations for clipping diagnostics.

Although `OpenBounds` is named as an open interval, its runtime clamp includes numeric endpoints. Documentation and proofs should therefore distinguish the intended mathematical domain from the representable clamped envelope.

## Domain Propagation

The intended model-wide policy measures initial activation extrema once and then propagates conservative ranges through interval arithmetic.

For example, [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTEncoder#forward]] wraps embedding output in a `Potential`; spiking linear layers derive output bounds from input and weight intervals; residual connections add lower and upper endpoints.

Some fallback and nonlinear paths still construct bounds from observed output minima and maxima. Those paths are useful for simulation but are data-dependent and should not be presented as fixed hardware calibration without an explicit calibration protocol.

The planned removal of runtime-derived bounds across all maintained operators and model adapters is tracked in [[todo#Static Bounds for All Operators]].

## TTFS Encoding

Potential-to-spike transforms encode larger analog values as earlier events inside a declared time window.

[[utils/transforms/potential_to_spike.py#neg_linear_transform]] maps a bounded potential to a negative-linear latency. On a symmetric domain `[-theta, theta]`, [[utils/transforms/potential_to_spike.py#neg_identity_transform]] reduces to the useful identity `t = theta - V`.

[[utils/transforms/potential_to_spike.py#neg_log_transform]] maps a strictly positive potential to a logarithmic latency proportional to `log(V_max / V)`. Its time constant controls the temporal scale and its upper time bound is fixed by the ratio of declared domain endpoints.

The encoders are decorated by the global noise boundary described in [[noise#Encoder Injection Boundary]]. Even when noise is disabled, their outputs are projected into the declared time range.

## Temporal-to-Potential Decoding

Time-to-potential operators turn latency differences into exponential analog values used by division and normalization.

[[utils/transforms/spike_to_potential.py#exp_operator]] computes a bounded exponential relative to the end of a time domain. [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] composes integration, affine encoding, and an exponential stage to represent an exponential of a time difference.

[[utils/transforms/spike_to_potential.py#normalized_exp_operator]] is a numerically stabilized simulation shortcut. Its current implementation applies `exp` directly to the supplied tensor, so any claim involving its `tau_m` argument must be verified against the implementation rather than inferred from the signature.

The event-aware exponential-difference path does not evaluate this shortcut at a missed event's stored deadline. It evolves and clamps the preceding integration state, re-encodes that potential, and returns the exp-temporal reset value zero if this internal event also misses.

## Dual Operator Algebra

The project’s “dual operators” alternate potential-to-time encoders with temporal integration or exponential time-to-potential operators.

The core pattern is:

- `phi` transforms encode an analog potential as a spike time.
- `psi` operators consume spike timing and a potential or reference signal.
- Composite `f` functions reproduce dense arithmetic such as products, ratios, attention weights, and activations.

This is an algebraic simulator of the proposed operator construction. It computes tensors directly and does not instantiate a timestep-resolved circuit or transistor-level current kernel; see [[decisions#Algebraic Operators Instead of Circuit Simulation]].

## Signed Values and Dual Rails

Operations with positive-only logarithmic encoding represent a signed centered value using separate positive and negative magnitudes.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]] centers the input, creates positive and negative rails, processes each through logarithmic and exponential-difference stages, and subtracts the results. This allows signed normalization while keeping each logarithmic encoder input positive.

`clip_margin` independently insets both potential endpoints to form `[clip_margin, theta - clip_margin]`, while `eps` only stabilizes the LayerNorm variance. Inactive rails remain clamped to the margin, so their finite residual is distinct from denominator regularization.

## Scale Parameters

`theta`, `tau_s`, and finite domain endpoints jointly determine representable magnitude, latency, clipping, and numerical conditioning.

- `theta` is commonly both the symmetric potential clamp and the reference endpoint used by affine TTFS multiplication.
- `tau_s` controls log-encoding and exponential-difference scale.
- `tau_m` appears in exponential operator interfaces, but not every stabilized implementation currently uses it consistently.
- `clip_margin` keeps LayerNorm logarithmic rails away from zero and below `theta`, while `eps` independently stabilizes its variance denominator.

These quantities are configuration and calibration assumptions, not learned circuit characteristics. Their trade-offs are discussed in [[decisions#Explicit Finite Domains and Clamping]].

## Finite-Window Semantics

Clamping turns an unbounded mathematical mapping into a finite simulation domain and creates approximation cases at both endpoints.

Values beyond a potential range are clipped before encoding. Maintained Gaussian sampling distinguishes a delivered event from a deadline miss explicitly, even though both use finite stored times.

A missed event does not erase the analog state. The receiving operator evolves whatever physical state exists until $T_{\mathrm{obs}}$, reads that potential, clamps it to the output rails, and continues the operator chain. The complete opening/closing truth table is [[noise#Observation-Time Potential Invariant]].

The stored deadline time of a missed spike is only a tensor carrier. It must never be mistaken for a valid latest spike or substituted into a temporal formula without consulting `fired`.

Accuracy claims should report clamp or miss behavior alongside task metrics when finite-window effects are active. The relevant instrumentation is described in [[evaluation#Diagnostics and Instrumentation]].
