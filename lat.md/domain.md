# Domain Model

The project models analog values, TTFS spike times, and their valid ranges as explicit domain objects so composed operators can track finite-window assumptions.

## Potential and Declared Bounds

A `Potential` is a tensor paired with a declared `PotentialBounds` envelope, not a simulated membrane-neuron object.

[[utils/transforms/types.py#Potential]] is the carrier used by Transformer layers. [[utils/transforms/types.py#ClosedBounds]] is an immutable range with a clamping operation, while `PotentialBounds` and `TimeBounds` distinguish voltage-like and time-like quantities at the type level. Derived ranges require new objects rather than endpoint mutation.

Bounds serve three roles:

- They determine the affine or logarithmic encoding window.
- They support interval arithmetic for composed output ranges.
- They provide explicit locations for clipping diagnostics.

`ClosedBounds` denotes the inclusive representable envelope used by clamping, domain membership, and deadline classification. Construction rejects non-real, non-finite, or reversed endpoints centrally, and decorated tensor checks raise explicit runtime exceptions rather than optimization-sensitive assertions.

## Domain Propagation

The intended model-wide policy combines tight, depth-independent interval arithmetic with per-site calibration for nonlinear or recursively widening ranges.

For example, spiking linear layers derive local output bounds from fixed input and weight intervals. Pre-norm residual streams do not recursively add those intervals across all blocks; each post-add block output instead uses a frozen calibrated range and records excursions before clamping.

Maintained paths no longer construct bounds from observed forward-output extrema. Analytic intervals and frozen calibration records now define every production envelope; some remain intentionally conservative and require empirical clipping and accuracy validation.

The planned removal of runtime-derived bounds across all maintained operators and model adapters is tracked in [[todo#Static Bounds for All Operators]]. The completed inventory and replacement formulas are [[bounds-audit]].

## TTFS Encoding

Potential-to-spike transforms encode larger analog values as earlier events inside a declared time window.

[[utils/transforms/potential_to_spike.py#neg_linear_transform]] maps a bounded potential to a negative-linear latency. It rejects invalid or dtype-unrepresentable potential widths and time windows. On $[l,u]$, [[utils/transforms/potential_to_spike.py#neg_identity_transform]] gives $t=u-V$ and deadline $u-l$.

Affine PWM adapters require $l\le0\le u$ and encode one zero reference at $t_0=u$. Therefore $t_0-t(V)=V$ for symmetric or asymmetric fixed rails, and the same upstream range controls both input clipping and parameter-derived output interval arithmetic.

[[utils/transforms/potential_to_spike.py#neg_log_transform]] maps a strictly positive potential to `tau_s log(V_max/V)`. It explicitly rejects invalid scales and non-positive domains; the declared endpoint ratio fixes its upper time bound.

The encoders are decorated by the global noise boundary described in [[noise#Encoder Injection Boundary]]. Even when noise is disabled, their outputs are projected into the declared time range.

## Temporal-to-Potential Decoding

Time-to-potential operators turn latency differences into exponential analog values used by division and normalization.

[[utils/transforms/spike_to_potential.py#exp_operator]] computes a bounded exponential relative to the time-domain deadline and rejects invalid scales or dtype-level positive underflow. [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] composes integration, affine encoding, and an exponential stage to represent an exponential of a time difference.

[[utils/transforms/spike_to_potential.py#normalized_exp_operator]] evaluates `exp(t/tau_m)` and transforms its declared endpoints in the input tensor’s dtype. It rejects invalid time constants plus endpoint overflow or positive-domain underflow before decoding the payload.

Both deterministic and Gaussian direct exponentials cancel the identity encoder's fixed offset inside the exponent before evaluation, avoiding an overflowing intermediate that a later gain would only cancel algebraically. Misses still return reset zero rather than decoding their stored deadline carrier.

The event-aware exponential-difference path evolves and clamps its integration state, re-encodes it, then evaluates `exp(delta/tau_s)` with dtype-safe endpoint checks. An internal event miss returns exp-temporal reset zero rather than decoding its stored deadline carrier.

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

`theta`, the applicable temporal scale, and finite domain endpoints jointly determine representable magnitude, latency, clipping, and numerical conditioning.

- `theta` remains the default symmetric calibration scale and the basis of the global absolute Gaussian time-noise standard deviation. Affine adapters themselves consume the upstream fixed potential interval and derive the zero-reference time from that interval.
- `tau_s` controls log-encoding and exponential-difference scale.
- `tau_m` remains the generic exponential-operator parameter. Softmin and attention expose one `tau`, derived from the model-wide `tau_s`, and have no separate `tau_m` or `tau_s` keyword.
- `clip_margin` keeps LayerNorm logarithmic rails away from zero and below `theta`, while `eps` independently stabilizes its variance denominator.

These quantities are configuration and calibration assumptions, not learned circuit characteristics. Their trade-offs are discussed in [[decisions#Explicit Finite Domains and Clamping]].

## Finite-Window Semantics

Clamping turns an unbounded mathematical mapping into a finite simulation domain and creates approximation cases at both endpoints.

Values beyond a potential range are clipped before encoding. Maintained Gaussian sampling distinguishes a delivered event from a deadline miss explicitly, even though both use finite stored times.

A missed event does not erase another rail's analog state. The receiving operator evolves each delivered causal rail until $T_{\mathrm{obs}}$, reads their differential potential, clamps it to the output rails, and continues the operator chain. The complete signed-PWM truth table is [[noise#Observation-Time Potential Invariant]].

The stored deadline time of a missed spike is only a tensor carrier. It must never be mistaken for a valid latest spike or substituted into a temporal formula without consulting `fired`.

Accuracy claims should report clamp or miss behavior alongside task metrics when finite-window effects are active. The relevant instrumentation is described in [[evaluation#Diagnostics and Instrumentation]].
