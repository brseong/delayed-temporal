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

Range selection distinguishes three cases. A function's fixed output range is distinct from an interval that depends on the input bounds but is frozen for inference:

1. A function whose output range is fixed by its definition and fixed parameters, independently of the input activation bounds or calibration observations, retains that range. Softmax output in $[0,1]$ is the representative example.
2. A range that is finite but becomes too wide through weights, reductions, or repeated residual addition is a calibration target at selected boundaries. A finite formula alone does not make calibration unnecessary.
3. A mapping with no finite output or timing bound on its original domain requires a restricted representable domain. Log encoding near zero is one example. Setting a finite limit is not, by itself, evidence that the limit was selected from data.

A finite interval computed from a restricted input is not automatically the first case. Nor does freezing a calibrated interval make the function's output range independent of its input bounds. GELU retains an upper bound from its input and a fixed lower bound; this output propagation is treated separately rather than as the first case.

Spiking linear layers derive output intervals from fixed input bounds and loaded weights. With frozen layer-wise calibration enabled, selected residual boundaries replace interval sums with persisted ranges after counting values outside the interval and clamping. With calibration disabled, those boundaries retain analytic interval addition. Both modes avoid bounds derived from the current batch, but only the former applies the layer-wise limits intended to control growth.

The three cases describe why a range needs attention, not the signed or one-sided record policies used to select its endpoints; see [[calibration#Two-pass Collection#Quantile and Margin Policy]]. Current experiment settings are recorded separately in [[noise#Local-Window Timing Noise Sweep]].

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

Actual magnitudes use `[0, r]`, where $r$ is the frozen centered-input radius selected by calibration or derived from the incoming interval width. `clip_margin` supplies only the positive floor of logarithmic inputs. The log upper endpoint is $\sqrt{r^2+\mathrm{eps}}$, so adding the variance stabilizer cannot introduce a hidden upper clamp. Zero magnitudes remain zero in the variance, and below-floor numerator rails are masked after decoding.

## Scale Parameters

Declared finite domains and temporal scales determine representable magnitude, latency, clipping, and numerical conditioning; there is no model-wide potential threshold setting.

- Each identity encoder consumes the zero-containing `PotentialBounds` carried by its operand and derives its zero-reference time from those endpoints.
- Attention consumes explicit Q/K/V bounds and limits scores only by frozen calibration plus the dtype/source-length representability ceiling.
- `tau_s` controls log-encoding and exponential-difference scale.
- `tau_m` remains the generic exponential-operator parameter. Softmin and attention expose one `tau`, derived from the model-wide `tau_s`, and have no separate `tau_m` or `tau_s` keyword.
- `clip_margin` keeps LayerNorm logarithmic inputs away from zero, while `eps` independently stabilizes its variance denominator and participates in the shared log upper endpoint.

These quantities are configuration and calibration assumptions, not learned circuit characteristics. Their trade-offs are discussed in [[decisions#Explicit Finite Domains and Clamping]].

## Finite-Window Semantics

Clamping turns an unbounded mathematical mapping into a finite simulation domain and creates approximation cases at both endpoints.

Values beyond a potential range are clipped before encoding. Maintained Gaussian sampling distinguishes a delivered event from a deadline miss explicitly, even though both use finite stored times.

A missed event does not erase another rail's analog state. The receiving operator evolves each delivered causal rail until $T_{\mathrm{obs}}$, reads their differential potential, clamps it to the output rails, and continues the operator chain. The complete signed-PWM truth table is [[noise#Observation-Time Potential Invariant]].

The stored deadline time of a missed spike is only a tensor carrier. It must never be mistaken for a valid latest spike or substituted into a temporal formula without consulting `fired`.

Accuracy claims should report clamp or miss behavior alongside task metrics when finite-window effects are active. The relevant instrumentation is described in [[evaluation#Diagnostics and Instrumentation]].
