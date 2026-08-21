# Operator System

The operator system builds Transformer arithmetic from a small timing-and-integration vocabulary while propagating numerical bounds alongside every result.

## Primitive PWM Integration

Pulse-width modulation uses causal event-to-deadline integration rails, with signed temporal differences formed by subtracting two rails sharing one deadline.

[[utils/transforms/primitive.py#unsigned_pulse_width_modulation_operator]] computes $V(T_{\mathrm{obs}}-t)$ on one causal rail whose configured deadline is no earlier than its event domain. [[utils/transforms/primitive.py#signed_pulse_width_modulation_operator]] evaluates the cancelled expression $V(t_B-t_A)$ directly for delivered tensors and exposes the two time-to-deadline durations only when a `SpikeSample` mask is needed.

The event-aware signed wrapper assigns reset duration zero to each missed rail independently. Two delivered events recover $V(t_B-t_A)$, an $A$-only event produces $V(T_{\mathrm{obs}}-t_A)$, a $B$-only event produces $-V(T_{\mathrm{obs}}-t_B)$, and two misses produce zero. Its returned domain remains the ideal both-event range so later saturation accounting can identify one-sided excursions before clamping.

### Differential Physical Realization

A differential accumulator can realize the signed operator with causal, non-negative current paths while the maintained tensor code evaluates only its terminal equation.

Split a signed drive into $V^+=\max(V,0)$ and $V^-=\max(-V,0)$, and let $d_A$ and $d_B$ be the delivered event-to-deadline durations, or zero for a missed event. Two physical accumulator rails may then store

$$
Q^+ = V^+d_A + V^-d_B,
\qquad
Q^- = V^+d_B + V^-d_A.
$$

Their differential readout is

$$
Q^+-Q^-=(V^+-V^-)(d_A-d_B)=V(d_A-d_B).
$$

This mapping has four possible current paths from the two timing rails and the two drive signs, but only two accumulator rails. It is a physically plausible differential PWM realization, not an additional computation performed by the behavioral tensor implementation. Device mismatch, common-mode limits, capacitor reset, and differential sensing remain circuit-level concerns outside the maintained simulator.

The signed wrapper returns one potential and derives its ideal range from the shared drive and signed time interval, avoiding the looser range obtained by treating deadline-terminated rails as independent. Existing model callers remain on [[utils/transforms/primitive.py#pulse_width_modulation_operator]] until each call site is migrated.

These functions evaluate the proposed rail behavior directly on tensors. They are not a circuit netlist, event router, deadline generator, or SPICE transient simulation.

## Missing-Event Readout

Every event-driven operator must eventually produce the clamped potential physically present at the observation deadline, even when an expected spike never arrives.

The new signed PWM primitive treats its two causal accumulator rails symmetrically: each delivered event contributes its time-to-deadline duration and each missed event contributes reset zero. Existing Gaussian operator call sites still use the older opening/closing trajectory documented in [[noise#Observation-Time Potential Invariant]] until they migrate to the signed wrapper one at a time.

Operator implementations may differ in their membrane trajectory, current kernel, or rail bounds, but not in this readout policy. A missed spike's stored deadline timestamp is metadata storage and cannot replace the physical state calculation.

## Composed Functions

Higher-level functions combine encoders, PWM integration, exponential decoding, reductions, and fixed affine scaling.

The principal compositions live in `utils/transforms/functions.py`. They share a common contract: tensor inputs plus declared bounds produce a tensor result plus derived bounds.

### Multiplication

Multiplication encodes one operand as a latency and uses the other as the integrated potential.

[[utils/transforms/functions.py#multiplication_operator]] clamps the encoded operand to `[-theta, theta]`, obtains `t = theta - B`, and integrates `V` from that event to `theta`. The resulting tensor is `V * B` under the ideal affine mapping.

Under maintained timing noise, the same function requests a decorated event for the encoded operand and one scalar zero-reference event for the operator call. It passes those existing samples to the signed PWM wrapper without resampling: each delivered event contributes its event-to-deadline rail and each miss contributes reset zero. The raw result is then saturation-counted and clamped to the original ideal product rails; no public Gaussian-specific multiplication API exists.

### Division

Division converts numerator and denominator to synchronized log latencies, then exponentiates their difference.

[[utils/transforms/functions.py#division_function]] requires `X <= Y` elementwise and uses the same positive joint domain for both log encoders. [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] then maps the latency difference back to the ratio.

The shared domain is essential because independent offsets would not cancel. Clamping, finite positive floors, and exponential implementation details determine where the simulated result is approximate.

In maintained event-aware execution, division passes both decorated log events into exponential difference. That operator first computes and rail-clamps the physical integration state at $T_{\mathrm{obs}}$, re-encodes this finite state through the ordinary decorated `phi_NP`, and evaluates `psi_NE` only if the internal event arrives. An internal event miss leaves the exponential response at reset zero, so the noisy output envelope includes zero.

### Exponential and Softmin

The attention normalization path represents a negated score followed by exponential normalization as softmin.

[[utils/transforms/functions.py#exponential_function]] composes affine encoding and an exponential temporal operator. [[utils/transforms/functions.py#softmin_function]] exponentiates scores, reduces the denominator, and invokes division to normalize along the last dimension.

Scaled dot product is implemented by [[utils/transforms/functions.py#scaled_dot_product_function]], which sums pairwise products and negates the usual attention logits. Applying softmin to those negated scores recovers the conventional softmax direction.

### Activations

Nonlinear activations are constructed from multiplication, exponential, division, and affine constants rather than treated as arbitrary current kernels.

[[utils/transforms/functions.py#gelu_approximation]] uses the cubic tanh approximation, including dynamic products for `x^2`, `x^3`, the gate, and gated output. [[utils/transforms/functions.py#tanh]] reduces tanh to an exponential and division identity. [[utils/transforms/functions.py#swiglu_function]] composes a sigmoid-like gate with two products.

Fixed scalar multiplication is conceptually absorbable into a synaptic weight in the paper’s operation-count abstraction, even when the reference tensor implementation calls the generic multiplication function. [[evaluation#Symbolic Operation-Count Check]] preserves that distinction.

SwiGLU treats the exponential neuron’s deadline response as `biased_exp` and applies a fixed current gain derived from the declared domain. The gain cancels the identity encoder’s constant offset without adding a runtime operator.

## Spiking Linear and Convolution

Dense and convolutional layers retain pretrained parameters but express multiply-accumulate behavior through the PWM identity.

[[utils/transformers/models/spiking_ops.py#SpikingLinear]] encodes its input once, broadcasts latency against the weight matrix, integrates, reduces the input dimension, and adds the original bias. [[utils/transformers/models/spiking_ops.py#SpikingConv2d]] applies the same principle after unfolding image patches.

These classes subclass PyTorch’s corresponding modules so parameter names and shapes remain checkpoint-compatible. In the noise-free tensor simulation they are intended to be numerically equivalent subject to clamping and floating-point arithmetic.

GPT-2 uses its own equivalent adapter, [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D]], to match Hugging Face’s transposed `Conv1D` parameter convention.

When maintained timing noise is enabled, all three affine adapters obtain both data and layer-shared zero-reference events through the decorated encoder. They form the two causal pulse widths and evaluate the complete signed PWM-MAC with `torch.nn.functional.linear`, `torch.nn.functional.conv2d`, or GPT-2's transposed `torch.addmm` contraction. These optimized kernels replace explicit per-synapse PWM tensor expansion; the learned weights still represent the integration drives. Every adapter clamps the raw affine result to its declared output rails.

## Spiking LayerNorm

LayerNorm is a multi-stage composition and the most delicate shared operator in the current model stack.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]] performs centering, dual-rail magnitude encoding, variance estimation, log encoding of variance and rails, exponential-difference normalization, and learned affine output scaling.

Three flags independently replace variance multiplication, log encoding, and exponential-difference decoding with tensor equivalents. These switches support causal attribution of error but also mean “spiking LayerNorm enabled” is not enough to identify the exact execution path; all three stage flags must be recorded.

The Gaussian path derives its output bounds without observing the current activation. The fully dense bypass uses the finite-feature bound $|z_i|\leq\sqrt{d-1}$, while mixed paths propagate exponential-difference ranges through subtraction and the learned affine map.

The current implementation has finite-floor and clipping behavior described in [[domain#Signed Values and Dual Rails]]. Ideal algebraic exactness and finite implementation fidelity should be reported separately.

## Spiking Attention

Attention composes spiking projections, signed dot products, softmin normalization, and PWM-weighted value accumulation.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_scaled_dot_product_attention]] clamps query and key to a fixed symmetric domain, computes negated scaled dot products, applies hard mask suppression, normalizes with softmin, and integrates encoded values against the resulting weights.

[[utils/transformers/integrations/spiking_sdpa_attention.py#attention_output_bounds]] memoizes the immutable value-integration rail for each $\theta$ and configured maximum source length. Masked scores use the same finite upper endpoint declared to softmin, and Gaussian and noise-free readouts clamp against the common output rail.

In maintained-noise execution, value and scalar zero-reference events come from the same decorated encoder used by affine PWM. Their physical durations are contracted with attention weights by matrix multiplication, avoiding an explicit `(L,S,D)` synapse tensor, and the observation-time output is clamped to its conservative summed rail envelope.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_sdpa_attention_forward]] adapts this implementation to the Hugging Face attention interface, including causal-mask selection and grouped-query compatibility checks. Grouped-query execution through the spiking kernel remains unsupported when native repetition cannot be used.

## Operator Validity Conditions

Every operator has domain conditions that are part of its contract rather than optional implementation details.

- Affine TTFS multiplication assumes the encoded operand lies inside the calibrated symmetric range.
- Log encoders require strictly positive inputs and synchronized domains when offsets must cancel.
- Division assumes the numerator does not exceed the denominator in its current contract.
- Exponential paths require a bounded input range to avoid overflow or underflow.
- Attention correctness assumes masks are broadcastable to the score tensor and suppressed before normalization.
- Signed integration requires event order to be preserved.

Violating these conditions typically produces clipping, assertions, or a numerically valid but semantically different result. [[evaluation#Diagnostics and Instrumentation]] describes the available checks.
