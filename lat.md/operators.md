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

The signed wrapper returns one potential and derives its ideal range from the shared drive and signed time interval, avoiding the looser range obtained by treating deadline-terminated rails as independent. The former algebraic single-rail helper has been removed; maintained callers now use these causal primitives or an explicitly equivalent optimized reduction.

These functions evaluate the proposed rail behavior directly on tensors. They are not a circuit netlist, event router, deadline generator, or SPICE transient simulation.

## Missing-Event Readout

Every event-driven operator must eventually produce the clamped potential physically present at the observation deadline, even when an expected spike never arrives.

The signed PWM primitive treats its two causal accumulator rails symmetrically: each delivered event contributes its time-to-deadline duration and each missed event contributes reset zero. Multiplication, affine adapters, attention value integration, exponential difference, and LayerNorm's direct exponential ablation now follow the same invariant.

Operator implementations may differ in their membrane trajectory, current kernel, or rail bounds, but not in this readout policy. A missed spike's stored deadline timestamp is metadata storage and cannot replace the physical state calculation.

## Composed Functions

Higher-level functions combine encoders, PWM integration, exponential decoding, reductions, and fixed affine scaling.

The principal compositions live in `utils/transforms/functions.py`. They share a common contract: tensor inputs plus declared bounds produce a tensor result plus derived bounds.

### Multiplication

Multiplication encodes one operand as a latency and uses the other as the integrated potential.

[[utils/transforms/functions.py#multiplication_operator]] clamps the encoded operand to `[-theta, theta]`, obtains $t=\theta-B$, and evaluates the delivered data time and scalar $\theta$ reference through signed PWM. The noise-free wrapper directly computes $V(\theta-t)=VB$ after algebraic deadline cancellation. Its ideal output interval uses the caller-declared factor endpoints after clamping them to the encoder rail, so fixed coefficients and bounded gates do not acquire a full-$\theta$ range.

Under maintained timing noise, the same function requests a decorated event for the encoded operand and one scalar zero-reference event for the operator call. It passes those existing samples to the signed PWM wrapper without resampling: each delivered event contributes its event-to-deadline rail and each miss contributes reset zero. The raw result is then saturation-counted and clamped to the same declared-factor product rails; no public Gaussian-specific multiplication API exists.

### Division

Division converts numerator and denominator to synchronized log latencies, then exponentiates their difference.

[[utils/transforms/functions.py#division_function]] requires `X <= Y` elementwise, uses the same positive joint domain for both log encoders, and returns the noise-independent public range $[0,1]$. [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] remains an unrestricted primitive because dual-rail LayerNorm requires both temporal orderings.

The shared domain is essential because independent offsets would not cancel. Clamping, finite positive floors, and exponential implementation details determine where the simulated result is approximate.

In maintained event-aware execution, division passes both decorated log events into exponential difference. That operator applies a fixed unit-negative drive to the two causal signed-PWM rails, clamps the intermediate $t_A-t_B$ state, re-encodes it through the ordinary decorated `phi_NP`, and evaluates `psi_NE` only if the internal event arrives. Either external miss leaves the other rail visible and an internal event miss leaves the response at reset zero. Raw division responses above one are counted as output overflow before the public $[0,1]$ clamp.

### Exponential and Softmin

The attention normalization path represents a negated score followed by exponential normalization as softmin.

[[utils/transforms/functions.py#exponential_function]] composes affine encoding and an exponential temporal operator. [[utils/transforms/functions.py#softmin_function]] exponentiates scores, reduces the denominator, and invokes division to normalize along the last dimension. It exposes one `tau` scale and maps that value to the generic exponential and logarithmic sub-operators. Its public output domain is always the structural normalized-weight interval $[0,1]$; Gaussian observation-time excursions are counted at `softmin.output` before the returned tensor is clamped to those rails.

Scaled dot product is implemented by [[utils/transforms/functions.py#scaled_dot_product_function]], which sums pairwise products and negates the usual attention logits. Applying softmin to those negated scores recovers the conventional softmax direction.

### Activations

Nonlinear activations are constructed from multiplication, exponential, division, and affine constants rather than treated as arbitrary current kernels.

[[utils/transforms/functions.py#gelu_approximation]] uses the cubic tanh approximation, including dynamic products for `x^2`, `x^3`, the gate, and gated output. [[utils/transforms/functions.py#tanh]] reduces tanh to an exponential and division identity, then returns the structural $[-1,1]$ domain and records Gaussian excursions before clamping. [[utils/transforms/functions.py#gelu_approximation_sigmoid]] and [[utils/transforms/functions.py#swiglu_function]] clamp their completed sigmoid-like gates to $[0,1]$ before downstream multiplication; Gaussian gate excursions are recorded without widening the propagated product domains.

Fixed scalar multiplication is conceptually absorbable into a synaptic weight in the paper’s operation-count abstraction, even when the reference tensor implementation calls the generic multiplication function. [[evaluation#Symbolic Operation-Count Check]] preserves that distinction.

SwiGLU treats the exponential neuron’s deadline response as `biased_exp` and applies a fixed current gain derived from the declared domain. The gain cancels the identity encoder’s constant offset without adding a runtime operator. [[utils/transforms/functions.py#_gaussian_swiglu_function]] and the deterministic branch share the same $[0,1]$ gate contract.

## Spiking Linear and Convolution

Dense and convolutional layers retain pretrained parameters but express multiply-accumulate behavior through the PWM identity.

[[utils/transformers/models/spiking_ops.py#SpikingLinear]] encodes its input once and evaluates the complete signed PWM reduction with the optimized linear kernel. [[utils/transformers/models/spiking_ops.py#SpikingConv2d]] applies the same principle with the grouped convolution kernel.

These classes subclass PyTorch’s corresponding modules so parameter names and shapes remain checkpoint-compatible. In the noise-free tensor simulation they are intended to be numerically equivalent subject to clamping and floating-point arithmetic.

The three affine adapters freeze parameter-derived safety rails after checkpoint loading and static perturbation. [[utils/transformers/models/spiking_ops.py#SpikingLinear#freeze_parameter_bounds]] and [[utils/transformers/models/spiking_ops.py#SpikingConv2d#freeze_parameter_bounds]] reduce each output row or grouped kernel, while [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D#freeze_parameter_bounds]] reduces the transposed weight's input dimension. For an upstream interval $[l,u]$, each weight selects the smaller and larger of $W_{ji}l$ and $W_{ji}u$ before reduction and optional bias addition. The symmetric formula $r_j=\theta\sum_i|W_{ji}|$ is only the special case $[l,u]=[-\theta,\theta]$. Each adapter reuses the resulting immutable interval and rejects later standard in-place parameter changes unless explicitly refreshed.

[[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]], [[utils/transformers/models/spiking_ops.py#SpikingConv2d#forward]], and [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D#forward]] attach the same frozen rail to deterministic and Gaussian outputs, so changing noise configuration does not change metadata. Their Gaussian helpers use it directly for saturation accounting and clamping without rescanning weight or bias bounds.

GPT-2 uses its own equivalent adapter, [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D]], to match Hugging Face’s transposed `Conv1D` parameter convention.

In noise-free execution, the scalar zero-reference time cancels algebraically and the delivered signed width is passed to `torch.nn.functional.linear`, `torch.nn.functional.conv2d`, or GPT-2's transposed matrix contraction. Under maintained timing noise, all three affine adapters instead obtain data and layer-shared zero-reference events through the decorated encoder and form their two causal widths before calling the same kernels. These optimized reductions replace explicit per-synapse PWM tensor expansion; learned weights remain the integration drives. Every noisy adapter clamps the raw affine result to its declared output rails.

## Spiking LayerNorm

LayerNorm is a multi-stage composition and the most delicate shared operator in the current model stack.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]] performs centering, dual-rail magnitude encoding, variance estimation, log encoding of variance and rails, exponential-difference normalization, and learned affine output scaling.

Three flags independently replace variance multiplication, log encoding, and exponential-difference decoding with tensor equivalents. These switches support causal attribution of error but also mean “spiking LayerNorm enabled” is not enough to identify the exact execution path; all three stage flags must be recorded.

Gaussian and deterministic paths derive output bounds without observing the current activation. The fully dense bypass uses $|z_i|\leq\sqrt{d-1}$. Every mixed dual-rail path uses $|z_i|\leq\sqrt d$, because one signed numerator square is bounded by the sum of all rail squares used in the denominator.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#freeze_parameter_bounds]] precomputes learned weight, bias, and final output domains for the active ablation configuration. Dense mode uses $\sqrt{d-1}$ and every mixed mode uses $\sqrt d$ before affine endpoint propagation. This avoids catastrophic cancellation when the next float32 identity encoder subtracts timestamps on a formerly $10^7$–$10^8$ log-ratio rail.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_gaussian_forward]] and [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]] reuse the same immutable domain. Noise-off roundoff and Gaussian one-sided-event excursions are clamped at the finite-feature normalization rail; Gaussian mode records pre-clamp saturation under `layernorm.normalized_output`.

[[utils/transformers/models/spiking_ops.py#_apply_norm]] applies the same finite-feature envelope to ordinary `torch.nn.LayerNorm`, then transforms both endpoints with its learned scale and optional bias. Modules without an affine stage retain the symmetric normalized envelope.

When log encoding is enabled but exponential difference is bypassed, the method computes causal residual and sigma pulse widths directly and applies $\exp((d_{\mathrm{err}}-d_\sigma)/\tau_s)$. This preserves symmetric one-sided misses without sampling the disabled internal exponential event.

The current implementation has finite-floor and clipping behavior described in [[domain#Signed Values and Dual Rails]]. Ideal algebraic exactness and finite implementation fidelity should be reported separately.

## Spiking Attention

Attention composes spiking projections, signed dot products, softmin normalization, and PWM-weighted value accumulation.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_scaled_dot_product_attention]] clamps query and key to a fixed symmetric domain, computes negated scaled dot products, applies hard mask suppression, normalizes with softmin, and integrates encoded values against the resulting weights.

[[utils/transformers/integrations/spiking_sdpa_attention.py#attention_output_bounds]] memoizes the immutable value-integration rail for each $\theta$ and configured maximum source length. Masked scores use the same finite upper endpoint declared to softmin, and Gaussian and noise-free readouts clamp against the common output rail.

In noise-free execution, the scalar zero-reference time cancels algebraically and matrix multiplication reduces the delivered signed value widths. Under maintained noise, value and scalar zero-reference events come from the same decorated encoder used by affine PWM. Each event supplies an independent causal pulse width, and a miss leaves only that rail at reset. The same matrix multiplication evaluates the complete attention-weight-driven signed PWM reduction while avoiding an explicit `(L,S,D)` synapse tensor. The raw observation-time output is then saturation-counted and clamped to its conservative summed rail envelope.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_sdpa_attention_forward]] adapts this implementation to the Hugging Face attention interface, including causal-mask selection and grouped-query compatibility checks. Grouped-query execution through the spiking kernel remains unsupported when native repetition cannot be used.

## Operator Validity Conditions

Every operator has domain conditions that are part of its contract rather than optional implementation details.

- The generic multiplication operator encodes its factor on the symmetric $[-\theta,\theta]$ rail, while affine adapters require their upstream fixed input interval to be finite, ordered, and contain zero.
- Log encoders require strictly positive inputs and synchronized domains when offsets must cancel.
- Division assumes the numerator does not exceed the denominator in its current contract.
- Exponential paths require a bounded input range to avoid overflow or underflow.
- Attention correctness assumes masks are broadcastable to the score tensor and suppressed before normalization.
- Signed integration requires both causal rails to share one fixed observation deadline.

Violating these conditions typically produces clipping, assertions, or a numerically valid but semantically different result. [[evaluation#Diagnostics and Instrumentation]] describes the available checks.
