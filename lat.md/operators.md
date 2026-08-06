# Operator System

The operator system builds Transformer arithmetic from a small timing-and-integration vocabulary while propagating numerical bounds alongside every result.

## Primitive PWM Integration

Pulse-width modulation is the central primitive: a potential is integrated over the signed interval between two event times.

[[utils/transforms/primitive.py#pulse_width_modulation_operator]] computes `V * (t_B - t_A)` and derives its output interval from all endpoint products. Arrival order therefore determines the sign of the temporal difference; preserving that order is a functional requirement of any physical realization.

The code evaluates this identity directly on tensors. It represents the behavior of the proposed primitive, not a circuit netlist, event router, or SPICE transient simulation.

## Composed Functions

Higher-level functions combine encoders, PWM integration, exponential decoding, reductions, and fixed affine scaling.

The principal compositions live in `utils/transforms/functions.py`. They share a common contract: tensor inputs plus declared bounds produce a tensor result plus derived bounds.

### Multiplication

Multiplication encodes one operand as a latency and uses the other as the integrated potential.

[[utils/transforms/functions.py#multiplication_operator]] clamps the encoded operand to `[-theta, theta]`, obtains `t = theta - B`, and integrates `V` from that event to `theta`. The resulting tensor is `V * B` under the ideal affine mapping.

### Division

Division converts numerator and denominator to synchronized log latencies, then exponentiates their difference.

[[utils/transforms/functions.py#division_function]] requires `X <= Y` elementwise and uses the same positive joint domain for both log encoders. [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] then maps the latency difference back to the ratio.

The shared domain is essential because independent offsets would not cancel. Clamping, finite positive floors, and exponential implementation details determine where the simulated result is approximate.

### Exponential and Softmin

The attention normalization path represents a negated score followed by exponential normalization as softmin.

[[utils/transforms/functions.py#exponential_function]] composes affine encoding and an exponential temporal operator. [[utils/transforms/functions.py#softmin_function]] exponentiates scores, reduces the denominator, and invokes division to normalize along the last dimension.

Scaled dot product is implemented by [[utils/transforms/functions.py#scaled_dot_product_function]], which sums pairwise products and negates the usual attention logits. Applying softmin to those negated scores recovers the conventional softmax direction.

### Activations

Nonlinear activations are constructed from multiplication, exponential, division, and affine constants rather than treated as arbitrary current kernels.

[[utils/transforms/functions.py#gelu_approximation]] uses the cubic tanh approximation, including dynamic products for `x^2`, `x^3`, the gate, and gated output. [[utils/transforms/functions.py#tanh]] reduces tanh to an exponential and division identity. [[utils/transforms/functions.py#swiglu_function]] composes a sigmoid-like gate with two products.

Fixed scalar multiplication is conceptually absorbable into a synaptic weight in the paper’s operation-count abstraction, even when the reference tensor implementation calls the generic multiplication function. [[evaluation#Symbolic Operation-Count Check]] preserves that distinction.

## Spiking Linear and Convolution

Dense and convolutional layers retain pretrained parameters but express multiply-accumulate behavior through the PWM identity.

[[utils/transformers/models/spiking_ops.py#SpikingLinear]] encodes its input once, broadcasts latency against the weight matrix, integrates, reduces the input dimension, and adds the original bias. [[utils/transformers/models/spiking_ops.py#SpikingConv2d]] applies the same principle after unfolding image patches.

These classes subclass PyTorch’s corresponding modules so parameter names and shapes remain checkpoint-compatible. In the noise-free tensor simulation they are intended to be numerically equivalent subject to clamping and floating-point arithmetic.

GPT-2 uses its own equivalent adapter, [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D]], to match Hugging Face’s transposed `Conv1D` parameter convention.

## Spiking LayerNorm

LayerNorm is a multi-stage composition and the most delicate shared operator in the current model stack.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]] performs centering, dual-rail magnitude encoding, variance estimation, log encoding of variance and rails, exponential-difference normalization, and learned affine output scaling.

Three flags independently replace variance multiplication, log encoding, and exponential-difference decoding with tensor equivalents. These switches support causal attribution of error but also mean “spiking LayerNorm enabled” is not enough to identify the exact execution path; all three stage flags must be recorded.

The current implementation has finite-floor and clipping behavior described in [[domain#Signed Values and Dual Rails]]. Ideal algebraic exactness and finite implementation fidelity should be reported separately.

## Spiking Attention

Attention composes spiking projections, signed dot products, softmin normalization, and PWM-weighted value accumulation.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_scaled_dot_product_attention]] clamps query and key to a fixed symmetric domain, computes negated scaled dot products, applies hard mask suppression, normalizes with softmin, and integrates encoded values against the resulting weights.

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
