"""Shared SNN operator classes used across spiking transformer models."""

import math
from typing import Optional

import torch
from torch import nn

from utils.transforms import neg_identity_transform
from utils.transforms.functions import multiplication_operator, division_function
from utils.transforms.noise import clamp_gaussian_output, get_gaussian_time_noise
from utils.transforms.potential_to_spike import neg_log_transform
from utils.transforms.spike_to_potential import exponential_difference_operator
from utils.transforms.types import Potential, PotentialBounds, SpikeSample, TimeBounds


class SpikingLayerNorm(nn.Module):
    """LayerNorm via SNN operators (Lemma 4.4).

    Computes (x - mean) / std using ψ_M for variance and ψ_ED for division,
    with dual-rail encoding to handle signed activations.
    Output is rescaled to match pretrained ANN LayerNorm weights, so the residual
    1/sqrt(theta) factor from the spiking derivation is compensated explicitly.

    Each of the three SNN stages can be replaced with its standard-PyTorch equivalent
    for ablation analysis:
      use_spiking_mul    : ψ_M  for variance  vs  x²
      use_spiking_log    : φ_NL for encoding  vs  τ·log(hi/x)
      use_spiking_expdiff: ψ_ED for division  vs  exp((t_σ - t_x)/τ)
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1.0e-5,
        theta: float = 200.0,
        tau_s: float = 1.0,
        clip_margin: float = 1.0e-5,
        use_spiking_mul: bool = False,
        use_spiking_log: bool = True,
        use_spiking_expdiff: bool = True,
    ) -> None:
        """Initialize a dual-rail spiking LayerNorm module.

        ``eps`` is exclusively the numerical stabilizer added to the feature
        variance. ``clip_margin`` independently moves both endpoints of the
        positive TTFS rail inward, keeping logarithmic inputs away from zero and
        the upper endpoint below ``theta``.

        Args:
            normalized_shape: Feature shape normalized by LayerNorm.
            eps: Non-negative variance stabilizer used by LayerNorm arithmetic.
            theta: Upper scale from which the positive encoding rail is formed.
            tau_s: Temporal scale used by logarithmic and exponential operators.
            clip_margin: Positive inset applied to both TTFS potential endpoints.
            use_spiking_mul: Whether variance squaring uses the spiking product.
            use_spiking_log: Whether magnitudes use the logarithmic encoder.
            use_spiking_expdiff: Whether normalization uses exponential difference.

        Raises:
            ValueError: If ``clip_margin`` is non-finite, non-positive, or too
                large to leave a non-empty interval below ``theta``.
        """
        # Normalize the feature shape exactly once so scalar and tuple construction
        # retain the parameter layout expected by pretrained LayerNorm checkpoints.
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        # The positive dual rail will be [margin, theta - margin]. Keeping the margin
        # below theta/2 guarantees a strictly ordered domain for logarithmic coding.
        normalized_margin = float(clip_margin)
        if (
            not math.isfinite(normalized_margin)
            or normalized_margin <= 0.0
            or normalized_margin >= float(theta) / 2.0
        ):
            raise ValueError(
                "clip_margin must be finite and satisfy "
                "0 < clip_margin < theta / 2"
            )

        # Store the variance stabilizer and clipping margin separately so changing a
        # pretrained model's LayerNorm epsilon cannot silently alter its TTFS window.
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.clip_margin = normalized_margin
        self.theta = theta
        self.tau_s = tau_s
        self.use_spiking_mul = use_spiking_mul
        self.use_spiking_log = use_spiking_log
        self.use_spiking_expdiff = use_spiking_expdiff
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def _gaussian_forward(self, pot: Potential) -> Potential:
        """Evaluate LayerNorm with event-aware timing and fixed output bounds.

        The three ablation switches retain their existing meanings. Enabled
        spiking stages use the event-aware operators, while disabled stages use
        their direct PyTorch formulas. When logarithmic encoding is enabled but
        exponential-difference decoding is disabled, this method explicitly
        resolves the two causal rail masks at the shared observation deadline before
        applying the direct exponential formula. Every returned interval is
        derived from feature count, declared operator bounds, or learned affine
        parameters; the current activation tensor is never measured to define it.

        Args:
            pot: Input activation tensor paired with its calibrated bounds.

        Returns:
            A ``Potential`` containing the normalized and affine-scaled output.

        Raises:
            RuntimeError: If an event-aware logarithmic encoder does not return a
                ``SpikeSample``.
            ValueError: If logarithmic samples participating in one temporal
                difference do not share an observation deadline, or if an
                analytically propagated exponential endpoint is non-finite.
        """
        x: torch.Tensor = pot.value

        # With every spiking stage disabled there is no temporal event boundary at
        # which Gaussian noise can act, so preserve the exact dense LayerNorm value.
        if (
            not self.use_spiking_mul
            and not self.use_spiking_log
            and not self.use_spiking_expdiff
        ):
            out = nn.functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

            # For d normalized features, the centered population-normalized value
            # satisfies |z_i| <= sqrt(d - 1); a non-negative epsilon only contracts
            # that envelope. This gives a batch-independent alternative to observing
            # the dense output extrema, including the exact zero rail when d == 1.
            feature_count = math.prod(self.normalized_shape)
            normalized_limit = math.sqrt(max(feature_count - 1, 0))

            # Apply each learned gamma and beta to both normalized endpoints before
            # reducing across features. Parameter inspection is checkpoint state,
            # not activation calibration, and preserves negative gamma correctly.
            weight = self.weight.detach()
            bias = self.bias.detach()
            lower_candidate = weight * -normalized_limit + bias
            upper_candidate = weight * normalized_limit + bias
            dense_domain = PotentialBounds(
                torch.minimum(lower_candidate, upper_candidate).min().item(),
                torch.maximum(lower_candidate, upper_candidate).max().item(),
            )
            return Potential(out, dense_domain)

        eps = self.eps
        clip_margin = self.clip_margin
        theta = self.theta
        tau_s = self.tau_s

        # LayerNorm first forms a signed residual around the feature mean. Dual
        # positive rails use the independent clipping margin to keep both signs
        # inside the strictly positive log domain and below the threshold endpoint.
        x_err = x - x.mean(dim=-1, keepdim=True)
        domain_err = PotentialBounds(clip_margin, theta - clip_margin)
        x_err_pos = domain_err.clamp(x_err, name="x_err_pos")
        x_err_neg = domain_err.clamp(-x_err, name="x_err_neg")

        # The variance ablation changes only the squaring implementation. Gaussian
        # multiplication already performs its own sampled event readout and clamp.
        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(
                x_err_pos,
                domain_err,
                x_err_pos,
                domain_err,
                theta,
            )
            M_neg, _ = multiplication_operator(
                x_err_neg,
                domain_err,
                x_err_neg,
                domain_err,
                theta,
            )
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (
                x_err_pos.pow(2) + x_err_neg.pow(2)
            ).mean(dim=-1, keepdim=True)

        # Add the numerical stabilizer before enforcing the calibrated variance
        # rails; these rails also determine the logarithmic observation deadline.
        var_x = var_x + eps
        domain_var = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)
        var_x = domain_var.clamp(var_x, name="var_x")
        T0 = tau_s * math.log(domain_err.max / domain_err.min)

        if self.use_spiking_log:
            # The variance code uses tau_s/2 so decoding produces its square root.
            # Since hi_var = hi_err^2, this also gives both log encoders the same
            # bias and fixed deadline: (tau_s/2) log(hi_var) = tau_s log(hi_err).
            t_sigma = neg_log_transform(
                var_x,
                domain_var,
                tau_s=tau_s / 2.0,
                return_spike_sample=True,
                noise_site="layernorm.log_sigma",
            )
            t_err_pos = neg_log_transform(
                x_err_pos,
                domain_err,
                tau_s=tau_s,
                return_spike_sample=True,
                noise_site="layernorm.log_positive",
            )
            t_err_neg = neg_log_transform(
                x_err_neg,
                domain_err,
                tau_s=tau_s,
                return_spike_sample=True,
                noise_site="layernorm.log_negative",
            )
            if not all(
                isinstance(event, SpikeSample)
                for event in (t_sigma, t_err_pos, t_err_neg)
            ):
                raise RuntimeError(
                    "Gaussian SpikingLayerNorm log encoding must return SpikeSample"
                )
            tb_sigma = t_sigma.domain
            tb_err = t_err_pos.domain
        else:
            # Direct logarithms represent events that have already been delivered;
            # later spiking stages may wrap these tensors with all-true masks.
            hi_t = x.new_tensor(domain_err.max)
            hi2_t = x.new_tensor(domain_err.max ** 2)
            t_sigma = (tau_s / 2.0) * torch.log(hi2_t / var_x)
            t_err_pos = tau_s * torch.log(hi_t / x_err_pos)
            t_err_neg = tau_s * torch.log(hi_t / x_err_neg)
            tb_sigma = TimeBounds(0.0, T0)
            tb_err = TimeBounds(0.0, T0)

        if self.use_spiking_expdiff:
            # The event-aware operator owns both causal external rails, internal
            # exponential misses, and output saturation statistics.
            y_pos, domain_y_pos = exponential_difference_operator(
                t_err_pos,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            y_neg, domain_y_neg = exponential_difference_operator(
                t_err_neg,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            result = y_pos - y_neg

            # Both exponential-difference outputs are non-negative dual-rail
            # magnitudes. Ignore their positive lower endpoints deliberately and
            # use one relaxed symmetric rail: the signed difference cannot exceed
            # either magnitude's largest declared upper endpoint in absolute value.
            result_limit = max(domain_y_pos.max, domain_y_neg.max)
            result_domain = PotentialBounds(-result_limit, result_limit)

            # Retain the existing spiking affine rescaling used by this ablation.
            # Its Gaussian dispatcher samples the learned-weight multiplication and
            # returns the fixed product rail used for final affine propagation.
            weight_domain = PotentialBounds(
                self.weight.detach().min().item(),
                self.weight.detach().max().item(),
            )
            scaled, scaled_domain = multiplication_operator(
                result,
                result_domain,
                self.weight,
                weight_domain,
                theta,
            )
            out = scaled + self.bias

            # Bias addition translates the product interval. Global bias endpoints
            # are sufficient because every feature receives one fixed learned bias.
            bias = self.bias.detach()
            out_domain = PotentialBounds(
                scaled_domain.min + bias.min().item(),
                scaled_domain.max + bias.max().item(),
            )
        else:
            if isinstance(t_sigma, SpikeSample):
                # All three log encoders describe two differential readouts and must
                # use the same observation deadline before their rail masks combine.
                if not (
                    math.isclose(
                        float(t_sigma.domain.max),
                        float(t_err_pos.domain.max),
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    )
                    and math.isclose(
                        float(t_sigma.domain.max),
                        float(t_err_neg.domain.max),
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError(
                        "LayerNorm log events require a shared observation deadline"
                    )

                # Convert the shared sigma event and both residual events to causal
                # time-to-deadline pulse widths. Each miss leaves only its own rail at
                # reset, matching signed PWM without invoking the disabled exponential-
                # difference operator or sampling its internal exponential event.
                deadline = t_sigma.time.new_tensor(float(t_sigma.domain.max))
                sigma_pulse_width = torch.where(
                    t_sigma.fired,
                    (deadline - t_sigma.time).clamp_min(0.0),
                    torch.zeros_like(t_sigma.time),
                )
                positive_pulse_width = torch.where(
                    t_err_pos.fired,
                    (deadline - t_err_pos.time).clamp_min(0.0),
                    torch.zeros_like(t_err_pos.time),
                )
                negative_pulse_width = torch.where(
                    t_err_neg.fired,
                    (deadline - t_err_neg.time).clamp_min(0.0),
                    torch.zeros_like(t_err_neg.time),
                )

                # For delivered pairs, d_err-d_sigma equals t_sigma-t_err, exactly
                # the exponent used by exponential difference. One-sided misses retain
                # the surviving rail's signed contribution at the common deadline.
                delta_pos = positive_pulse_width - sigma_pulse_width
                delta_neg = negative_pulse_width - sigma_pulse_width
                y_pos = torch.exp(delta_pos / tau_s)
                y_neg = torch.exp(delta_neg / tau_s)
            else:
                # With direct log tensors there are no miss masks to resolve, so the
                # original analytical exponential-difference formula remains exact.
                y_pos = torch.exp((t_sigma - t_err_pos) / tau_s)
                y_neg = torch.exp((t_sigma - t_err_neg) / tau_s)

            # Each causal width lies in its declared deadline interval and a miss adds
            # the reset width zero. Their signed difference therefore spans the same
            # fixed endpoint interval before monotonic exponentiation.
            delta_min = min(
                0.0,
                float(tb_sigma.min) - float(tb_err.max),
            )
            delta_max = max(
                0.0,
                float(tb_sigma.max) - float(tb_err.min),
            )
            exponential_endpoints = torch.exp(
                x.new_tensor([delta_min / tau_s, delta_max / tau_s])
            )
            if not bool(torch.isfinite(exponential_endpoints).all()):
                raise ValueError(
                    "LayerNorm exponential bounds must be finite in the activation dtype"
                )
            y_domain = PotentialBounds(
                exponential_endpoints[0].item(),
                exponential_endpoints[1].item(),
            )

            # Positive and negative rails share the same exponential envelope. Their
            # difference therefore has a fixed signed interval independent of the
            # sampled delivery masks and of the current normalized activation.
            result = y_pos - y_neg
            result_domain = PotentialBounds(
                y_domain.min - y_domain.max,
                y_domain.max - y_domain.min,
            )
            out = self.weight * result + self.bias

            # Propagate the signed result through each feature's learned affine map.
            # Evaluating both endpoints handles either sign of gamma without using
            # activation extrema; reducing afterward yields one scalar module rail.
            weight = self.weight.detach()
            bias = self.bias.detach()
            lower_candidate = weight * float(result_domain.min) + bias
            upper_candidate = weight * float(result_domain.max) + bias
            out_domain = PotentialBounds(
                torch.minimum(lower_candidate, upper_candidate).min().item(),
                torch.maximum(lower_candidate, upper_candidate).max().item(),
            )

        # Values and metadata now share a predeclared mathematical envelope for all
        # Gaussian ablation combinations; no forward-pass observation widens it.
        return Potential(out, out_domain)
    
    def forward(self, pot: Potential) -> Potential:
        """Apply LayerNorm through the selected deterministic or Gaussian stages.

        Gaussian timing noise is process-wide mutable state, so the decision is
        made once at the public method boundary. Event-aware execution delegates
        to :meth:`_gaussian_forward`; the remainder of this method preserves the
        original tensor-only implementation used when timing noise is disabled.

        Args:
            pot: Input activation tensor paired with its calibrated bounds.

        Returns:
            A normalized ``Potential`` with bounds synchronized to its output.
        """
        # Keep sampled timestamps and delivery masks confined to the dedicated
        # implementation so deterministic tensor arithmetic remains event-free.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(pot)

        # The disabled branch deliberately remains the established deterministic
        # path, including all three stage-ablation combinations and their formulas.
        x: torch.Tensor = pot.value

        if not self.use_spiking_mul and not self.use_spiking_log and not self.use_spiking_expdiff:
            out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return Potential(out, PotentialBounds(out.min().item(), out.max().item()))

        eps = self.eps
        clip_margin = self.clip_margin
        theta = self.theta
        tau_s = self.tau_s

        x_err = x - x.mean(dim=-1, keepdim=True)
        
        # Debug: check if x_err exceeds theta
        # max_val = x_err.abs().max().item()
        # if max_val > theta:
        #     print(f"[DEBUG] x_err max {max_val:.2f} exceeds theta {theta}")
            
        # The clip margin defines only the representable positive dual-rail domain;
        # it is independent of the epsilon later added to the feature variance.
        domain_err: PotentialBounds = PotentialBounds(
            clip_margin,
            theta - clip_margin,
        )
        x_err_pos = domain_err.clamp(x_err, name="x_err_pos")
        x_err_neg = domain_err.clamp(-x_err, name="x_err_neg")

        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(x_err_pos, domain_err, x_err_pos, domain_err, theta)
            M_neg, _ = multiplication_operator(x_err_neg, domain_err, x_err_neg, domain_err, theta)
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (x_err_pos.pow(2) + x_err_neg.pow(2)).mean(dim=-1, keepdim=True)

        var_x = var_x + eps
        domain_var: PotentialBounds = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)
        var_x = domain_var.clamp(var_x, name="var_x")

        T0 = tau_s * math.log(domain_err.max / domain_err.min)
        if self.use_spiking_log:
            # First, we need tau_s/2 for sigma, to get sqrt of variance.
            # Thus, to match the bias terms between sigma and x_err terms
            # in the exponential difference operator, we also need to use tau_s/2 for x_err:
            # tau_s/2 * log(hi^2) = tau_s * log(hi).
            # hi^2 is the upper bound of variance, and hi is the upper bound of x_err, so this ensures the same bias term of tau_s * log(hi) for both sigma and x_err in the expdiff operator.
            t_sigma, tb_sigma = neg_log_transform(var_x, domain_var, tau_s=tau_s/2)
            t_err_pos, tb_err = neg_log_transform(x_err_pos, domain_err, tau_s=tau_s)
            t_err_neg, _ = neg_log_transform(x_err_neg, domain_err, tau_s=tau_s)
        else:
            _hi_t = x.new_tensor(domain_err.max)
            _hi2_t = x.new_tensor(domain_err.max ** 2)
            t_sigma = (tau_s / 2.0) * torch.log(_hi2_t / var_x)
            t_err_pos = tau_s * torch.log(_hi_t / x_err_pos)
            t_err_neg = tau_s * torch.log(_hi_t / x_err_neg)
            tb_sigma = TimeBounds(0.0, T0)
            tb_err = TimeBounds(0.0, T0)

        if self.use_spiking_expdiff:
            y_pos, _ = exponential_difference_operator(t_err_pos, tb_err, t_sigma, tb_sigma, tau_s=tau_s)
            y_neg, _ = exponential_difference_operator(t_err_neg, tb_err, t_sigma, tb_sigma, tau_s=tau_s)
            result: torch.Tensor = y_pos - y_neg
            out = multiplication_operator(
                result,
                PotentialBounds(result.min().item(), result.max().item()),
                self.weight,
                PotentialBounds(self.weight.min().item(), self.weight.max().item()),
                theta)[0] + self.bias
        else:
            y_pos = torch.exp((t_sigma - t_err_pos) / tau_s)
            y_neg = torch.exp((t_sigma - t_err_neg) / tau_s)
            result = y_pos - y_neg
            out = self.weight * result + self.bias

        out_domain = PotentialBounds(out.min().item(), out.max().item())
        return Potential(out, out_domain)


class SpikingLinear(nn.Linear):
    """Linear layer via ψ_PWM operator. Numerically identical to nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def _gaussian_forward(
        self,
        x: torch.Tensor,
        encoded_x: torch.Tensor,
        domain_x: PotentialBounds,
        domain_W: PotentialBounds,
    ) -> Potential:
        """Evaluate the affine layer from sampled data and reference events.

        ``encoded_x`` must already be clamped to the layer's symmetric identity-code
        domain. Every data element and one scalar zero-reference event supply the two
        causal rails of a signed PWM readout. Each missed event leaves its own rail at
        reset, while the delivered rail remains observable at the fixed deadline.

        Args:
            x: Original input tensor used for dtype, device, and scalar allocation.
            encoded_x: Input tensor clamped to ``domain_x``.
            domain_x: Symmetric input rails ``[-theta, theta]``.
            domain_W: Min/max bounds of the learned weight tensor.

        Returns:
            A bounded ``Potential`` containing the physical affine readout.

        Raises:
            RuntimeError: If either event-aware encoder call fails to return a
                ``SpikeSample``.
        """
        # Each input element independently opens its synaptic integration trajectory.
        # The delivery mask, not the finite carrier timestamp, decides whether it starts.
        data_event = neg_identity_transform(
            encoded_x,
            domain_x,
            return_spike_sample=True,
            noise_site="linear.data",
        )
        if not isinstance(data_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingLinear encoding must return SpikeSample"
            )

        # One zero codeword is a physical layer-wide reference event. Its scalar time
        # and fired state broadcast over batch, sequence, and input-feature dimensions.
        reference_event = neg_identity_transform(
            x.new_zeros(()),
            domain_x,
            return_spike_sample=True,
            noise_site="linear.reference",
        )
        if not isinstance(reference_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingLinear reference must return SpikeSample"
            )

        # Convert the two sampled events into causal pulse widths measured against
        # the same observation deadline. Each miss leaves only its own physical rail
        # at reset; no event ordering or additional sampling is introduced here.
        deadline = data_event.time.new_tensor(float(data_event.domain.max))
        data_pulse_width = torch.where(
            data_event.fired,
            (deadline - data_event.time).clamp_min(0.0),
            torch.zeros_like(data_event.time),
        )
        reference_pulse_width = torch.where(
            reference_event.fired,
            (deadline - reference_event.time).clamp_min(0.0),
            torch.zeros_like(reference_event.time),
        )
        signed_pulse_width = data_pulse_width - reference_pulse_width

        # This optimized kernel evaluates the complete PWM-MAC directly:
        # y_j = sum_i W_ji * (d_Ai - d_B) + b_j. It replaces only the explicit
        # per-synapse tensor expansion; the weights remain the physical PWM drives.
        # The conceptually equivalent, deliberately unmaterialized inner operation is:
        #
        # pwm_ji, _ = signed_pulse_width_modulation_operator(
        #     data_event_i, data_event.domain,
        #     reference_event, reference_event.domain,
        #     self.weight[j, i], domain_W,
        #     observation_deadline=float(data_event.domain.max),
        # )
        # y_j = sum_i(pwm_ji) + bias_j
        y = nn.functional.linear(signed_pulse_width, self.weight, self.bias)

        # Timing noise does not expand the calibrated ideal affine rails. Form every
        # weight/input endpoint product, then reduce over the full input fan-in.
        product_candidates = (
            domain_W.min * domain_x.min,
            domain_W.min * domain_x.max,
            domain_W.max * domain_x.min,
            domain_W.max * domain_x.max,
        )
        domain_y = PotentialBounds(
            min(product_candidates) * self.in_features,
            max(product_candidates) * self.in_features,
        )

        # Bias is part of both the physical affine output and its declared interval.
        # It remains independent of the differential synaptic contribution, including
        # the nonzero contribution that a surviving rail may produce after one miss.
        if self.bias is not None:
            domain_y = PotentialBounds(
                domain_y.min + self.bias.min().item(),
                domain_y.max + self.bias.max().item(),
            )

        # Count raw affine saturation before returning the rail-clamped potential to
        # the next Transformer operation.
        return Potential(
            clamp_gaussian_output(
                y,
                domain_y,
                site="linear.output",
                name="linear_y",
            ),
            domain_y,
        )

    def forward(self, input: Potential) -> Potential:
        """Apply the spiking affine map through deterministic or Gaussian PWM.

        Common input calibration and learned-weight bounds are computed once before
        dispatch. Gaussian execution delegates sampled event readout to
        :meth:`_gaussian_forward`; noise-free execution evaluates the delivered-time
        PWM-MAC with the same optimized linear kernel.

        Args:
            input: Tensor value paired with its upstream potential bounds.

        Returns:
            The affine output paired with conservative ideal potential rails.
        """
        # The affine encoder uses this layer's calibrated symmetric rails rather than
        # the upstream Potential domain, matching the existing pretrained conversion.
        x: torch.Tensor = input.value
        domain_x = PotentialBounds(-self.theta, self.theta)
        encoded_x = domain_x.clamp(x, name="linear_x")

        # Compute learned-weight endpoints once so both physical implementations use
        # an identical interval-arithmetic contract for the affine output.
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
        domain_W: PotentialBounds = PotentialBounds(w_min, w_max)

        # Keep event sampling, shared-reference handling, and saturation statistics
        # isolated in the private Gaussian method.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                domain_W,
            )

        # Encode delivered data times and subtract them from the scalar zero-codeword
        # time. This is the noise-free signed pulse width theta-t_A; no deadline-sized
        # rail tensors or output-by-input synapse tensor need to be materialized.
        data_time, _ = neg_identity_transform(encoded_x, domain_x)
        signed_pulse_width = self.theta - data_time

        # The optimized kernel is algebraically identical to summing one explicit
        # signed PWM call per synapse:
        # pwm_ji, _ = signed_pulse_width_modulation_operator(
        #     data_time_i, data_time_domain,
        #     theta, theta,
        #     self.weight[j, i], weight_domain,
        #     observation_deadline=2.0 * theta,
        # )
        # y_j = sum_i(pwm_ji) + bias_j.
        y = nn.functional.linear(signed_pulse_width, self.weight, self.bias)

        # Propagate the same ideal PWM product endpoints and input-feature fan-in used
        # by the explicit construction, independently of the current activation.
        product_candidates = (
            domain_W.min * domain_x.min,
            domain_W.min * domain_x.max,
            domain_W.max * domain_x.min,
            domain_W.max * domain_x.max,
        )
        domain_y = PotentialBounds(
            min(product_candidates) * self.in_features,
            max(product_candidates) * self.in_features,
        )

        # The optimized kernel already applied bias to the value; translate only the
        # declared rails here. Layers constructed without bias skip this adjustment.
        if self.bias is not None:
            b_min, b_max = self.bias.min().item(), self.bias.max().item()
            domain_y = PotentialBounds(domain_y.min + b_min, domain_y.max + b_max)
        return Potential(y, domain_y)

class SpikingConv2d(nn.Conv2d):
    """2D convolution via ψ_PWM operator. Numerically identical to nn.Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def _gaussian_forward(
        self,
        x: torch.Tensor,
        encoded_x: torch.Tensor,
        domain_x: PotentialBounds,
        domain_W: PotentialBounds,
    ) -> Potential:
        """Evaluate convolution from sampled data and shared reference events.

        Every encoded input element and one scalar zero-reference event supply the
        two causal rails of a signed PWM readout. Their signed pulse width is passed
        directly to PyTorch's grouped convolution kernel, which accelerates the full
        PWM-MAC while preserving stride, padding, dilation, learned bias, and the
        checkpoint-compatible parameter layout.

        Args:
            x: Original input tensor used for metadata and scalar allocation.
            encoded_x: Input clamped to the symmetric identity-code domain.
            domain_x: Symmetric input rails ``[-theta, theta]``.
            domain_W: Min/max bounds of the convolution weights.

        Returns:
            A bounded ``Potential`` containing the physical convolution readout.

        Raises:
            RuntimeError: If either event-aware encoder call fails to return a
                ``SpikeSample``.
        """
        # Sample one opening event per input activation. The complete spatial tensor
        # retains independent fired states for convolutional receptive-field readout.
        data_event = neg_identity_transform(
            encoded_x,
            domain_x,
            return_spike_sample=True,
            noise_site="conv2d.data",
        )
        if not isinstance(data_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingConv2d encoding must return SpikeSample"
            )

        # A single zero-reference event is shared across batches, channels, and
        # spatial locations; scalar broadcasting avoids resampling it per synapse.
        reference_event = neg_identity_transform(
            x.new_zeros(()),
            domain_x,
            return_spike_sample=True,
            noise_site="conv2d.reference",
        )
        if not isinstance(reference_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingConv2d reference must return SpikeSample"
            )

        # Convert both sampled events into causal pulse widths against one deadline.
        # Missing data or reference events independently leave their own rail at reset,
        # so the surviving rail retains its signed observation-time contribution.
        deadline = data_event.time.new_tensor(float(data_event.domain.max))
        data_pulse_width = torch.where(
            data_event.fired,
            (deadline - data_event.time).clamp_min(0.0),
            torch.zeros_like(data_event.time),
        )
        reference_pulse_width = torch.where(
            reference_event.fired,
            (deadline - reference_event.time).clamp_min(0.0),
            torch.zeros_like(reference_event.time),
        )
        signed_pulse_width = data_pulse_width - reference_pulse_width

        # The optimized convolution evaluates the complete per-receptive-field PWM
        # reduction. Conceptually, each unmaterialized synapse is equivalent to:
        #
        # pwm_synapse, _ = signed_pulse_width_modulation_operator(
        #     data_event_at_input, data_event.domain,
        #     reference_event, reference_event.domain,
        #     self.weight[out_channel, in_channel, kh, kw], domain_W,
        #     observation_deadline=float(data_event.domain.max),
        # )
        # y = sum_receptive_field(pwm_synapse) + bias
        #
        # Ordinary conv2d zero padding remains a zero-potential input outside the image.
        y = nn.functional.conv2d(
            signed_pulse_width,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Preserve the ideal calibrated envelope: combine every weight/input endpoint
        # product and multiply by the grouped receptive-field fan-in.
        product_candidates = (
            domain_W.min * domain_x.min,
            domain_W.min * domain_x.max,
            domain_W.max * domain_x.min,
            domain_W.max * domain_x.max,
        )
        kernel_height, kernel_width = self.kernel_size
        fan_in = (
            (self.in_channels // self.groups)
            * kernel_height
            * kernel_width
        )
        domain_y = PotentialBounds(
            min(product_candidates) * fan_in,
            max(product_candidates) * fan_in,
        )

        # Bias shifts both the physical value and its declared rails. It remains
        # independent of any nonzero contribution left by a surviving one-sided rail.
        if self.bias is not None:
            domain_y = PotentialBounds(
                domain_y.min + self.bias.min().item(),
                domain_y.max + self.bias.max().item(),
            )

        # Record clamp-before-rail saturation over every output activation, then
        # return the bounded potential expected by downstream Transformer blocks.
        return Potential(
            clamp_gaussian_output(
                y,
                domain_y,
                site="conv2d.output",
                name="conv2d_y",
            ),
            domain_y,
        )

    def forward(self, input: Potential) -> Potential:
        """Apply convolution through deterministic or Gaussian PWM integration.

        The method performs common input calibration and weight-bound extraction,
        then dispatches to the private event-aware implementation or the delivered-
        time PWM path. Both use the optimized convolution kernel and preserve stride,
        padding, dilation, grouping, bias, and conservative fan-in bounds.

        Args:
            input: Spatial activation tensor paired with upstream potential bounds.

        Returns:
            The convolution output paired with conservative ideal potential rails.
        """
        # Convolution uses its own symmetric encoder calibration, matching the
        # pretrained conversion path independently of the upstream Potential domain.
        x: torch.Tensor = input.value
        domain_x = PotentialBounds(-self.theta, self.theta)
        encoded_x = domain_x.clamp(x, name="conv2d_x")

        # Share one learned-weight interval between the direct convolution helper and
        # the explicit deterministic PWM bound propagation.
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
        domain_W: PotentialBounds = PotentialBounds(w_min, w_max)

        # Keep event sampling, shared-reference readout, and output saturation logging
        # isolated in the private Gaussian method.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                domain_W,
            )

        # Convert delivered identity-code times to signed pulse widths. Applying
        # ordinary zero padding to these widths is equivalent to padding event times
        # with theta, because theta-theta represents zero potential.
        data_time, _ = neg_identity_transform(encoded_x, domain_x)
        signed_pulse_width = self.theta - data_time

        # The optimized grouped convolution evaluates the sum of the conceptually
        # equivalent signed PWM call at every receptive-field synapse:
        # pwm_synapse, _ = signed_pulse_width_modulation_operator(
        #     data_time_synapse, data_time_domain,
        #     theta, theta,
        #     weight_synapse, weight_domain,
        #     observation_deadline=2.0 * theta,
        # )
        # Avoiding unfold also avoids the explicit output-channel-by-fan-in tensor.
        y = nn.functional.conv2d(
            signed_pulse_width,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Derive the unchanged ideal rail from weight/input endpoint products and the
        # grouped receptive-field fan-in used by the convolution reduction.
        kh, kw = self.kernel_size
        product_candidates = (
            domain_W.min * domain_x.min,
            domain_W.min * domain_x.max,
            domain_W.max * domain_x.min,
            domain_W.max * domain_x.max,
        )
        fan_in = (self.in_channels // self.groups) * kh * kw
        domain_y = PotentialBounds(
            min(product_candidates) * fan_in,
            max(product_candidates) * fan_in,
        )

        # Bias is already included in the convolution value, so translate only the
        # propagated bounds using the learned parameter endpoints.
        if self.bias is not None:
            b_min, b_max = self.bias.min().item(), self.bias.max().item()
            domain_y = PotentialBounds(domain_y.min + b_min, domain_y.max + b_max)
        return Potential(y, domain_y)


def _apply_norm(norm: nn.Module, pot: Potential) -> Potential:
    if isinstance(norm, SpikingLayerNorm):
        return norm(pot)
    out = norm(pot.value)
    return Potential(out, PotentialBounds(out.min().item(), out.max().item()))


if __name__ == "__main__":
    import torch
    from torch import nn

    torch.manual_seed(42)
    
    dim = 768
    theta = 400.0
    
    # Initialize layers
    ln = nn.LayerNorm(dim)
    sln = SpikingLayerNorm(dim, theta=theta)
    
    # Sync weights
    with torch.no_grad():
        sln.weight.copy_(ln.weight)
        sln.bias.copy_(ln.bias)
    
    max_diff = -1.0
    worst_std = -1
    max_x_err_at_worst = 0.0
    
    print(f"Testing standard deviations from 1 to 128 for dim={dim}, theta={theta}...")
    
    for std in range(1, 129):
        # Create input tensor with mean 0 and current std
        x = torch.randn(1, dim) * std
        # Create Potential object for SpikingLayerNorm
        pot = Potential(x, PotentialBounds(x.min().item(), x.max().item()))
        
        with torch.no_grad():
            x_err = x - x.mean(dim=-1, keepdim=True)
            max_x_err = x_err.abs().max().item()
            
            ln_out = ln(x)
            sln_out = sln(pot).value
            
            diff = (ln_out - sln_out).abs().max().item()
            
            if diff > max_diff:
                max_diff = diff
                worst_std = std
                max_x_err_at_worst = max_x_err
                
    print("\n=== Result ===")
    print(f"Standard deviation with maximum difference: {worst_std}")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Max abs(x_err) at worst std: {max_x_err_at_worst:.2f} (theta={theta})")
