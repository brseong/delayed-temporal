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
    Matched logarithmic references cancel the finite-domain upper endpoint, so the
    normalized result has no residual theta scale before pretrained affine weights.

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

    def freeze_parameter_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[PotentialBounds, PotentialBounds, PotentialBounds]:
        """Freeze learned parameter and final output bounds for this ablation.

        LayerNorm has two finite-feature affine-input envelopes. A fully dense module
        uses ``sqrt(d - 1)`` for population-normalized features. Every dual-rail
        spiking or mixed path uses ``sqrt(d)``; Gaussian excursions beyond that
        ideal normalization rail are saturated before the learned affine stage.
        This method combines the active envelope with learned scale and bias once,
        after checkpoint loading and static perturbation.

        Args:
            refresh: Recompute metadata after an intentional parameter or
                configuration update. The default rejects stale cached bounds.

        Returns:
            A tuple containing frozen weight bounds, frozen bias bounds, and the
            module's final output domain for its current ablation configuration.

        Raises:
            RuntimeError: If parameters or bound-relevant configuration changed
                after freezing without refresh, or changed during recomputation.
            ValueError: If a physical scale, stabilizer, parameter, or derived
                endpoint is non-finite or violates the LayerNorm domain contract.

        Notes:
            Parameter mutation checks use PyTorch version counters and therefore do
            not support direct ``parameter.data`` writes. Standard checkpoint loads
            and ``torch.no_grad()`` perturbations are detected.
        """
        # Validate every scalar that determines the active mathematical envelope
        # before consulting the cache. A configuration change must never reuse rails
        # computed for a different TTFS window or ablation topology.
        if isinstance(self.theta, bool):
            raise ValueError("SpikingLayerNorm theta must be finite and positive")
        theta = float(self.theta)
        margin = float(self.clip_margin)
        tau_s = float(self.tau_s)
        eps = float(self.eps)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ValueError("SpikingLayerNorm theta must be finite and positive")
        if not math.isfinite(margin) or margin <= 0.0 or margin >= theta / 2.0:
            raise ValueError(
                "SpikingLayerNorm clip_margin must satisfy 0 < margin < theta / 2"
            )
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("SpikingLayerNorm tau_s must be finite and positive")
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError("SpikingLayerNorm eps must be finite and non-negative")

        # Parameter versions and all bound-relevant scalar switches form one cache
        # identity. This catches threshold, margin, time-scale, epsilon, and ablation
        # mutations even when gamma and beta remain byte-identical.
        identity = (
            self.weight._version,
            self.bias._version,
            theta,
            margin,
            tau_s,
            eps,
            bool(self.use_spiking_mul),
            bool(self.use_spiking_log),
            bool(self.use_spiking_expdiff),
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")
        if cached is not None and not refresh:
            cached_identity, cached_bounds = cached
            if identity != cached_identity:
                raise RuntimeError(
                    "SpikingLayerNorm parameters or configuration changed after "
                    "bounds were frozen; call freeze_parameter_bounds(refresh=True) "
                    "before inference"
                )
            return cached_bounds

        # Read learned affine parameters once in float64. The scalar weight and bias
        # domains are retained because the spiking final-scale multiplication still
        # consumes the declared gamma interval as part of its operator contract.
        weight = self.weight.detach().to(dtype=torch.float64)
        bias = self.bias.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError("SpikingLayerNorm affine parameters must be finite")
        weight_domain = PotentialBounds(weight.min().item(), weight.max().item())
        bias_domain = PotentialBounds(bias.min().item(), bias.max().item())

        # Dense population normalization has the tight sqrt(d-1) theorem. For every
        # mixed dual-rail path, |a_i-b_i|^2 <= a_i^2+b_i^2 and the denominator
        # contains the mean of all d rail-square sums, giving the bound sqrt(d).
        all_dense = not (
            self.use_spiking_mul
            or self.use_spiking_log
            or self.use_spiking_expdiff
        )
        feature_count = math.prod(self.normalized_shape)
        if all_dense:
            result_limit = math.sqrt(max(feature_count - 1, 0))
            effective_weight = weight
        else:
            result_limit = math.sqrt(feature_count)
            effective_weight = (
                weight.clamp(-theta, theta)
                if self.use_spiking_expdiff
                else weight
            )
        if not math.isfinite(result_limit) or result_limit < 0.0:
            raise ValueError("SpikingLayerNorm normalized bound must be finite")

        # Dense and direct branches apply gamma featurewise. The spiking final
        # multiplication propagates one global gamma interval; pre-affine Gaussian
        # excursions are already saturated at the finite-feature normalization rail.
        if self.use_spiking_expdiff and not all_dense:
            effective_min = effective_weight.min().item()
            effective_max = effective_weight.max().item()
            product_candidates = (
                -result_limit * effective_min,
                -result_limit * effective_max,
                result_limit * effective_min,
                result_limit * effective_max,
            )
            output_domain = PotentialBounds(
                min(product_candidates) + bias_domain.min,
                max(product_candidates) + bias_domain.max,
            )
        else:
            lower_candidate = effective_weight * -result_limit + bias
            upper_candidate = effective_weight * result_limit + bias
            output_domain = PotentialBounds(
                torch.minimum(lower_candidate, upper_candidate).min().item(),
                torch.maximum(lower_candidate, upper_candidate).max().item(),
            )
        if not math.isfinite(float(output_domain.min)) or not math.isfinite(
            float(output_domain.max)
        ):
            raise ValueError("SpikingLayerNorm output bounds must be finite")

        # Rebuild the identity after reductions so concurrent parameter or scalar
        # configuration writes cannot publish a mixed-version metadata tuple.
        final_identity = (
            self.weight._version,
            self.bias._version,
            float(self.theta),
            float(self.clip_margin),
            float(self.tau_s),
            float(self.eps),
            bool(self.use_spiking_mul),
            bool(self.use_spiking_log),
            bool(self.use_spiking_expdiff),
        )
        if final_identity != identity:
            raise RuntimeError(
                "SpikingLayerNorm parameters or configuration changed while bounds "
                "were being frozen"
            )

        # Derived bounds are intentionally absent from the state dict. Checkpoint
        # compatibility remains unchanged, while later calls reuse immutable scalar
        # metadata until an explicit refresh establishes a new inference regime.
        frozen_bounds = (weight_domain, bias_domain, output_domain)
        self.__dict__["_frozen_parameter_bounds"] = (
            final_identity,
            frozen_bounds,
        )
        return frozen_bounds

    def _gaussian_forward(self, pot: Potential) -> Potential:
        """Evaluate LayerNorm with event-aware timing and fixed output bounds.

        The three ablation switches retain their existing meanings. Enabled
        spiking stages use the event-aware operators, while disabled stages use
        their direct PyTorch formulas. When logarithmic encoding is enabled but
        exponential-difference decoding is disabled, this method explicitly
        resolves the two causal rail masks at the shared observation deadline before
        applying the direct exponential formula. Every returned interval is
        retrieved from one frozen parameter/configuration contract; the current
        activation tensor is never measured to define it.

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

        # Freeze all learned-parameter and ablation-dependent intervals before any
        # event sampling occurs. Reusing this immutable contract keeps Gaussian
        # delivery masks from changing the declared output domain between calls.
        weight_domain, _bias_domain, output_domain = (
            self.freeze_parameter_bounds()
        )

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

            # Even though this branch has no sampled temporal stage, it shares the
            # same frozen metadata lifecycle as every other ablation combination.
            return Potential(out, output_domain)

        eps = self.eps
        clip_margin = self.clip_margin
        theta = self.theta
        tau_s = self.tau_s

        # LayerNorm first forms exact non-negative magnitudes for the two signed
        # rails. Only the later logarithmic carrier copies receive the strictly
        # positive encoder floor; inactive rails remain zero in the variance.
        x_err = x - x.mean(dim=-1, keepdim=True)
        magnitude_domain = PotentialBounds(0.0, theta - clip_margin)
        x_err_pos_magnitude = magnitude_domain.clamp(
            x_err.clamp_min(0.0), name="x_err_pos_magnitude"
        )
        x_err_neg_magnitude = magnitude_domain.clamp(
            (-x_err).clamp_min(0.0), name="x_err_neg_magnitude"
        )
        domain_err = PotentialBounds(clip_margin, theta - clip_margin)
        x_err_pos = domain_err.clamp(
            x_err_pos_magnitude, name="x_err_pos_log_carrier"
        )
        x_err_neg = domain_err.clamp(
            x_err_neg_magnitude, name="x_err_neg_log_carrier"
        )
        positive_active = x_err_pos_magnitude >= clip_margin
        negative_active = x_err_neg_magnitude >= clip_margin

        # The variance ablation changes only the squaring implementation. Gaussian
        # multiplication already performs its own sampled event readout and clamp.
        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(
                x_err_pos_magnitude,
                magnitude_domain,
                x_err_pos_magnitude,
                magnitude_domain,
                theta,
            )
            M_neg, _ = multiplication_operator(
                x_err_neg_magnitude,
                magnitude_domain,
                x_err_neg_magnitude,
                magnitude_domain,
                theta,
            )
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (
                x_err_pos_magnitude.pow(2) + x_err_neg_magnitude.pow(2)
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
            y_pos, _ = exponential_difference_operator(
                t_err_pos,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            y_neg, _ = exponential_difference_operator(
                t_err_neg,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            y_pos = torch.where(positive_active, y_pos, torch.zeros_like(y_pos))
            y_neg = torch.where(negative_active, y_neg, torch.zeros_like(y_neg))
            result = y_pos - y_neg

            # Event misses may leave a one-sided exponential excursion. Count and
            # saturate it at the finite-feature ideal normalization rail before the
            # learned affine multiplication.
            result_limit = math.sqrt(math.prod(self.normalized_shape))
            result_domain = PotentialBounds(-result_limit, result_limit)
            result = clamp_gaussian_output(
                result,
                result_domain,
                site="layernorm.normalized_output",
                name="layernorm_normalized",
            )

            # Retain the existing spiking affine rescaling used by this ablation.
            # The multiplier consumes the frozen gamma interval, while the final
            # module interval comes from the same pre-sampling metadata contract.
            scaled, _ = multiplication_operator(
                result,
                result_domain,
                self.weight,
                weight_domain,
                theta,
            )
            out = scaled + self.bias
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
                y_pos = torch.where(positive_active, y_pos, torch.zeros_like(y_pos))
                y_neg = torch.where(negative_active, y_neg, torch.zeros_like(y_neg))
            else:
                # With direct log tensors there are no miss masks to resolve, so the
                # original analytical exponential-difference formula remains exact.
                y_pos = torch.exp((t_sigma - t_err_pos) / tau_s)
                y_neg = torch.exp((t_sigma - t_err_neg) / tau_s)
                y_pos = torch.where(positive_active, y_pos, torch.zeros_like(y_pos))
                y_neg = torch.where(negative_active, y_neg, torch.zeros_like(y_neg))

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
            # The finite endpoint check above protects the actual activation dtype.
            # The corresponding signed interval and affine propagation were already
            # evaluated in ``freeze_parameter_bounds`` using stable scalar math.
            result = y_pos - y_neg
            result_limit = math.sqrt(math.prod(self.normalized_shape))
            result_domain = PotentialBounds(-result_limit, result_limit)
            result = clamp_gaussian_output(
                result,
                result_domain,
                site="layernorm.normalized_output",
                name="layernorm_normalized",
            )
            out = self.weight * result + self.bias

        # Every Gaussian ablation combination now returns the same immutable object
        # until parameters or bound-defining configuration are explicitly refreshed.
        return Potential(out, output_domain)
    
    def forward(self, pot: Potential) -> Potential:
        """Apply LayerNorm with configuration-derived output bounds.

        Gaussian timing noise is process-wide mutable state, so the decision is
        made once at the public method boundary. Event-aware execution delegates
        to :meth:`_gaussian_forward`; deterministic execution reuses the same frozen
        parameter and output-domain contract without reading runtime extrema.

        Args:
            pot: Input activation tensor paired with its calibrated bounds.

        Returns:
            A normalized ``Potential`` with bounds synchronized to its output.
        """
        # Keep sampled timestamps and delivery masks confined to the dedicated
        # implementation so deterministic tensor arithmetic remains event-free.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(pot)

        # The deterministic branch retains all three stage-ablation combinations,
        # but its metadata must be fixed before observing this invocation's output.
        x: torch.Tensor = pot.value

        # Freeze the learned affine intervals and active ablation envelope before
        # deterministic arithmetic begins. Repeated inference calls validate only
        # the cache identity and reuse the same immutable output-domain object.
        weight_domain, _bias_domain, output_domain = (
            self.freeze_parameter_bounds()
        )

        if not self.use_spiking_mul and not self.use_spiking_log and not self.use_spiking_expdiff:
            out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

            # The dense value remains PyTorch-exact, while its finite-feature affine
            # envelope comes from the precomputed contract shared with Gaussian mode.
            return Potential(out, output_domain)

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
        magnitude_domain = PotentialBounds(0.0, theta - clip_margin)
        x_err_pos_magnitude = magnitude_domain.clamp(
            x_err.clamp_min(0.0), name="x_err_pos_magnitude"
        )
        x_err_neg_magnitude = magnitude_domain.clamp(
            (-x_err).clamp_min(0.0), name="x_err_neg_magnitude"
        )
        domain_err: PotentialBounds = PotentialBounds(
            clip_margin,
            theta - clip_margin,
        )
        x_err_pos = domain_err.clamp(
            x_err_pos_magnitude, name="x_err_pos_log_carrier"
        )
        x_err_neg = domain_err.clamp(
            x_err_neg_magnitude, name="x_err_neg_log_carrier"
        )
        positive_active = x_err_pos_magnitude >= clip_margin
        negative_active = x_err_neg_magnitude >= clip_margin

        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(
                x_err_pos_magnitude,
                magnitude_domain,
                x_err_pos_magnitude,
                magnitude_domain,
                theta,
            )
            M_neg, _ = multiplication_operator(
                x_err_neg_magnitude,
                magnitude_domain,
                x_err_neg_magnitude,
                magnitude_domain,
                theta,
            )
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (
                x_err_pos_magnitude.pow(2) + x_err_neg_magnitude.pow(2)
            ).mean(dim=-1, keepdim=True)

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
            # Preserve both exponential output domains rather than recovering a rail
            # from their realized tensor difference after the operator has run.
            y_pos, _ = exponential_difference_operator(
                t_err_pos,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            y_neg, _ = exponential_difference_operator(
                t_err_neg,
                tb_err,
                t_sigma,
                tb_sigma,
                tau_s=tau_s,
            )
            y_pos = torch.where(positive_active, y_pos, torch.zeros_like(y_pos))
            y_neg = torch.where(negative_active, y_neg, torch.zeros_like(y_neg))
            result: torch.Tensor = y_pos - y_neg

            # Enforce the finite-feature normalization contract before affine
            # scaling; deterministic roundoff is clamped without creating stats.
            result_limit = math.sqrt(math.prod(self.normalized_shape))
            result_domain = PotentialBounds(-result_limit, result_limit)
            result = clamp_gaussian_output(
                result,
                result_domain,
                site="layernorm.normalized_output",
                name="layernorm_normalized",
            )

            # Propagate the signed interval through the learned scale using its
            # frozen gamma domain. The final module rail was derived from this same
            # ablation contract, so no parameter reduction is needed here.
            scaled, _ = multiplication_operator(
                result,
                result_domain,
                self.weight,
                weight_domain,
                theta,
            )
            out = scaled + self.bias
        else:
            y_pos = torch.exp((t_sigma - t_err_pos) / tau_s)
            y_neg = torch.exp((t_sigma - t_err_neg) / tau_s)
            y_pos = torch.where(positive_active, y_pos, torch.zeros_like(y_pos))
            y_neg = torch.where(negative_active, y_neg, torch.zeros_like(y_neg))
            result = y_pos - y_neg
            result_limit = math.sqrt(math.prod(self.normalized_shape))
            result_domain = PotentialBounds(-result_limit, result_limit)
            result = clamp_gaussian_output(
                result,
                result_domain,
                site="layernorm.normalized_output",
                name="layernorm_normalized",
            )
            out = self.weight * result + self.bias

            # Both delivered time tensors lie in their fixed declared windows. Form
            # the complete temporal-difference interval and transform its endpoints
            # monotonically through the direct exponential ablation.
            delta_min = float(tb_sigma.min) - float(tb_err.max)
            delta_max = float(tb_sigma.max) - float(tb_err.min)
            exponential_endpoints = torch.exp(
                x.new_tensor([delta_min / tau_s, delta_max / tau_s])
            )
            if not bool(torch.isfinite(exponential_endpoints).all()):
                raise ValueError(
                    "LayerNorm exponential bounds must be finite in the activation "
                    "dtype"
                )
            # This check protects representability in the activation dtype. Stable
            # scalar endpoint propagation and learned affine reduction were already
            # completed once by ``freeze_parameter_bounds``.

        # Noise configuration now changes only value-generation semantics; both
        # execution modes attach the exact same frozen metadata object.
        return Potential(out, output_domain)


def _validate_pwm_input_domain(
    domain: PotentialBounds,
    *,
    operator_name: str,
) -> tuple[float, float]:
    """Validate a fixed potential interval used by signed identity-code PWM.

    Signed PWM reconstructs an activation by subtracting its data-event time from a
    zero-reference-event time. The shared domain must therefore contain zero, and its
    finite endpoints must be strictly ordered so the identity encoder has a positive
    observation window.

    Args:
        domain: Upstream analytic or calibrated potential bounds.
        operator_name: Stable name included in validation diagnostics.

    Returns:
        The lower and upper endpoints as ordinary Python floats.
    """
    # Require the immutable domain type rather than accepting a pair whose mutation
    # could invalidate a memoized parameter-derived output rail.
    if not isinstance(domain, PotentialBounds):
        raise TypeError(f"{operator_name} input domain must be PotentialBounds")
    lower = float(domain.min)
    upper = float(domain.max)

    # Finiteness and strict ordering define a nonzero identity-code window. Including
    # zero guarantees that the physical reference event lies inside that same window.
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"{operator_name} input domain must have finite ordered endpoints"
        )
    if lower > 0.0 or upper < 0.0:
        raise ValueError(f"{operator_name} input domain must contain zero")
    return lower, upper


class SpikingLinear(nn.Linear):
    """Linear layer via ψ_PWM operator. Numerically identical to nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def freeze_parameter_bounds(
        self,
        input_domain: PotentialBounds,
        *,
        refresh: bool = False,
    ) -> PotentialBounds:
        """Memoize the affine output domain for one fixed input interval.

        For a scalar input interval ``[l, u]`` shared by every input feature, output
        feature ``j`` has the exact interval obtained by summing
        ``min(W_ji*l, W_ji*u)`` and ``max(W_ji*l, W_ji*u)`` before adding ``b_j``.
        This supports symmetric calibrated rails and one-sided zero-containing rails
        without rescanning parameters on repeated calls.

        Args:
            input_domain: Immutable analytic or calibrated input rail containing zero.
            refresh: Recompute the frozen domain after an intentional parameter
                update and discard entries for prior input domains. The default
                rejects parameter mutation after any bound has been memoized.

        Returns:
            The immutable module-wide affine output domain.

        Raises:
            RuntimeError: If parameters changed after memoization and ``refresh`` is
                false, or change during recomputation.
            ValueError: If the input domain is invalid or a derived endpoint is not
                finite.

        Notes:
            Mutation detection relies on PyTorch's parameter version counters.
            Standard ``torch.no_grad()`` in-place updates are detected; unsupported
            ``parameter.data`` writes bypass autograd bookkeeping and must not be
            used by checkpoint conversion or perturbation code.
        """
        # The fixed input interval is part of the memoization key. Zero containment is
        # a physical requirement of the shared reference event, not a statistical
        # property inferred from the activation reaching this invocation.
        lower_input, upper_input = _validate_pwm_input_domain(
            input_domain,
            operator_name="SpikingLinear",
        )
        domain_key = (lower_input, upper_input)

        # Parameter mutation counters define one cache generation. Multiple immutable
        # input domains may coexist within that generation without rescanning weights.
        versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")

        # A parameter update invalidates every domain entry together. Explicit refresh
        # begins a new coherent generation; otherwise stale physical rails are rejected.
        memoized_domains: dict[tuple[float, float], PotentialBounds]
        if cached is not None:
            cached_versions, memoized_domains = cached
            if versions != cached_versions and not refresh:
                raise RuntimeError(
                    "SpikingLinear parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            if (
                versions == cached_versions
                and not refresh
                and domain_key in memoized_domains
            ):
                return memoized_domains[domain_key]
            if refresh:
                memoized_domains = {}
        else:
            memoized_domains = {}

        # Evaluate interval arithmetic in float64 once for this domain. Positive and
        # negative weights select opposite input endpoints, retaining bias exactly and
        # avoiding the symmetric absolute-sum relaxation for one-sided rails.
        weight = self.weight.detach().to(dtype=torch.float64)
        lower_terms = torch.minimum(weight * lower_input, weight * upper_input)
        upper_terms = torch.maximum(weight * lower_input, weight * upper_input)
        if self.bias is None:
            bias = torch.zeros(
                self.out_features,
                dtype=torch.float64,
                device=weight.device,
            )
        else:
            bias = self.bias.detach().to(dtype=torch.float64)
        lower = lower_terms.sum(dim=1) + bias
        upper = upper_terms.sum(dim=1) + bias
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SpikingLinear parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower.min().item(), upper.max().item())

        # Detect concurrent parameter writes before publishing this entry. The input
        # domain is immutable, so parameter versions are the only mutable identity.
        final_versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        if final_versions != versions:
            raise RuntimeError(
                "SpikingLinear parameters changed while bounds were being frozen"
            )

        # Publish a fresh dictionary copy outside the state dict. Existing entries stay
        # immutable and checkpoint compatibility remains unchanged.
        memoized_domains = {**memoized_domains, domain_key: output_domain}
        self.__dict__["_frozen_parameter_bounds"] = (
            final_versions,
            memoized_domains,
        )
        return output_domain

    def _gaussian_forward(
        self,
        x: torch.Tensor,
        encoded_x: torch.Tensor,
        domain_x: PotentialBounds,
        output_domain: PotentialBounds,
    ) -> Potential:
        """Evaluate the affine layer from sampled data and reference events.

        ``encoded_x`` must already be clamped to the layer's fixed identity-code
        domain. Every data element and one scalar zero-reference event supply the two
        causal rails of a signed PWM readout. Each missed event leaves its own rail at
        reset, while the delivered rail remains observable at the fixed deadline.

        Args:
            x: Original input tensor used for dtype, device, and scalar allocation.
            encoded_x: Input tensor clamped to ``domain_x``.
            domain_x: Fixed zero-containing analytic or calibrated input rails.
            output_domain: Pre-frozen affine output rail shared with deterministic
                execution and already incorporating learned weight and bias bounds.

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
        #     self.weight[j, i], weight_domain,
        #     observation_deadline=float(data_event.domain.max),
        # )
        # y_j = sum_i(pwm_ji) + bias_j
        y = nn.functional.linear(signed_pulse_width, self.weight, self.bias)

        # Count raw affine saturation against the frozen ideal rail before passing
        # the clamped physical result to the next Transformer operation. A one-sided
        # event miss may exceed the delivered-event safety interval and is measured
        # here rather than widening the frozen domain.
        return Potential(
            clamp_gaussian_output(
                y,
                output_domain,
                site="linear.output",
                name="linear_y",
            ),
            output_domain,
        )

    def forward(self, input: Potential) -> Potential:
        """Apply the affine PWM map against one frozen parameter-derived rail.

        The identity-code domain comes from the upstream analytic or calibrated
        ``Potential``. An output-specific parameter interval is memoized for those
        immutable endpoints and shared by deterministic and Gaussian execution.
        Gaussian execution delegates sampled event readout to
        :meth:`_gaussian_forward`; noise-free execution evaluates the same PWM-MAC.

        Args:
            input: Tensor value paired with its upstream potential bounds.

        Returns:
            The affine output paired with conservative ideal potential rails.
        """
        # The upstream Potential owns the physical input rail. Validate zero
        # containment before clamping so both data and reference events can share it.
        x: torch.Tensor = input.value
        domain_x = input.domain
        _validate_pwm_input_domain(domain_x, operator_name="SpikingLinear")
        encoded_x = domain_x.clamp(x, name="linear_x")

        # Freeze the output-specific absolute-sum rail on first use and validate the
        # parameter mutation counters thereafter. Both execution modes receive this
        # exact immutable object, so a noise toggle cannot alter declared bounds.
        output_domain = self.freeze_parameter_bounds(domain_x)

        # Keep event sampling, shared-reference handling, and saturation statistics
        # isolated in the private Gaussian method. Passing only the frozen output
        # rail avoids every forward-time weight or bias endpoint reduction.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                output_domain,
            )

        # Encode delivered data and one scalar zero codeword on the same fixed window.
        # Their time difference recovers the clamped activation for symmetric and
        # one-sided domains alike without materializing per-synapse reference tensors.
        data_time, _ = neg_identity_transform(encoded_x, domain_x)
        reference_time, _ = neg_identity_transform(x.new_zeros(()), domain_x)
        signed_pulse_width = reference_time - data_time

        # The optimized kernel is algebraically identical to summing one explicit
        # signed PWM call per synapse:
        # pwm_ji, _ = signed_pulse_width_modulation_operator(
        #     data_time_i, data_time_domain,
        #     reference_time, data_time_domain,
        #     self.weight[j, i], weight_domain,
        #     observation_deadline=float(data_time_domain.max),
        # )
        # y_j = sum_i(pwm_ji) + bias_j.
        y = nn.functional.linear(signed_pulse_width, self.weight, self.bias)

        # The optimized kernel already includes learned bias. Return the previously
        # frozen affine rail directly; no current activation or parameter reduction
        # participates in deterministic forward-time metadata construction.
        return Potential(y, output_domain)

class SpikingConv2d(nn.Conv2d):
    """2D convolution via ψ_PWM operator. Numerically identical to nn.Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def freeze_parameter_bounds(
        self,
        input_domain: PotentialBounds,
        *,
        refresh: bool = False,
    ) -> PotentialBounds:
        """Memoize the convolution output domain for one fixed input interval.

        For input values in ``[l, u]``, each kernel element contributes
        ``min(w*l, w*u)`` to the lower endpoint and ``max(w*l, w*u)`` to the upper
        endpoint. Summing those terms over each grouped receptive field and adding
        bias gives an exact channel interval for the shared scalar input rail.

        Args:
            input_domain: Immutable analytic or calibrated input rail containing zero.
            refresh: Recompute after an intentional parameter update and discard
                entries for prior input domains. Without this flag, parameter
                mutation after the first memoization is rejected.

        Returns:
            The immutable module-wide convolution output domain.

        Raises:
            RuntimeError: If parameters changed after memoization without refresh,
                or changed while the new domain was being calculated.
            ValueError: If the input domain or derived endpoints are invalid.

        Notes:
            PyTorch parameter version counters detect normal ``torch.no_grad()``
            in-place updates. Direct ``parameter.data`` writes bypass this mechanism
            and are unsupported in checkpoint and perturbation code.
        """
        # The upstream fixed interval is the encoder rail and cache key. Requiring
        # zero containment also makes ordinary convolution padding a valid zero
        # potential and keeps the scalar reference event inside the same window.
        lower_input, upper_input = _validate_pwm_input_domain(
            input_domain,
            operator_name="SpikingConv2d",
        )
        domain_key = (lower_input, upper_input)

        # Parameter mutation establishes a new cache generation. Within a stable
        # generation, distinct immutable calibration rails can coexist by endpoint.
        versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")

        # Reject an unapproved parameter transition instead of pairing new kernels
        # with old physical rails. Explicit refresh clears every prior domain entry.
        memoized_domains: dict[tuple[float, float], PotentialBounds]
        if cached is not None:
            cached_versions, memoized_domains = cached
            if versions != cached_versions and not refresh:
                raise RuntimeError(
                    "SpikingConv2d parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            if (
                versions == cached_versions
                and not refresh
                and domain_key in memoized_domains
            ):
                return memoized_domains[domain_key]
            if refresh:
                memoized_domains = {}
        else:
            memoized_domains = {}

        # Evaluate interval arithmetic in float64. The stored kernel already contains
        # only the input channels belonging to its group, so dimensions 1-3 are the
        # complete receptive-field reduction and need no separate grouping factor.
        weight = self.weight.detach().to(dtype=torch.float64)
        lower_terms = torch.minimum(weight * lower_input, weight * upper_input)
        upper_terms = torch.maximum(weight * lower_input, weight * upper_input)
        if self.bias is None:
            bias = torch.zeros(
                self.out_channels,
                dtype=torch.float64,
                device=weight.device,
            )
        else:
            bias = self.bias.detach().to(dtype=torch.float64)
        lower = lower_terms.sum(dim=(1, 2, 3)) + bias
        upper = upper_terms.sum(dim=(1, 2, 3)) + bias
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SpikingConv2d parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower.min().item(), upper.max().item())

        # Refuse to publish a mixed-version interval if a concurrent writer changed
        # either tensor during the one-time reduction.
        final_versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        if final_versions != versions:
            raise RuntimeError(
                "SpikingConv2d parameters changed while bounds were being frozen"
            )

        # Publish a fresh dictionary outside the state dict. Existing immutable
        # entries remain reusable and checkpoint parameter keys stay unchanged.
        memoized_domains = {**memoized_domains, domain_key: output_domain}
        self.__dict__["_frozen_parameter_bounds"] = (
            final_versions,
            memoized_domains,
        )
        return output_domain

    def _gaussian_forward(
        self,
        x: torch.Tensor,
        encoded_x: torch.Tensor,
        domain_x: PotentialBounds,
        output_domain: PotentialBounds,
    ) -> Potential:
        """Evaluate convolution from sampled data and shared reference events.

        Every encoded input element and one scalar zero-reference event supply the
        two causal rails of a signed PWM readout. Their signed pulse width is passed
        directly to PyTorch's grouped convolution kernel, which accelerates the full
        PWM-MAC while preserving stride, padding, dilation, learned bias, and the
        checkpoint-compatible parameter layout.

        Args:
            x: Original input tensor used for metadata and scalar allocation.
            encoded_x: Input clamped to the fixed identity-code domain.
            domain_x: Fixed zero-containing analytic or calibrated input rails.
            output_domain: Frozen parameter-derived convolution output rail.

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
        #     self.weight[out_channel, in_channel, kh, kw], weight_domain,
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

        # Record saturation against the frozen output-specific safety rail before
        # returning the bounded potential. One-sided timing misses may leave a raw
        # value outside the ideal delivered-event range, but never widen its domain.
        return Potential(
            clamp_gaussian_output(
                y,
                output_domain,
                site="conv2d.output",
                name="conv2d_y",
            ),
            output_domain,
        )

    def forward(self, input: Potential) -> Potential:
        """Apply convolution through deterministic or Gaussian PWM integration.

        The method performs common input calibration, freezes an output-specific
        parameter rail, then dispatches to the event-aware or delivered-time PWM
        path. Both use the optimized convolution kernel and preserve stride, padding,
        dilation, grouping, bias, and identical output metadata.

        Args:
            input: Spatial activation tensor paired with upstream potential bounds.

        Returns:
            The convolution output paired with conservative ideal potential rails.
        """
        # The upstream Potential owns the fixed encoder rail. Zero containment also
        # preserves the physical meaning of ordinary zero padding outside the image.
        x: torch.Tensor = input.value
        domain_x = input.domain
        _validate_pwm_input_domain(domain_x, operator_name="SpikingConv2d")
        encoded_x = domain_x.clamp(x, name="conv2d_x")

        # Freeze grouped-kernel and bias bounds once after loading or perturbation.
        # Subsequent calls validate mutation counters and reuse this exact rail.
        output_domain = self.freeze_parameter_bounds(domain_x)

        # Keep event sampling, shared-reference readout, and output saturation logging
        # isolated in the private Gaussian method.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                output_domain,
            )

        # Encode one scalar zero codeword on the same window as every data value.
        # Subtracting times recovers the clamped potential for arbitrary zero-
        # containing intervals, while ordinary convolution padding remains zero.
        data_time, _ = neg_identity_transform(encoded_x, domain_x)
        reference_time, _ = neg_identity_transform(x.new_zeros(()), domain_x)
        signed_pulse_width = reference_time - data_time

        # The optimized grouped convolution evaluates the sum of the conceptually
        # equivalent signed PWM call at every receptive-field synapse:
        # pwm_synapse, _ = signed_pulse_width_modulation_operator(
        #     data_time_synapse, data_time_domain,
        #     reference_time, data_time_domain,
        #     weight_synapse, weight_domain,
        #     observation_deadline=float(data_time_domain.max),
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

        # Bias is already included by conv2d, and its contribution is already stored
        # in the frozen rail. No parameter reduction occurs during this forward pass.
        return Potential(y, output_domain)


def freeze_dense_layer_norm_bounds(
    norm: nn.LayerNorm,
    *,
    refresh: bool = False,
) -> PotentialBounds:
    """Freeze the analytic output range of an ordinary PyTorch LayerNorm.

    Population normalization over ``d`` features satisfies
    ``abs(z_i) <= sqrt(d - 1)``. This setup function propagates that fixed interval
    through the learned featurewise scale and bias once, then stores one scalar
    module-wide range outside the state dict for all later forward calls.

    Args:
        norm: Ordinary ``torch.nn.LayerNorm`` whose parameters are already loaded and
            placed at their inference dtype.
        refresh: Recompute after an intentional parameter, dtype, or configuration
            change. The default rejects a stale cache.

    Returns:
        Immutable analytic output bounds for the current parameter version.

    Raises:
        TypeError: If ``norm`` is not an ordinary PyTorch LayerNorm or ``refresh`` is
            not Boolean.
        RuntimeError: If cached parameters or configuration changed without refresh,
            or parameters change while the new range is being reduced.
        ValueError: If epsilon, parameters, or derived endpoints are non-finite.
    """
    # Keep this setup path distinct from SpikingLayerNorm, which has its own ablation-
    # aware parameter-bound cache. Boolean validation prevents truthy numeric aliases
    # from accidentally discarding a valid cache.
    if not isinstance(norm, nn.LayerNorm) or isinstance(norm, SpikingLayerNorm):
        raise TypeError("norm must be an ordinary torch.nn.LayerNorm")
    if not isinstance(refresh, bool):
        raise TypeError("refresh must be a bool")
    if not math.isfinite(float(norm.eps)) or float(norm.eps) < 0.0:
        raise ValueError("LayerNorm epsilon must be finite and non-negative")

    # Parameter object identity, version, dtype, and analytic configuration define
    # the cache. Dtype is explicit because an in-place model precision conversion may
    # round endpoints even if a framework version counter does not expose that write.
    weight = norm.weight
    bias = norm.bias
    identity = (
        tuple(norm.normalized_shape),
        float(norm.eps),
        bool(norm.elementwise_affine),
        id(weight) if weight is not None else None,
        weight._version if weight is not None else None,
        weight.dtype if weight is not None else None,
        id(bias) if bias is not None else None,
        bias._version if bias is not None else None,
        bias.dtype if bias is not None else None,
    )
    cached = norm.__dict__.get("_delayed_temporal_frozen_output_bounds")
    if cached is not None and not refresh:
        cached_identity, cached_bounds = cached
        if cached_identity != identity:
            raise RuntimeError(
                "LayerNorm parameters or configuration changed after bounds were "
                "frozen; call freeze_dense_layer_norm_bounds(refresh=True)"
            )
        return cached_bounds

    # The no-affine path needs no parameter reduction. For the affine path, float64
    # setup arithmetic prevents a low-precision checkpoint from rounding a true
    # endpoint inward while scale signs select opposite normalized endpoints.
    feature_count = math.prod(norm.normalized_shape)
    normalized_limit = math.sqrt(max(feature_count - 1, 0))
    if not norm.elementwise_affine:
        output_domain = PotentialBounds(-normalized_limit, normalized_limit)
    else:
        if weight is None:
            raise RuntimeError("affine LayerNorm must define a weight parameter")
        frozen_weight = weight.detach().to(dtype=torch.float64)
        frozen_bias: torch.Tensor | float = (
            bias.detach().to(dtype=torch.float64) if bias is not None else 0.0
        )
        if not bool(torch.isfinite(frozen_weight).all()):
            raise ValueError("LayerNorm weight must be finite")
        if isinstance(frozen_bias, torch.Tensor) and not bool(
            torch.isfinite(frozen_bias).all()
        ):
            raise ValueError("LayerNorm bias must be finite")
        lower_candidate = frozen_weight * -normalized_limit + frozen_bias
        upper_candidate = frozen_weight * normalized_limit + frozen_bias
        lower = torch.minimum(lower_candidate, upper_candidate).min().item()
        upper = torch.maximum(lower_candidate, upper_candidate).max().item()
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("LayerNorm parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower, upper)

    # Rebuild the identity after reduction so a concurrent parameter mutation cannot
    # publish an interval assembled from mixed versions. Store ordinary attributes to
    # keep checkpoints and pretrained state-dict keys unchanged.
    final_identity = (
        tuple(norm.normalized_shape),
        float(norm.eps),
        bool(norm.elementwise_affine),
        id(norm.weight) if norm.weight is not None else None,
        norm.weight._version if norm.weight is not None else None,
        norm.weight.dtype if norm.weight is not None else None,
        id(norm.bias) if norm.bias is not None else None,
        norm.bias._version if norm.bias is not None else None,
        norm.bias.dtype if norm.bias is not None else None,
    )
    if final_identity != identity:
        raise RuntimeError("LayerNorm parameters changed while bounds were frozen")
    norm.__dict__["_delayed_temporal_frozen_output_bounds"] = (
        final_identity,
        output_domain,
    )
    return output_domain


def _apply_norm(norm: nn.Module, pot: Potential) -> Potential:
    """Apply a supported LayerNorm while preserving a static potential domain.

    Spiking LayerNorm owns its complete operator-specific bound propagation. An
    ordinary PyTorch LayerNorm instead uses the finite-feature population-normalized
    envelope and propagates its learned affine parameters over both endpoints. This
    helper never derives metadata from the normalized tensor produced in this call.

    Args:
        norm: A ``SpikingLayerNorm`` or ``torch.nn.LayerNorm`` module.
        pot: Input tensor and its upstream potential bounds.

    Returns:
        The normalized tensor paired with a configuration- and parameter-derived
        output domain.

    Raises:
        TypeError: If ``norm`` is not one of the supported LayerNorm modules.
    """
    # SpikingLayerNorm already propagates every enabled ablation stage and learned
    # affine transformation, so preserve that single source of domain semantics.
    if isinstance(norm, SpikingLayerNorm):
        return norm(pot)

    # A generic nn.Module has no known analytic normalization envelope. Reject it
    # explicitly instead of observing its current output and silently presenting a
    # batch-specific interval as a static physical range.
    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(
            "_apply_norm supports only SpikingLayerNorm or torch.nn.LayerNorm"
        )

    # Compute the ordinary LayerNorm value without changing PyTorch semantics. Its
    # analytic and parameter-derived output rail is established by the explicit cache
    # function and reused without a parameter reduction on subsequent forwards.
    out = norm(pot.value)
    output_domain = freeze_dense_layer_norm_bounds(norm)

    # Return the unchanged dense result with the frozen analytic envelope. Neither
    # current activations nor learned parameters are reduced in this forward helper.
    return Potential(out, output_domain)


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
