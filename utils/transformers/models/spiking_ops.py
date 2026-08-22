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

    def freeze_parameter_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[PotentialBounds, PotentialBounds, PotentialBounds]:
        """Freeze learned parameter and final output bounds for this ablation.

        LayerNorm has three distinct affine-input envelopes. A fully dense module
        uses ``sqrt(d - 1)`` for population-normalized features. A direct exponential
        path uses ``R - 1/R``, and a spiking exponential-difference path uses the
        relaxed magnitude ``R``, where ``R = (theta-clip_margin)/clip_margin``.
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

        # Select the fixed pre-affine magnitude for the active ablation. The spiking
        # expdiff branch deliberately preserves its existing relaxed [-R,R] contract,
        # while direct exponential subtraction can use the tighter R-1/R interval.
        all_dense = not (
            self.use_spiking_mul
            or self.use_spiking_log
            or self.use_spiking_expdiff
        )
        ratio = (theta - margin) / margin
        if all_dense:
            feature_count = math.prod(self.normalized_shape)
            result_limit = math.sqrt(max(feature_count - 1, 0))
            effective_weight = weight
        elif self.use_spiking_expdiff:
            result_limit = ratio
            effective_weight = weight.clamp(-theta, theta)
        else:
            result_limit = ratio - 1.0 / ratio
            effective_weight = weight
        if not math.isfinite(result_limit) or result_limit < 0.0:
            raise ValueError("SpikingLayerNorm normalized bound must be finite")

        # Dense and direct branches apply gamma featurewise, giving an exact global
        # endpoint reduction. The spiking final multiplication currently propagates
        # one global gamma interval, so preserve that broader operator-level contract
        # to contain Gaussian factor-event excursions before final bias addition.
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
            # The finite endpoint check above protects the actual activation dtype.
            # The corresponding signed interval and affine propagation were already
            # evaluated in ``freeze_parameter_bounds`` using stable scalar math.
            result = y_pos - y_neg
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
            # Preserve both exponential output domains rather than recovering a rail
            # from their realized tensor difference after the operator has run.
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
            result: torch.Tensor = y_pos - y_neg

            # The dual-rail exponential magnitudes are non-negative. Relax their
            # signed difference to one symmetric interval whose magnitude is the
            # larger declared upper rail, matching the event-aware construction.
            result_limit = max(domain_y_pos.max, domain_y_neg.max)
            result_domain = PotentialBounds(-result_limit, result_limit)

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
            result = y_pos - y_neg
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


class SpikingLinear(nn.Linear):
    """Linear layer via ψ_PWM operator. Numerically identical to nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def freeze_parameter_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> PotentialBounds:
        """Freeze the parameter-derived affine output domain for inference.

        The first call is intended to occur after checkpoint loading, dtype/device
        conversion, and any static weight or bias perturbation. For the symmetric
        input rail ``[-theta, theta]``, each output feature ``j`` has the exact
        parameter-derived interval ``[b_j-r_j, b_j+r_j]``, where
        ``r_j = theta * sum_i(abs(W_ji))``. Reducing those featurewise endpoints is
        tighter than multiplying global weight extrema by the input fan-in.

        Args:
            refresh: Recompute the frozen domain after an intentional parameter
                update. The default rejects mutations made after the first freeze.

        Returns:
            The immutable module-wide affine output domain.

        Raises:
            RuntimeError: If parameters or ``theta`` changed after freezing and
                ``refresh`` is false, or change during recomputation.
            ValueError: If ``theta`` or a derived endpoint is non-finite, or if
                ``theta`` is not positive.

        Notes:
            Mutation detection relies on PyTorch's parameter version counters.
            Standard ``torch.no_grad()`` in-place updates are detected; unsupported
            ``parameter.data`` writes bypass autograd bookkeeping and must not be
            used by checkpoint conversion or perturbation code.
        """
        # Validate and capture the configured symmetric input magnitude before cache
        # lookup. Changing theta changes both identity-code rails and affine bounds,
        # so it belongs to the same frozen configuration identity as parameters.
        if isinstance(self.theta, bool):
            raise ValueError("SpikingLinear theta must be finite and positive")
        theta = float(self.theta)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ValueError("SpikingLinear theta must be finite and positive")

        # Parameter mutation counters let repeated forward calls reuse scalar bounds
        # without rescanning tensors. Bias-less layers use a stable sentinel, and the
        # validated theta token detects threshold changes after the initial freeze.
        versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
            theta,
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")

        # A valid cache is returned verbatim. If checkpoint loading, perturbation, or
        # optimization changed a parameter after freezing, fail instead of silently
        # pairing new values with stale physical rails.
        if cached is not None and not refresh:
            cached_versions, cached_domain = cached
            if versions != cached_versions:
                raise RuntimeError(
                    "SpikingLinear parameters or theta changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            return cached_domain

        # Accumulate rounded checkpoint parameters in float64 once, avoiding both
        # low-precision reduction overflow and the old global-extrema-times-fan-in
        # relaxation. The operation stays on the parameter device and returns only
        # two scalar endpoints to Python.
        weight = self.weight.detach()
        radius = weight.abs().sum(dim=1, dtype=torch.float64) * theta
        if self.bias is None:
            bias = torch.zeros_like(radius)
        else:
            bias = self.bias.detach().to(dtype=torch.float64)
        lower = bias - radius
        upper = bias + radius
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SpikingLinear parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower.min().item(), upper.max().item())

        # Detect a concurrent write across the reduction boundary before publishing
        # the cache. Intentional static perturbation can explicitly refresh afterward;
        # an unnoticed race must never establish a mixed-version bound.
        final_versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
            float(self.theta),
        )
        if final_versions != versions:
            raise RuntimeError(
                "SpikingLinear parameters changed while bounds were being frozen"
            )

        # Store an immutable PotentialBounds object outside the state dict. Checkpoint
        # compatibility is unchanged, while every later inference call can reuse the
        # same scalar rail until an explicit refresh authorizes new parameters.
        self.__dict__["_frozen_parameter_bounds"] = (
            final_versions,
            output_domain,
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

        ``encoded_x`` must already be clamped to the layer's symmetric identity-code
        domain. Every data element and one scalar zero-reference event supply the two
        causal rails of a signed PWM readout. Each missed event leaves its own rail at
        reset, while the delivered rail remains observable at the fixed deadline.

        Args:
            x: Original input tensor used for dtype, device, and scalar allocation.
            encoded_x: Input tensor clamped to ``domain_x``.
            domain_x: Symmetric input rails ``[-theta, theta]``.
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

        The symmetric identity-code domain is fixed by ``theta``. The affine output
        domain is frozen on first use after checkpoint loading and static parameter
        perturbation, then shared by deterministic and Gaussian execution. Gaussian
        execution delegates sampled event readout to :meth:`_gaussian_forward`;
        noise-free execution evaluates the same PWM-MAC with the optimized kernel.

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

        # Freeze the output-specific absolute-sum rail on first use and validate the
        # parameter mutation counters thereafter. Both execution modes receive this
        # exact immutable object, so a noise toggle cannot alter declared bounds.
        output_domain = self.freeze_parameter_bounds()

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
        *,
        refresh: bool = False,
    ) -> PotentialBounds:
        """Freeze an output-specific convolution safety rail for inference.

        For input values in ``[-theta, theta]``, output channel ``j`` has radius
        ``theta * sum(abs(kernel_j))`` over its grouped receptive field. Optional
        bias translates that channel interval before all channel endpoints are
        reduced to the scalar domain carried by ``Potential``. The bound is frozen
        after loading and static perturbation, not rebuilt from every forward pass.

        Args:
            refresh: Recompute after an intentional parameter update. Without this
                flag, mutation after the first freeze is rejected.

        Returns:
            The immutable module-wide convolution output domain.

        Raises:
            RuntimeError: If parameters or ``theta`` changed after freezing without
                refresh, or changed while the new domain was being calculated.
            ValueError: If ``theta`` is invalid or the derived endpoints are not
                finite.

        Notes:
            PyTorch parameter version counters detect normal ``torch.no_grad()``
            in-place updates. Direct ``parameter.data`` writes bypass this mechanism
            and are unsupported in checkpoint and perturbation code.
        """
        # Validate and capture theta before cache lookup. The convolution's symmetric
        # input rail changes with theta even when its kernel stays byte-identical.
        if isinstance(self.theta, bool):
            raise ValueError("SpikingConv2d theta must be finite and positive")
        theta = float(self.theta)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ValueError("SpikingConv2d theta must be finite and positive")

        # Capture both parameter versions and the threshold configuration before
        # reading tensor values. Bias-less convolutions retain a stable None sentinel.
        versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
            theta,
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")

        # Reuse a valid immutable rail without scanning any kernel values. A changed
        # version must fail clearly rather than combine perturbed weights with stale
        # physical output bounds.
        if cached is not None and not refresh:
            cached_versions, cached_domain = cached
            if versions != cached_versions:
                raise RuntimeError(
                    "SpikingConv2d parameters or theta changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            return cached_domain

        # Sum each output channel's actual grouped kernel in float64. The stored
        # kernel already contains only the input channels belonging to its group, so
        # reducing dimensions 1-3 handles grouping without a separate fan-in factor.
        weight = self.weight.detach()
        radius = weight.abs().sum(dim=(1, 2, 3), dtype=torch.float64) * theta
        if self.bias is None:
            bias = torch.zeros_like(radius)
        else:
            bias = self.bias.detach().to(dtype=torch.float64)
        lower = bias - radius
        upper = bias + radius
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SpikingConv2d parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower.min().item(), upper.max().item())

        # Refuse to publish a mixed-version interval if a concurrent writer changed
        # either tensor during the one-time reduction.
        final_versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
            float(self.theta),
        )
        if final_versions != versions:
            raise RuntimeError(
                "SpikingConv2d parameters changed while bounds were being frozen"
            )

        # Keep the cache out of the state dict so checkpoint keys remain compatible.
        # Repeated inference calls reuse this exact immutable PotentialBounds object.
        self.__dict__["_frozen_parameter_bounds"] = (
            final_versions,
            output_domain,
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
            encoded_x: Input clamped to the symmetric identity-code domain.
            domain_x: Symmetric input rails ``[-theta, theta]``.
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
        # Convolution uses its own symmetric encoder calibration, matching the
        # pretrained conversion path independently of the upstream Potential domain.
        x: torch.Tensor = input.value
        domain_x = PotentialBounds(-self.theta, self.theta)
        encoded_x = domain_x.clamp(x, name="conv2d_x")

        # Freeze grouped-kernel and bias bounds once after loading or perturbation.
        # Subsequent calls validate mutation counters and reuse this exact rail.
        output_domain = self.freeze_parameter_bounds()

        # Keep event sampling, shared-reference readout, and output saturation logging
        # isolated in the private Gaussian method.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                output_domain,
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

        # Bias is already included by conv2d, and its contribution is already stored
        # in the frozen rail. No parameter reduction occurs during this forward pass.
        return Potential(y, output_domain)


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

    # Compute the ordinary LayerNorm value without changing PyTorch semantics.
    # Population normalization over d elements satisfies |z_i| <= sqrt(d-1), and
    # non-negative epsilon can only contract this conservative endpoint interval.
    out = norm(pot.value)
    feature_count = math.prod(norm.normalized_shape)
    normalized_limit = math.sqrt(max(feature_count - 1, 0))

    # LayerNorm without an affine stage returns the normalized value directly. Its
    # symmetric analytic envelope is independent of input ordering and batch size.
    if not norm.elementwise_affine:
        return Potential(
            out,
            PotentialBounds(-normalized_limit, normalized_limit),
        )

    # Apply each learned scale and optional bias to both endpoints. Featurewise
    # minima and maxima handle negative gamma, while the final reduction produces
    # the scalar module-wide domain expected by Potential.
    weight = norm.weight.detach()
    bias: torch.Tensor | float = (
        norm.bias.detach() if norm.bias is not None else 0.0
    )
    lower_candidate = weight * -normalized_limit + bias
    upper_candidate = weight * normalized_limit + bias
    output_domain = PotentialBounds(
        torch.minimum(lower_candidate, upper_candidate).min().item(),
        torch.maximum(lower_candidate, upper_candidate).max().item(),
    )

    # Return the unchanged dense result with a predeclared analytic envelope; no
    # extrema from ``out`` participate in the physical domain contract.
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
