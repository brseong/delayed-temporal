"""Run ViT GELU-internal operator ablations without changing production noise scope."""

from __future__ import annotations

import argparse
from math import isfinite, log
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.evaluation.error_analysis_vit import (
    Arguments,
    evaluate_vit_model,
    parse_arguments as parse_vit_arguments,
)
from utils.transformers.models.spiking_vit import modeling_spiking_vit
from utils.transforms.functions import (
    division_function,
    exponential_function,
    multiplication_operator,
)
from utils.transforms.noise import (
    _sample_gaussian_spike_time,
    get_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds, TimeBounds


_OPERATOR_NAMES = frozenset({"multiplication", "exponential", "division"})


def _consume_shadow_gaussian_event(
    nominal_time: torch.Tensor,
    domain: TimeBounds,
) -> None:
    """Advance the replica RNG for one bypassed event without recording a sample.

    Atomic-operator variants are compared with common random numbers. A selected
    dense operator must therefore consume exactly the draw that its event-aware
    counterpart would have used, even though the sampled timestamp is deliberately
    excluded from the operator result. Statistics omit these shadow events because
    they are not physically applied in that ablation condition.
    """
    config = get_gaussian_time_noise()

    # Noise-off parity calls own no generator and need no shadow event. Returning
    # here also preserves the zero-noise production contract outside experiments.
    if not config.enabled:
        return
    if not isinstance(config.generator, torch.Generator):
        raise RuntimeError("enabled Gaussian ablation requires a generator")

    # Use the maintained sampler rather than a raw randn call so dtype, device,
    # broadcasting, and the zero-standard-deviation RNG contract remain identical.
    _sample_gaussian_spike_time(
        nominal_time,
        time_std=config.time_std,
        domain=domain,
        generator=config.generator,
        time_mean=config.time_mean,
    )


def _dense_gelu_multiplication(
    value: torch.Tensor,
    value_domain: PotentialBounds,
    encoded_factor: torch.Tensor,
    factor_domain: PotentialBounds,
    *,
    theta: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate one GELU multiplication densely while preserving temporal rails.

    The production operator encodes its second operand against the symmetric
    ``[-theta, theta]`` physical interval while deriving ideal product bounds from
    the caller factor domain clipped to that interval. This bypass reproduces both
    the encoded carrier and the final rail clamp so only sampled events differ.

    Args:
        value: Potential supplying the multiplication drive.
        value_domain: Declared bounds of the drive.
        encoded_factor: Operand normally converted to a spike time.
        factor_domain: Declared factor bounds used for ideal product propagation.
        theta: Symmetric identity-code rail used by production multiplication.

    Returns:
        The direct tensor product and production-equivalent ideal output rails.
    """
    # Match the encoder-side clamp before constructing its nominal spike time. This
    # is essential when an intermediate GELU value exceeds the calibrated rail.
    factor = encoded_factor.clamp(min=-theta, max=theta)
    ideal_factor_domain = PotentialBounds(
        min(max(float(factor_domain.min), -theta), theta),
        min(max(float(factor_domain.max), -theta), theta),
    )

    # Reproduce the negative-linear encoder in the payload dtype. Computing the
    # product directly would also erase theta-scale float32 codeword rounding and
    # would therefore confound timing-noise removal with a precision improvement.
    encoder_endpoints = factor.new_tensor([-theta, theta])
    encoder_width = encoder_endpoints[1] - encoder_endpoints[0]
    window = factor.new_tensor(2.0 * theta)
    normalized_time = 1.0 - (
        (factor - encoder_endpoints[0]) / encoder_width
    )
    opening_time = window * normalized_time.clamp(min=0.0, max=1.0)

    # Consume the tensor opening draw followed by the scalar shared-reference draw,
    # matching the event-aware multiplication call's exact generator order.
    time_domain = TimeBounds(0.0, 2.0 * theta)
    _consume_shadow_gaussian_event(opening_time, time_domain)
    _consume_shadow_gaussian_event(opening_time.new_tensor(theta), time_domain)

    # Use the same scalar nominal zero-reference time and PWM arithmetic order as
    # the production noise-off operator, but do not draw either Gaussian event.
    result = value * (theta - opening_time)

    # Reuse the caller-derived factor interval and apply the production final clamp.
    candidates = (
        value_domain.min * ideal_factor_domain.min,
        value_domain.min * ideal_factor_domain.max,
        value_domain.max * ideal_factor_domain.min,
        value_domain.max * ideal_factor_domain.max,
    )
    result_domain = PotentialBounds(min(candidates), max(candidates))
    return (
        result_domain.clamp(result, name="gelu_ablation_multiplication_result"),
        result_domain,
    )


def _dense_gelu_exponential(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate the GELU tanh exponential densely with identical endpoint bounds.

    GELU's tanh construction requests the normalized exponential composition, whose
    deterministic value is exp(-x/tau_s). Computing the same endpoint exponents in
    the payload dtype retains float overflow and underflow behavior while removing
    only the exponential input event and its deadline decision.
    """
    # Validate the scale locally because this bypass does not enter the production
    # exponential function, which normally owns this input contract.
    if not isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError("tau_s must be finite and positive")

    # Evaluate declared endpoints beside the payload so the analysis branch retains
    # the same dtype-dependent representability as the temporal operator.
    endpoint_exponents = input_value.new_tensor(
        [-float(domain.max) / tau_s, -float(domain.min) / tau_s]
    )
    endpoint_values = torch.exp(endpoint_exponents)
    if not bool(
        (
            torch.isfinite(endpoint_values)
            & (endpoint_values > 0.0)
        ).all()
    ):
        raise ValueError("dense GELU exponential endpoints must be representable")

    # Reproduce the nominal negative-identity carrier in the payload dtype. This
    # retains the encoder's finite-precision rounding while omitting its noise draw.
    domain_endpoints = input_value.new_tensor(
        [float(domain.min), float(domain.max)]
    )
    domain_width = domain_endpoints[1] - domain_endpoints[0]
    window = input_value.new_tensor(float(domain.range))
    normalized_time = 1.0 - (
        (input_value - domain_endpoints[0]) / domain_width
    )
    nominal_time = window * normalized_time.clamp(min=0.0, max=1.0)

    # Preserve the run-wide stream position for the bypassed exponential input. The
    # sampled carrier is intentionally discarded before the nominal decoder below.
    _consume_shadow_gaussian_event(
        nominal_time,
        TimeBounds(0.0, float(domain.range)),
    )

    # Apply the same offset-adjusted decoder as the production deterministic path.
    # Directly evaluating exp(-x/tau_s) would remove carrier-rounding error too.
    return (
        torch.exp((nominal_time - float(domain.max)) / tau_s),
        PotentialBounds(endpoint_values[0].item(), endpoint_values[1].item()),
    )


def _dense_gelu_division(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    joint_domain: PotentialBounds,
    *,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate the GELU tanh ratio densely while retaining Gaussian-path rails.

    The ratio bypass removes numerator, denominator, and internal exponential-
    difference events as one atomic division operator. Its output retains the
    production reset-inclusive ordered-ratio rail ``[0, 1]`` and final clamp.
    """
    # Validate the scale and shared logarithmic interval locally because the bypass
    # does not enter either public encoder validation boundary.
    if not isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError("tau_s must be finite and positive")
    if (
        not isfinite(float(joint_domain.min))
        or not isfinite(float(joint_domain.max))
        or float(joint_domain.min) <= 0.0
        or float(joint_domain.min) >= float(joint_domain.max)
    ):
        raise ValueError("GELU division requires a finite positive joint domain")

    # Apply the same shared positive-domain clamp and ordering rule as the public
    # division function before constructing its two nominal logarithmic events.
    numerator = joint_domain.clamp(numerator, name="gelu_ablation_division_X")
    denominator = joint_domain.clamp(
        denominator,
        name="gelu_ablation_division_Y",
    )
    if not bool((numerator <= denominator).all()):
        raise ValueError("GELU division requires numerator <= denominator")

    # Construct both negative-log carriers with the exact production arithmetic
    # order. Keeping these nominal codewords avoids crediting the ablation for also
    # eliminating float32 logarithmic timing quantization.
    domain_max = numerator.new_tensor(float(joint_domain.max))
    numerator_time = tau_s * (
        torch.log(domain_max) - torch.log(numerator)
    )
    denominator_time = tau_s * (
        torch.log(domain_max) - torch.log(denominator)
    )
    deadline = tau_s * (
        log(float(joint_domain.max)) - log(float(joint_domain.min))
    )
    numerator_time = numerator_time.clamp(min=0.0, max=deadline)
    denominator_time = denominator_time.clamp(min=0.0, max=deadline)

    # Division normally draws numerator and denominator log events independently in
    # this order. Shadow both before constructing the nominal temporal difference.
    logarithmic_domain = TimeBounds(0.0, deadline)
    _consume_shadow_gaussian_event(numerator_time, logarithmic_domain)
    _consume_shadow_gaussian_event(denominator_time, logarithmic_domain)

    # Reproduce the nominal exponential-difference composition: integrate the unit
    # negative drive, encode the bounded intermediate, then decode its shifted time.
    intermediate = -(denominator_time - numerator_time)
    intermediate_domain = PotentialBounds(-deadline, deadline)
    intermediate_endpoints = numerator.new_tensor(
        [intermediate_domain.min, intermediate_domain.max]
    )
    intermediate_width = intermediate_endpoints[1] - intermediate_endpoints[0]
    internal_window = numerator.new_tensor(2.0 * deadline)
    internal_time = internal_window * (
        1.0
        - (
            (intermediate - intermediate_endpoints[0])
            / intermediate_width
        )
    ).clamp(min=0.0, max=1.0)

    # The exponential-difference stage re-encodes its intermediate potential once.
    # Its random draw is also shadowed so later non-GELU sites stay seed-paired.
    _consume_shadow_gaussian_event(
        internal_time,
        TimeBounds(0.0, 2.0 * deadline),
    )
    result = torch.exp(
        (internal_time - float(intermediate_domain.max)) / tau_s
    )

    # Match the public ordered-division contract, including its final rail clamp.
    result_domain = PotentialBounds(0.0, 1.0)
    return (
        result_domain.clamp(result, name="gelu_ablation_division_result"),
        result_domain,
    )


def gelu_operator_ablation(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    dense_operators: frozenset[str],
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_: object,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate cubic-tanh GELU with selected atomic operators computed densely.

    This analysis-only implementation preserves the production GELU formula and
    operator order. Selecting multiplication replaces every GELU-local product,
    selecting exponential replaces tanh's exp(-2z), and selecting division replaces
    its complete log-time ratio including internal exponential difference. Any
    unselected operator remains on the ordinary event-aware Gaussian path.

    Args:
        input_value: GELU input tensor from the first MLP affine layer.
        domain: Propagated input bounds used by the production composition.
        dense_operators: Atomic operator names bypassed only inside this GELU call.
        tau_s: Shared temporal scale used by tanh exponential and division.
        theta: Symmetric multiplication encoder rail.

    Returns:
        The ablated GELU value and its propagated production-compatible bounds.
    """
    unknown = dense_operators - _OPERATOR_NAMES
    if unknown:
        raise ValueError(f"unknown GELU operator ablations: {sorted(unknown)}")

    # Route each product through either the ordinary sampled operator or the direct
    # rail-preserving counterpart without mutating process-wide noise state.
    def multiply(
        value: torch.Tensor,
        value_domain: PotentialBounds,
        factor: torch.Tensor,
        factor_domain: PotentialBounds,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        if "multiplication" in dense_operators:
            return _dense_gelu_multiplication(
                value,
                value_domain,
                factor,
                factor_domain,
                theta=theta,
            )
        return multiplication_operator(
            value,
            value_domain,
            factor,
            factor_domain,
            theta,
        )

    # Reproduce x^2 and x^3 through the same two multiplication sites used by the
    # maintained cubic approximation.
    input_clamped = domain.clamp(input_value, name="gelu_ablation_x")
    x2, domain_x2 = multiply(
        input_clamped,
        domain,
        input_clamped,
        domain,
    )
    x3, domain_x3 = multiply(x2, domain_x2, input_clamped, domain)

    # Apply both fixed polynomial coefficients through the selected multiplication
    # path so the multiplication ablation covers constant as well as data factors.
    cubic_coefficient = input_value.new_tensor(0.044715).expand_as(input_value)
    x3_scaled, domain_x3_scaled = multiply(
        x3,
        domain_x3,
        cubic_coefficient,
        PotentialBounds(0.044715, 0.044715),
    )
    inner_domain = PotentialBounds(
        domain.min + domain_x3_scaled.min,
        domain.max + domain_x3_scaled.max,
    )
    inner = inner_domain.clamp(
        input_clamped + x3_scaled,
        name="gelu_ablation_inner",
    )
    tanh_scale = input_value.new_tensor(0.7978845608028654).expand_as(input_value)
    tanh_input, tanh_input_domain = multiply(
        inner,
        inner_domain,
        tanh_scale,
        PotentialBounds(0.7978845608028654, 0.7978845608028654),
    )

    # Expand tanh as 2/(1+exp(-2z))-1. Keeping this decomposition local permits
    # exponential and division ablations without changing the shared tanh operator.
    two = input_value.new_tensor(2.0).expand_as(input_value)
    scaled_tanh_input, scaled_tanh_domain = multiply(
        tanh_input,
        tanh_input_domain,
        two,
        PotentialBounds(2.0, 2.0),
    )
    stability_cap = 80.0
    scaled_tanh_input = scaled_tanh_input.clamp(
        min=-stability_cap,
        max=stability_cap,
    )
    scaled_tanh_domain = PotentialBounds(
        max(scaled_tanh_domain.min, -stability_cap),
        min(scaled_tanh_domain.max, stability_cap),
    )

    # The exponential bypass removes only its encoder sample; otherwise the normal
    # event-aware implementation retains reset-on-opening-miss behavior.
    if "exponential" in dense_operators:
        negative_exponential, negative_exponential_domain = (
            _dense_gelu_exponential(
                scaled_tanh_input,
                scaled_tanh_domain,
                tau_s=tau_s,
            )
        )
    else:
        negative_exponential, negative_exponential_domain = exponential_function(
            scaled_tanh_input,
            scaled_tanh_domain,
            tau_m=tau_s,
        )

    # Division consumes a constant numerator and the sigmoid denominator. Treat its
    # log encoders plus exponential difference as one atomic operator boundary.
    denominator = 1.0 + negative_exponential
    division_domain = PotentialBounds(
        1.0,
        1.0 + negative_exponential_domain.max,
    )
    numerator = torch.ones_like(denominator)
    if "division" in dense_operators:
        ratio, ratio_domain = _dense_gelu_division(
            numerator,
            denominator,
            division_domain,
            tau_s=tau_s,
        )
    else:
        ratio, ratio_domain = division_function(
            numerator,
            denominator,
            division_domain,
            tau_s=tau_s,
        )
    tanh_output = 2.0 * ratio - 1.0
    tanh_output_domain = PotentialBounds(
        2.0 * ratio_domain.min - 1.0,
        2.0 * ratio_domain.max - 1.0,
    )

    # Finish 0.5*x*(1+tanh) through the same two multiplication stages. Their
    # inclusion is important because the final product determines whether a gate
    # error is amplified by the original MLP activation magnitude.
    one_plus_tanh_domain = PotentialBounds(
        1.0 + tanh_output_domain.min,
        1.0 + tanh_output_domain.max,
    )
    one_plus_tanh = one_plus_tanh_domain.clamp(
        1.0 + tanh_output,
        name="gelu_ablation_one_plus",
    )
    half = input_value.new_tensor(0.5).expand_as(input_value)
    gate, gate_domain = multiply(
        one_plus_tanh,
        one_plus_tanh_domain,
        half,
        PotentialBounds(0.5, 0.5),
    )
    return multiply(input_clamped, domain, gate, gate_domain)


def install_gelu_operator_ablation(dense_operators: frozenset[str]) -> None:
    """Install the analysis GELU replacement in the local ViT module namespace."""
    unknown = dense_operators - _OPERATOR_NAMES
    if unknown:
        raise ValueError(f"unknown GELU operator ablations: {sorted(unknown)}")

    # ViTIntermediate resolves gelu_approximation from this module namespace at
    # forward time, so replacing only that symbol leaves every other model family
    # and every non-GELU use of the atomic operators unchanged.
    def configured_gelu(
        input_value: torch.Tensor,
        domain: PotentialBounds,
        **kwargs: object,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        return gelu_operator_ablation(
            input_value,
            domain,
            dense_operators=dense_operators,
            **kwargs,
        )

    # This mutation is process-local and happens before model evaluation. The shell
    # driver runs one condition per process, so variants never coexist in one RNG.
    modeling_spiking_vit.gelu_approximation = configured_gelu


def parse_arguments() -> tuple[Arguments, frozenset[str]]:
    """Parse analysis-only operator choices followed by the ordinary ViT CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gelu-dense-operators",
        nargs="*",
        choices=sorted(_OPERATOR_NAMES),
        default=(),
    )

    # Remove the analysis option before delegating all model, dataset, and Gaussian
    # controls to the maintained evaluator parser.
    analysis_args, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        vit_args = parse_vit_arguments()
    finally:
        sys.argv = original_argv

    dense_operators = frozenset(analysis_args.gelu_dense_operators)
    if vit_args.spiking_mlp_exact_gelu or vit_args.spiking_mlp_exact_gelu_layers:
        raise ValueError(
            "GELU operator ablation cannot be combined with another exact-GELU mode"
        )

    # Dataclasses without slots permit the analysis identity to join vars(args), so
    # the evaluator records it in W&B without widening the production Arguments API.
    vit_args.gelu_dense_operators = tuple(sorted(dense_operators))
    return vit_args, dense_operators


def main() -> None:
    """Install one GELU-local operator variant and run the ordinary ViT evaluator."""
    args, dense_operators = parse_arguments()
    install_gelu_operator_ablation(dense_operators)
    print(
        "Dense GELU-local operators: "
        + (", ".join(sorted(dense_operators)) if dense_operators else "none")
    )
    evaluate_vit_model(args)


if __name__ == "__main__":
    main()
