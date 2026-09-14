"""Compare ViT GELU cubic constructions without changing production modules."""

from __future__ import annotations

import argparse
from math import isfinite
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
    _tanh_sigmoid_gate,
    clamp_gelu_output,
    multiplication_operator,
)
from utils.transforms.noise import get_gaussian_time_noise
from utils.transforms.potential_to_spike import neg_log_transform
from utils.transforms.spike_to_potential import exponential_difference_operator
from utils.transforms.types import PotentialBounds, SpikeSample


_CUBIC_IMPLEMENTATIONS = ("multiplication", "phi_nl_psi_ed")


# @lat: [[evaluation#Evaluation and Verification#Noise and Ablation Sweeps#GELU Cubic Construction Comparison]]
def phi_nl_psi_ed_cube(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float,
    theta: float,
    magnitude_floor: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Construct the signed GELU cubic term with scaled logarithmic encoding.

    Positive and negative magnitudes use ``phi_NL`` with ``3 * tau_s``. Their
    domain upper endpoint reference has latency zero, so ``psi_ED`` with scale
    ``tau_s`` returns the normalized magnitude cube. A fixed receiving gain restores
    the original potential scale, and the two rails recover the odd sign.

    Under Gaussian timing noise, the positive and negative magnitude encoders and
    their shared upper-endpoint reference return event-aware samples. The existing
    exponential-difference operator then applies the same delivery and internal
    event semantics as the rest of the maintained noise path.
    """
    tau_value = float(tau_s)
    theta_value = float(theta)
    floor_value = float(magnitude_floor)
    if not isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau_s must be finite and positive")
    if not isfinite(theta_value) or theta_value <= 0.0:
        raise ValueError("theta must be finite and positive")
    if not isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError("magnitude_floor must be finite and positive")
    magnitude_upper = min(
        max(abs(float(domain.min)), abs(float(domain.max))),
        theta_value,
    )
    if magnitude_upper <= floor_value:
        raise ValueError("GELU magnitude domain must exceed magnitude_floor")

    input_clamped = domain.clamp(input_value, name="gelu_phi_nl_x")
    magnitude_domain = PotentialBounds(floor_value, magnitude_upper)
    positive_magnitude = input_clamped.clamp(min=0.0, max=magnitude_upper)
    negative_magnitude = (-input_clamped).clamp(min=0.0, max=magnitude_upper)
    positive_active = positive_magnitude >= floor_value
    negative_active = negative_magnitude >= floor_value
    positive_carrier = magnitude_domain.clamp(
        positive_magnitude,
        name="gelu_phi_nl_positive_carrier",
    )
    negative_carrier = magnitude_domain.clamp(
        negative_magnitude,
        name="gelu_phi_nl_negative_carrier",
    )

    encoder_tau = 3.0 * tau_value
    gaussian_enabled = get_gaussian_time_noise().enabled
    encoder_kwargs: dict[str, object] = {}
    if gaussian_enabled:
        encoder_kwargs["return_spike_sample"] = True

    positive_time = neg_log_transform(
        positive_carrier,
        magnitude_domain,
        tau_s=encoder_tau,
        noise_site="gelu.cubic.log_positive",
        **encoder_kwargs,
    )
    negative_time = neg_log_transform(
        negative_carrier,
        magnitude_domain,
        tau_s=encoder_tau,
        noise_site="gelu.cubic.log_negative",
        **encoder_kwargs,
    )
    reference_time = neg_log_transform(
        input_value.new_tensor(magnitude_upper),
        magnitude_domain,
        tau_s=encoder_tau,
        noise_site="gelu.cubic.log_reference",
        **encoder_kwargs,
    )
    if gaussian_enabled:
        if not all(
            isinstance(event, SpikeSample)
            for event in (positive_time, negative_time, reference_time)
        ):
            raise RuntimeError(
                "Gaussian GELU cubic encoders must return SpikeSample"
            )
        time_domain = positive_time.domain
        negative_time_domain = negative_time.domain
        reference_time_domain = reference_time.domain
    else:
        positive_time, time_domain = positive_time
        negative_time, negative_time_domain = negative_time
        reference_time, reference_time_domain = reference_time
    if negative_time_domain != time_domain or reference_time_domain != time_domain:
        raise RuntimeError("GELU cubic log encoders require one shared time domain")

    positive_normalized, _ = exponential_difference_operator(
        positive_time,
        time_domain,
        reference_time,
        reference_time_domain,
        tau_s=tau_value,
    )
    negative_normalized, _ = exponential_difference_operator(
        negative_time,
        negative_time_domain,
        reference_time,
        reference_time_domain,
        tau_s=tau_value,
    )
    unit_domain = PotentialBounds(0.0, 1.0)
    positive_normalized = torch.where(
        positive_active,
        unit_domain.clamp(
            positive_normalized,
            name="gelu_phi_nl_positive_cube",
        ),
        torch.zeros_like(positive_normalized),
    )
    negative_normalized = torch.where(
        negative_active,
        unit_domain.clamp(
            negative_normalized,
            name="gelu_phi_nl_negative_cube",
        ),
        torch.zeros_like(negative_normalized),
    )

    cube_domain = PotentialBounds(
        -(magnitude_upper ** 3),
        magnitude_upper ** 3,
    )
    cube = magnitude_upper ** 3 * (
        positive_normalized - negative_normalized
    )
    return (
        cube_domain.clamp(cube, name="gelu_phi_nl_cube"),
        cube_domain,
    )


def gelu_with_phi_nl_psi_ed_cube(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    magnitude_floor: float = 1.0e-5,
    **_: object,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate GELU with only the cubic path of its tanh approximation changed."""
    input_clamped = domain.clamp(input_value, name="gelu_phi_nl_input")
    cube, cube_domain = phi_nl_psi_ed_cube(
        input_clamped,
        domain,
        tau_s=tau_s,
        theta=theta,
        magnitude_floor=magnitude_floor,
    )

    coefficient = 0.044715
    scaled_cube, scaled_cube_domain = multiplication_operator(
        cube,
        cube_domain,
        input_value.new_tensor(coefficient).expand_as(input_value),
        PotentialBounds(coefficient, coefficient),
        theta,
    )
    inner_domain = PotentialBounds(
        domain.min + scaled_cube_domain.min,
        domain.max + scaled_cube_domain.max,
    )
    inner = inner_domain.clamp(
        input_clamped + scaled_cube,
        name="gelu_phi_nl_inner",
    )

    tanh_scale = 0.7978845608028654
    tanh_input, tanh_input_domain = multiplication_operator(
        inner,
        inner_domain,
        input_value.new_tensor(tanh_scale).expand_as(input_value),
        PotentialBounds(tanh_scale, tanh_scale),
        theta,
    )
    gate, gate_domain = _tanh_sigmoid_gate(
        tanh_input,
        tanh_input_domain,
        tau_s=tau_s,
        theta=theta,
    )
    result, _ = multiplication_operator(
        input_clamped,
        domain,
        gate,
        gate_domain,
        theta,
    )
    return clamp_gelu_output(result, domain)


def install_phi_nl_psi_ed_cube(*, magnitude_floor: float) -> None:
    """Patch only the GELU symbol resolved by the local ViT adapter."""
    def configured_gelu(
        input_value: torch.Tensor,
        domain: PotentialBounds,
        **kwargs: object,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        return gelu_with_phi_nl_psi_ed_cube(
            input_value,
            domain,
            magnitude_floor=magnitude_floor,
            **kwargs,
        )

    modeling_spiking_vit.gelu_approximation = configured_gelu


def parse_arguments() -> tuple[Arguments, str, float]:
    """Parse the cubic selection before delegating the ordinary ViT arguments."""
    help_requested = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gelu-cubic-implementation",
        choices=_CUBIC_IMPLEMENTATIONS,
        required=not help_requested,
    )
    parser.add_argument(
        "--gelu-cubic-floor",
        type=float,
        default=1.0e-5,
    )
    analysis_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        vit_args = parse_vit_arguments()
    finally:
        sys.argv = original_argv

    implementation = str(analysis_args.gelu_cubic_implementation)
    magnitude_floor = float(analysis_args.gelu_cubic_floor)
    if vit_args.spiking_mlp_exact_gelu or vit_args.spiking_mlp_exact_gelu_layers:
        raise ValueError(
            "GELU cubic comparison cannot be combined with exact GELU modes"
        )
    if not isfinite(magnitude_floor) or magnitude_floor <= 0.0:
        raise ValueError("gelu-cubic-floor must be finite and positive")

    vit_args.gelu_cubic_implementation = implementation
    vit_args.gelu_cubic_floor = magnitude_floor
    return vit_args, implementation, magnitude_floor


def main() -> None:
    """Install the selected analysis path and run the maintained ViT evaluator."""
    args, implementation, magnitude_floor = parse_arguments()
    if implementation == "phi_nl_psi_ed":
        install_phi_nl_psi_ed_cube(magnitude_floor=magnitude_floor)
    print(f"GELU cubic implementation: {implementation}")
    print(f"GELU cubic magnitude floor: {magnitude_floor:.9g}")
    evaluate_vit_model(args)


if __name__ == "__main__":
    main()
