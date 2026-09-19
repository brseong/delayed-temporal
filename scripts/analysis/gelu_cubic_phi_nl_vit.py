"""Compare ViT GELU cubic constructions without duplicating the production path."""

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
    GELU_CUBIC_MAGNITUDE_FLOOR,
    _constant_synaptic_scale,
    _tanh_sigmoid_gate,
    clamp_gelu_output,
    clamp_gelu_square_output,
    gelu_approximation,
    gelu_cubic_power_operator,
    multiplication_operator,
)
from utils.transforms.types import PotentialBounds


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
    """Compatibility entry point for the canonical signed cubic power operator."""
    return gelu_cubic_power_operator(
        input_value,
        domain,
        tau_s=tau_s,
        theta=theta,
        magnitude_floor=magnitude_floor,
    )


def gelu_with_phi_nl_psi_ed_cube(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    magnitude_floor: float = GELU_CUBIC_MAGNITUDE_FLOOR,
    **kwargs: object,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Compatibility entry point for the canonical composed GELU."""
    return gelu_approximation(
        input_value,
        domain,
        tau_s=tau_s,
        theta=theta,
        magnitude_floor=magnitude_floor,
        **kwargs,
    )


def gelu_with_multiplication_cube(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_: object,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Retain the repeated multiplication cubic only as a comparison condition."""
    input_clamped = domain.clamp(input_value, name="gelu_x")
    square, square_domain = multiplication_operator(
        input_clamped, domain, input_clamped, domain, theta,
    )
    square, square_domain = clamp_gelu_square_output(
        square, domain, theta=theta,
    )
    cube, cube_domain = multiplication_operator(
        square, square_domain, input_clamped, domain, theta,
    )
    scaled_cube, scaled_cube_domain = _constant_synaptic_scale(
        cube,
        cube_domain,
        0.044715,
        name="gelu_cubic_coefficient",
    )
    inner_domain = PotentialBounds(
        domain.min + scaled_cube_domain.min,
        domain.max + scaled_cube_domain.max,
    )
    inner = inner_domain.clamp(input_clamped + scaled_cube, name="gelu_inner")
    tanh_input, tanh_input_domain = _constant_synaptic_scale(
        inner,
        inner_domain,
        0.7978845608028654,
        name="gelu_tanh_scale",
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


def install_gelu_cubic_implementation(
    implementation: str,
    *,
    magnitude_floor: float,
) -> None:
    """Install one explicitly selected comparison path in the local ViT adapter."""
    if implementation not in _CUBIC_IMPLEMENTATIONS:
        raise ValueError("unsupported GELU cubic implementation")

    def configured_gelu(
        input_value: torch.Tensor,
        domain: PotentialBounds,
        **kwargs: object,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        if implementation == "multiplication":
            return gelu_with_multiplication_cube(input_value, domain, **kwargs)
        return gelu_approximation(
            input_value,
            domain,
            magnitude_floor=magnitude_floor,
            **kwargs,
        )

    modeling_spiking_vit.gelu_approximation = configured_gelu


def install_phi_nl_psi_ed_cube(*, magnitude_floor: float) -> None:
    """Keep the archived analysis API while delegating to the canonical owner."""
    install_gelu_cubic_implementation(
        "phi_nl_psi_ed",
        magnitude_floor=magnitude_floor,
    )


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
        default=GELU_CUBIC_MAGNITUDE_FLOOR,
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
        raise ValueError("GELU cubic comparison cannot be combined with exact GELU modes")
    if not isfinite(magnitude_floor) or magnitude_floor <= 0.0:
        raise ValueError("gelu-cubic-floor must be finite and positive")

    vit_args.gelu_cubic_implementation = implementation
    vit_args.gelu_cubic_floor = magnitude_floor
    return vit_args, implementation, magnitude_floor


def main() -> None:
    """Install the selected analysis path and run the maintained ViT evaluator."""
    args, implementation, magnitude_floor = parse_arguments()
    install_gelu_cubic_implementation(
        implementation,
        magnitude_floor=magnitude_floor,
    )
    print(f"GELU cubic implementation: {implementation}")
    print(f"GELU cubic magnitude floor: {magnitude_floor:.9g}")
    evaluate_vit_model(args)


if __name__ == "__main__":
    main()
