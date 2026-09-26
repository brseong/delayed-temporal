"""Compare ViT GELU cubic constructions without duplicating the production path."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import isfinite
from pathlib import Path
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.evaluation.error_analysis_vit import (
    Arguments,
    GeluCubicImplementation,
    evaluate_vit_model,
    parse_arguments as parse_vit_arguments,
)
from utils.transforms.functions import (
    GELU_CUBIC_MAGNITUDE_FLOOR,
    _constant_synaptic_scale,
    _tanh_sigmoid_gate,
    clamp_gelu_output,
    clamp_gelu_square_output,
    gelu_approximation,
    multiplication_operator,
)
from utils.transforms.types import PotentialBounds
from utils.transformers.models.spiking_vit.modeling_spiking_vit import GeluOperator


def gelu_with_multiplication_cube(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    **_: object,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Retain the repeated multiplication cubic only as a comparison condition."""
    input_clamped = domain.clamp(input_value, name="gelu_x")
    square, square_domain = multiplication_operator(
        input_clamped, domain, input_clamped, domain,
    )
    square, square_domain = clamp_gelu_square_output(
        square, domain,
    )
    cube, cube_domain = multiplication_operator(
        square, square_domain, input_clamped, domain,
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
    )
    result, _product_bounds = multiplication_operator(
        input_clamped,
        domain,
        gate,
        gate_domain,
    )
    return clamp_gelu_output(result, domain)


def make_gelu_cubic_implementation(
    implementation: GeluCubicImplementation,
    *,
    magnitude_floor: float,
) -> GeluOperator:
    """Build one explicitly selected comparison path for a ViT instance."""
    if not isinstance(implementation, GeluCubicImplementation):
        raise TypeError("expected GeluCubicImplementation")

    def configured_gelu(
        input_value: torch.Tensor,
        domain: PotentialBounds,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        if implementation is GeluCubicImplementation.MULTIPLICATION:
            return gelu_with_multiplication_cube(input_value, domain)
        return gelu_approximation(
            input_value,
            domain,
            magnitude_floor=magnitude_floor,
        )

    return configured_gelu


def parse_arguments(argv: Sequence[str] | None = None) -> Arguments:
    """Parse the cubic selection before delegating the ordinary ViT arguments."""
    input_args = list(sys.argv[1:] if argv is None else argv)
    help_requested = any(arg in {"-h", "--help"} for arg in input_args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gelu-cubic-implementation",
        choices=[member.value for member in GeluCubicImplementation],
        required=not help_requested,
    )
    parser.add_argument(
        "--gelu-cubic-floor",
        type=float,
        default=GELU_CUBIC_MAGNITUDE_FLOOR,
    )
    analysis_args, remaining = parser.parse_known_args(input_args)
    vit_args = parse_vit_arguments(remaining)

    implementation = GeluCubicImplementation(analysis_args.gelu_cubic_implementation)
    magnitude_floor = float(analysis_args.gelu_cubic_floor)
    if vit_args.spiking_mlp_exact_gelu or vit_args.spiking_mlp_exact_gelu_layers:
        raise ValueError("GELU cubic comparison cannot be combined with exact GELU modes")
    if not isfinite(magnitude_floor) or magnitude_floor <= 0.0:
        raise ValueError("gelu-cubic-floor must be finite and positive")

    return replace(
        vit_args,
        gelu_cubic_implementation=implementation,
        gelu_cubic_floor=magnitude_floor,
    )


def main() -> None:
    """Construct the selected analysis function and run the ViT evaluator."""
    args = parse_arguments()
    implementation = args.gelu_cubic_implementation
    magnitude_floor = args.gelu_cubic_floor
    if implementation is None or magnitude_floor is None:
        raise ValueError("GELU cubic configuration is required")
    gelu_operator = make_gelu_cubic_implementation(
        implementation,
        magnitude_floor=magnitude_floor,
    )
    print(f"GELU cubic implementation: {implementation}")
    print(f"GELU cubic magnitude floor: {magnitude_floor:.9g}")
    evaluate_vit_model(args, gelu_operator=gelu_operator)


if __name__ == "__main__":
    main()
