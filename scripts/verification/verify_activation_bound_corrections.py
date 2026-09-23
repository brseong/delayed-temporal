"""CPU regressions for activation caps, intermediate bounds, and event accounting."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transforms import functions as functions
from utils.transforms.functions import (
    SWISH_OUTPUT_MIN,
    clamp_gelu_square_output,
    clamp_sigmoid_exponential_input,
    multiplication_operator,
)
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds


def _check_closed(value: torch.Tensor, domain: PotentialBounds) -> None:
    assert bool(torch.isfinite(value).all())
    assert bool((value >= domain.min).all())
    assert bool((value <= domain.max).all())


# @lat: [[bounds-audit#2026-09-14 Bound Corrections#Activation Intermediate Bounds]]
def verify_exponential_cap_bounds() -> None:
    """Map saturated endpoints and retain a usable fixed interval for encoding."""
    for dtype in (torch.float32, torch.float64):
        for limit in (20.0, 80.0):
            for tau in (0.5, 1.0, 2.0):
                cap = limit * tau
                cases = (
                    (PotentialBounds(-2.0 * cap, -1.5 * cap), PotentialBounds(-cap, cap)),
                    (PotentialBounds(1.5 * cap, 2.0 * cap), PotentialBounds(-cap, cap)),
                    (PotentialBounds(-2.0 * cap, -0.5 * cap), PotentialBounds(-cap, -0.5 * cap)),
                    (PotentialBounds(-0.5 * cap, 2.0 * cap), PotentialBounds(-0.5 * cap, cap)),
                    (PotentialBounds(-tau, tau), PotentialBounds(-tau, tau)),
                    (PotentialBounds(0.0, 0.0), PotentialBounds(-cap, cap)),
                    (PotentialBounds(2.0 * cap, 2.0 * cap), PotentialBounds(-cap, cap)),
                    (PotentialBounds(-2.0 * cap, -2.0 * cap), PotentialBounds(-cap, cap)),
                )
                for domain, expected in cases:
                    values = torch.linspace(domain.min, domain.max, 5, dtype=dtype)
                    output, bounds = clamp_sigmoid_exponential_input(values, domain, tau_s=tau, limit=limit)
                    assert bounds == expected
                    assert bounds.max > bounds.min
                    assert torch.equal(output, values.clamp(-cap, cap))
                    _check_closed(output, bounds)

        # These inputs are inside the cap but would round 1 + exp(-x) to 1.
        lower = 18.0 if dtype == torch.float32 else 38.0
        domain = PotentialBounds(lower, lower + 0.5)
        values = torch.linspace(domain.min, domain.max, 3, dtype=dtype)
        output, bounds = clamp_sigmoid_exponential_input(values, domain, tau_s=1.0, limit=80.0)
        assert bounds == PotentialBounds(-80.0, 80.0)
        assert torch.equal(output, values)
        # A numerically tiny interval is expanded independently of tensor extrema.
        domain = PotentialBounds(1.0, 1.0 + 0.5 * torch.finfo(dtype).eps)
        output, bounds = clamp_sigmoid_exponential_input(torch.ones(1, dtype=dtype), domain, tau_s=1.0, limit=80.0)
        assert bounds == PotentialBounds(-80.0, 80.0)
        _check_closed(output, bounds)


def _activation_cases(values: torch.Tensor, domain: PotentialBounds):
    yield "tanh", lambda: functions.tanh(values, domain)
    yield "gelu_tanh", lambda: functions.gelu_approximation(values, domain)
    yield "gelu_sigmoid", lambda: functions.gelu_approximation_sigmoid(values, domain)
    for beta in (-1.0, 0.0, 1.0):
        yield f"swiglu_{beta}", lambda beta=beta: functions.swiglu_function(
            values, domain, torch.ones_like(values), PotentialBounds(1.0, 1.0),
            beta=beta,
        )


def verify_activation_cap_consumers() -> None:
    """Check finite bounded outputs and noise-off versus zero-noise parity."""
    for dtype in (torch.float32, torch.float64):
        tolerance = 3.0e-5 if dtype == torch.float32 else 3.0e-12
        rounding_lower = 18.0 if dtype == torch.float32 else 38.0
        for domain in (
            PotentialBounds(-2.0, 2.0), PotentialBounds(0.0, 0.0),
            PotentialBounds(100.0, 100.0), PotentialBounds(-100.0, -100.0),
            PotentialBounds(100.0, 101.0), PotentialBounds(-101.0, -100.0),
            PotentialBounds(rounding_lower, rounding_lower + 0.5),
        ):
            values = torch.linspace(domain.min, domain.max, 5, dtype=dtype)
            for name, evaluate in _activation_cases(values, domain):
                set_gaussian_time_noise(enabled=False)
                clean, clean_bounds = evaluate()
                _check_closed(clean, clean_bounds)
                set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=85, device="cpu")
                zero, zero_bounds = evaluate()
                _check_closed(zero, zero_bounds)
                assert clean_bounds == zero_bounds, name
                torch.testing.assert_close(zero, clean, rtol=tolerance, atol=tolerance, msg=name)
                assert sum(entry["events"] for entry in get_gaussian_noise_stats().values()) > 0
    set_gaussian_time_noise(enabled=False)


def verify_activation_event_stages() -> None:
    """Retain every sampled stage even when the exponential input is saturated."""
    event_counts = {}
    for domain in (
        PotentialBounds(-2.0, 2.0),
        PotentialBounds(100.0, 101.0), PotentialBounds(-101.0, -100.0),
    ):
        values = torch.linspace(domain.min, domain.max, 5, dtype=torch.float64)
        for name, evaluate in _activation_cases(values, domain):
            set_gaussian_time_noise(enabled=True, time_std_fraction=0.01, seed=86, device="cpu")
            output, bounds = evaluate()
            _check_closed(output, bounds)
            stats = get_gaussian_noise_stats()
            counts = {site: entry["events"] for site, entry in stats.items() if entry["events"]}
            if name not in event_counts:
                event_counts[name] = counts
            else:
                assert counts == event_counts[name], (name, domain, counts)
            for site in ("division.numerator", "division.denominator", "exponential_difference.internal"):
                assert counts[site] >= values.numel()
            if name != "tanh":
                assert counts["multiplication.data"] > 0
                assert counts["multiplication.reference"] > 0
            if name.startswith("swiglu"):
                assert counts["swiglu.exponential_input"] == values.numel()
                assert counts["multiplication.data"] == 2 * values.numel()
                assert counts["multiplication.reference"] == 2
                assert stats["swish.output"]["outputs"] == values.numel()
    set_gaussian_time_noise(enabled=False)


def verify_capped_swish_output_clipping() -> None:
    """Count the Swish output limit when the exponential cap changes its minimum."""
    u = torch.tensor([-1.0e9], dtype=torch.float64)
    domain = PotentialBounds(-1.0e9, -1.0e9)
    recorded = []
    original = functions.clamp_swish_output

    def observe(value, input_domain, *, beta=1.0):
        recorded.append(value.clone())
        return original(value, input_domain, beta=beta)

    set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=87, device="cpu")
    with patch.object(functions, "clamp_swish_output", observe):
        output, bounds = functions.swiglu_function(u, domain, torch.ones_like(u), PotentialBounds(1.0, 1.0))
    assert len(recorded) == 1
    assert float(recorded[0][0]) < SWISH_OUTPUT_MIN
    assert bounds == PotentialBounds(SWISH_OUTPUT_MIN, 0.0)
    _check_closed(output, bounds)
    torch.testing.assert_close(output, torch.full_like(output, SWISH_OUTPUT_MIN), rtol=0.0, atol=1.0e-14)
    stats = get_gaussian_noise_stats()["swish.output"]
    assert stats["outputs"] == stats["output_underflows"] == 1
    assert stats["output_overflows"] == stats["events"] == 0
    set_gaussian_time_noise(enabled=False)


def verify_repeated_input_square_bounds() -> None:
    """Tighten a repeated-input square while retaining signed generic products."""
    for dtype in (torch.float32, torch.float64):
        for magnitude in (2.0, 40.0, 80.0):
            domain = PotentialBounds(-magnitude, magnitude)
            values = torch.linspace(-magnitude, magnitude, 9, dtype=dtype)
            set_gaussian_time_noise(enabled=False)
            product, generic_bounds = multiplication_operator(values, domain, values, domain)
            squared, bounds = clamp_gelu_square_output(product, domain)
            maximum = magnitude * magnitude
            assert bounds == PotentialBounds(0.0, maximum)
            assert generic_bounds == PotentialBounds(-maximum, maximum)
            assert torch.equal(squared, product)
            torch.testing.assert_close(squared, values * values)
            _check_closed(squared, bounds)

    domain = PotentialBounds(-2.0, 2.0)
    raw = torch.tensor([-1.0, 0.0, 4.0, 5.0], dtype=torch.float64)
    set_gaussian_time_noise(enabled=True, time_std_fraction=0.2, seed=88, device="cpu")
    before = get_gaussian_time_noise().generator.get_state().clone()
    output, bounds = clamp_gelu_square_output(raw, domain)
    assert torch.equal(before, get_gaussian_time_noise().generator.get_state())
    assert torch.equal(output, raw.clamp(0.0, 4.0))
    stats = get_gaussian_noise_stats()
    assert set(stats) == {"gelu.square_output"}
    assert stats["gelu.square_output"]["outputs"] == 4
    assert stats["gelu.square_output"]["output_underflows"] == 1
    assert stats["gelu.square_output"]["output_overflows"] == 1
    assert stats["gelu.square_output"]["events"] == 0
    set_gaussian_time_noise(enabled=False)
    product, bounds = multiplication_operator(
        torch.tensor([-2.0, 2.0]), domain, torch.tensor([2.0, -2.0]), domain,
    )
    assert torch.equal(product, torch.tensor([-4.0, -4.0]))
    assert bounds == PotentialBounds(-4.0, 4.0)


def main() -> None:
    """Run only small CPU tensors, with no external assets or GPU allocation."""
    torch.set_num_threads(1)
    try:
        for verify in (
            verify_exponential_cap_bounds,
            verify_activation_cap_consumers,
            verify_activation_event_stages,
            verify_capped_swish_output_clipping,
            verify_repeated_input_square_bounds,
        ):
            verify()
            print(f"PASS {verify.__name__}")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
