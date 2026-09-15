"""Verify fixed GELU gains, dynamic products, and Gaussian event accounting."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis import gelu_cubic_phi_nl_vit as cubic
from utils.transforms import functions, noise
from utils.transforms.types import PotentialBounds


def verify_fixed_gains() -> None:
    """Fixed gains scale both endpoints without a temporal event or random draw."""
    values = torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64)
    domain = PotentialBounds(-3.0, 2.0)
    noise.set_gaussian_time_noise(
        enabled=True, time_std=1.0e-4, deadline_margin=4.0e-4, seed=17, device="cpu",
    )
    generator = noise.get_gaussian_time_noise().generator
    assert isinstance(generator, torch.Generator)
    before = generator.get_state().clone()
    with patch.object(functions, "multiplication_operator", side_effect=AssertionError):
        for scale in (0.044715, 0.7978845608028654, 50.0, -2.0, 0.0):
            actual, bounds = functions._constant_synaptic_scale(
                values, domain, scale, name="verification_fixed_gain",
            )
            expected = values * scale
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            assert bounds == PotentialBounds(expected.min().item(), expected.max().item())
        clipped, bounds = functions._constant_synaptic_scale(
            values * 10.0, domain, -2.0, name="verification_fixed_gain",
        )
        torch.testing.assert_close(clipped, torch.tensor([6.0, 0.0, -4.0]).double())
        assert bounds == PotentialBounds(-4.0, 6.0)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            try:
                functions._constant_synaptic_scale(values, domain, invalid, name="invalid")
            except ValueError:
                pass
            else:
                raise AssertionError("accepted a non-finite fixed gain")
    assert torch.equal(before, generator.get_state())
    assert noise.get_gaussian_noise_stats() == {}
    noise.set_gaussian_time_noise(enabled=False)


def verify_tanh_time_constants() -> None:
    """A fixed time scale cancels at decoding without theta clipping of its gain."""
    values = torch.linspace(-3.0, 3.0, 79, dtype=torch.float64)
    domain = PotentialBounds(-3.0, 3.0)
    for enabled in (False, True):
        for tau_s in (0.125, 0.5, 1.0, 2.0, 25.0):
            noise.set_gaussian_time_noise(enabled=enabled, time_std=0.0, seed=3, device="cpu")
            with patch.object(functions, "multiplication_operator", side_effect=AssertionError):
                actual, bounds = functions._tanh_sigmoid_gate(
                    values, domain, tau_s=tau_s, theta=40.0,
                )
            torch.testing.assert_close(actual, torch.sigmoid(2.0 * values), rtol=2.0e-12, atol=2.0e-12)
            assert bounds == PotentialBounds(0.0, 1.0)
            stats = noise.get_gaussian_noise_stats()
            assert stats.get("multiplication.data", {}).get("events", 0) == 0
            assert stats.get("multiplication.reference", {}).get("events", 0) == 0
    noise.set_gaussian_time_noise(enabled=False)


def verify_cubic_boundaries() -> None:
    """Signed magnitudes retain the floor, inclusive theta cap, and fixed output bounds."""
    floor = 1.0e-5
    values = torch.tensor(
        [-80.0, -40.0, -3.0, -0.752461, -floor, -floor / 2, 0.0,
         floor / 2, floor, 3.0, 40.0, 80.0],
        dtype=torch.float64,
    )
    domain = PotentialBounds(-80.0, 80.0)
    reference_cube = values.clamp(-40.0, 40.0).pow(3)
    reference_cube = torch.where(values.abs() >= floor, reference_cube, 0.0)
    reference_gate = torch.sigmoid(
        2.0 * 0.7978845608028654 * (values + 0.044715 * reference_cube),
    )
    expected_gelu = (values * reference_gate).clamp(functions.GELU_OUTPUT_MIN, 80.0)
    for tau_s in (0.5, 1.0, 2.0, 25.0):
        outputs = []
        for enabled in (False, True):
            noise.set_gaussian_time_noise(enabled=enabled, time_std=0.0, seed=3, device="cpu")
            actual_cube, cube_bounds = cubic.phi_nl_psi_ed_cube(
                values, domain, tau_s=tau_s, theta=40.0, magnitude_floor=floor,
            )
            torch.testing.assert_close(actual_cube, reference_cube, rtol=3.0e-13, atol=2.0e-20)
            assert cube_bounds == PotentialBounds(-64000.0, 64000.0)
            actual, bounds = cubic.gelu_with_phi_nl_psi_ed_cube(
                values, domain, tau_s=tau_s, theta=40.0, magnitude_floor=floor,
            )
            assert functions.OUTPUT_BOUNDS_VERSION == 3
            assert bounds == PotentialBounds(functions.GELU_OUTPUT_MIN, 80.0)
            assert bool(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected_gelu, rtol=2.0e-11, atol=2.0e-11)
            outputs.append(actual)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2.0e-12, atol=2.0e-12)
    noise.set_gaussian_time_noise(enabled=False)


def verify_dynamic_products() -> None:
    """Only data-dependent products enter the composed multiplication operator."""
    values = torch.linspace(-3.0, 3.0, 53, dtype=torch.float64)
    domain = PotentialBounds(-3.0, 3.0)
    variants = (
        (functions, functions.gelu_approximation, 3),
        (functions, functions.gelu_approximation_sigmoid, 1),
        (cubic, cubic.gelu_with_phi_nl_psi_ed_cube, 1),
    )
    for owner, evaluator, expected_count in variants:
        calls = []
        original = functions.multiplication_operator

        def traced(value, value_domain, factor, factor_domain, theta):
            assert factor_domain.min < factor_domain.max, "a fixed gain entered multiplication"
            calls.append((factor.clone(), factor_domain))
            return original(value, value_domain, factor, factor_domain, theta)

        noise.set_gaussian_time_noise(enabled=False)
        with patch.object(functions, "multiplication_operator", side_effect=traced):
            if owner is cubic:
                with patch.object(cubic, "multiplication_operator", side_effect=traced):
                    result, _ = evaluator(values, domain, theta=40.0)
            else:
                result, _ = evaluator(values, domain, theta=40.0)
        assert len(calls) == expected_count
        assert calls[-1][1] == PotentialBounds(0.0, 1.0)
        assert bool(torch.isfinite(result).all())
    # The generic operator must still encode a caller-supplied constant operand.
    noise.set_gaussian_time_noise(enabled=True, time_std=0.0, seed=3, device="cpu")
    functions.multiplication_operator(
        values, domain, torch.full_like(values, 0.5), PotentialBounds(0.5, 0.5), 40.0,
    )
    assert noise.get_gaussian_noise_stats()["multiplication.data"]["events"] == values.numel()
    noise.set_gaussian_time_noise(enabled=False)


def verify_gaussian_events_and_replay() -> None:
    """Count every remaining event and match the exact dedicated generator advance."""
    values = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64)
    domain = PotentialBounds(-3.0, 3.0)
    count = values.numel()
    expected = {
        "gelu.cubic.log_positive": count,
        "gelu.cubic.log_negative": count,
        "gelu.cubic.log_reference": 1,
        "exponential_difference.internal": 3 * count,
        "exponential.input": count,
        "division.numerator": count,
        "division.denominator": count,
        "multiplication.data": count,
        "multiplication.reference": 1,
    }
    outputs = []
    for seed in (17, 17, 18):
        noise.set_gaussian_time_noise(
            enabled=True, time_std=1.0e-3, deadline_margin=4.0e-3, seed=seed, device="cpu",
        )
        generator = noise.get_gaussian_time_noise().generator
        assert isinstance(generator, torch.Generator)
        with patch.object(
            noise, "_sample_gaussian_spike_time", wraps=noise._sample_gaussian_spike_time,
        ) as sampler:
            output, _ = cubic.gelu_with_phi_nl_psi_ed_cube(values, domain, theta=40.0)
        outputs.append(output)
        stats = noise.get_gaussian_noise_stats()
        events = {site: counts["events"] for site, counts in stats.items() if counts["events"]}
        assert events == expected, events
        shapes = [tuple(call.args[0].shape) for call in sampler.call_args_list]
        assert shapes == [(count,), (count,), (), *[(count,)] * 7, ()], shapes
        independent = torch.Generator(device="cpu").manual_seed(seed)
        for shape in shapes:
            torch.normal(
                mean=torch.zeros(shape, dtype=values.dtype),
                std=torch.full(shape, 1.0e-3, dtype=values.dtype),
                generator=independent,
            )
        assert torch.equal(generator.get_state(), independent.get_state())
        assert all(isfinite(float(item)) for item in output)
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    noise.set_gaussian_time_noise(enabled=False)


def verify_gelu_synaptic_scaling() -> None:
    """Run the fixed-gain regression without evaluating a model or using a GPU."""
    try:
        for check in (
            verify_fixed_gains,
            verify_tanh_time_constants,
            verify_cubic_boundaries,
            verify_dynamic_products,
            verify_gaussian_events_and_replay,
        ):
            check()
            print(f"PASS {check.__name__}")
    finally:
        noise.set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    verify_gelu_synaptic_scaling()
