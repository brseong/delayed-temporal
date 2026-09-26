"""Verify SwiGLU composition and signed integration against independent equations."""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path
import sys

import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.transforms.clock import set_clock_driven
from utils.transforms.functions import swiglu_function
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.primitive import signed_pulse_width_modulation_operator
from utils.transforms.spike_to_potential import exponential_difference_operator
from utils.transforms.types import PotentialBounds, SpikeSample, TimeBounds


def verify_symbolic_identities() -> None:
    """Check gains, time order and the two positive accumulators in exact algebra."""
    z, a, b, t1, t2 = sp.symbols("z a b t1 t2", real=True)
    tau = sp.symbols("tau", positive=True)
    corrected_exp = sp.exp(-a / tau) * sp.exp(((b - z) - (b - a)) / tau)
    assert sp.simplify(corrected_exp - sp.exp(-z / tau)) == 0

    limit = sp.symbols("limit", positive=True)
    intermediate = -(t2 - t1)
    encoded = limit - intermediate
    decoded = sp.exp(limit / tau) * sp.exp((encoded - 2 * limit) / tau)
    assert sp.simplify(decoded - sp.exp((t2 - t1) / tau)) == 0

    upper, x, y = sp.symbols("upper x y", positive=True)
    tx = tau * sp.log(upper / x)
    ty = tau * sp.log(upper / y)
    assert sp.simplify(sp.exp((ty - tx) / tau) - x / y) == 0

    positive, negative, da, db = sp.symbols("positive negative da db", nonnegative=True)
    q_positive = positive * da + negative * db
    q_negative = positive * db + negative * da
    assert sp.expand(q_positive - q_negative - (positive - negative) * (da - db)) == 0


def verify_signed_accumulators() -> None:
    """Check both time orders and every delivery mask without ordering the events."""
    for mode, origin, drive, fired_a, fired_b in product(
        ("continuous", "fixed_step", "fixed_window"), (0.0, 3.0),
        (-2.0, 0.0, 3.0), (False, True), (False, True),
    ):
        set_clock_driven(
            enabled=mode != "continuous",
            time_step=0.25 if mode == "fixed_step" else 0.0,
            time_steps_per_window=8 if mode == "fixed_window" else 0,
        )
        times = torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64) + origin
        ta, tb = torch.meshgrid(times, times, indexing="ij")
        deadline = origin + 2.0
        domain = TimeBounds(origin, deadline)
        event_a = SpikeSample(ta, domain, torch.full_like(ta, fired_a, dtype=torch.bool), deadline)
        event_b = SpikeSample(tb, domain, torch.full_like(tb, fired_b, dtype=torch.bool), deadline)
        actual, bounds = signed_pulse_width_modulation_operator(
            event_a, domain, event_b, domain,
            torch.full_like(ta, drive), PotentialBounds(drive, drive),
            observation_deadline=deadline,
        )
        # Independently integrate nonnegative currents, one clock interval at a time.
        q_positive = torch.zeros_like(ta)
        q_negative = torch.zeros_like(ta)
        for index in range(8):
            active_a = (ta <= origin + index * 0.25) & fired_a
            active_b = (tb <= origin + index * 0.25) & fired_b
            q_positive += 0.25 * (max(drive, 0.0) * active_a + max(-drive, 0.0) * active_b)
            q_negative += 0.25 * (max(drive, 0.0) * active_b + max(-drive, 0.0) * active_a)
        assert torch.all(q_positive >= 0) and torch.all(q_negative >= 0)
        torch.testing.assert_close(actual, q_positive - q_negative, rtol=0, atol=0)
        assert bounds == PotentialBounds(-2 * abs(drive), 2 * abs(drive))
    set_clock_driven(enabled=False)


def verify_exponential_difference_form() -> None:
    """Compare the primitive with its nonnegative-time implementation and target."""
    times = torch.tensor([0.0, 0.25, 1.0, 2.0], dtype=torch.float64)
    ta, tb = torch.meshgrid(times, times, indexing="ij")
    domain = TimeBounds(0.0, 2.0)
    for tau in (0.5, 1.0, 2.0):
        actual, _ = exponential_difference_operator(ta, domain, tb, domain, tau_s=tau)
        p = ta - tb
        encoded = 2.0 - p
        expanded = math.exp(2.0 / tau) * torch.exp((encoded - 4.0) / tau)
        expected = torch.exp((tb - ta) / tau)
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
        torch.testing.assert_close(expanded, expected, atol=2e-13, rtol=2e-13)


def verify_swiglu_form() -> None:
    """Compare independent signs, local domains, beta and time constants."""
    for dtype, atol, rtol in ((torch.float64, 2e-13, 2e-13), (torch.float32, 2e-5, 2e-5)):
        largest_error = 0.0
        count = 0
        for limits, beta, tau in product(
            ((-3.0, 4.0), (-3.0, -0.5), (0.5, 4.0)),
            (-1.0, 0.0, 0.7, 1.0), (0.5, 1.0, 2.0),
        ):
            u = torch.linspace(*limits, 33, dtype=dtype)[:, None]
            v = torch.tensor([-3.0, -0.5, 0.0, 0.75, 2.0], dtype=dtype)[None, :]
            actual, bounds = swiglu_function(
                u, PotentialBounds(*limits), v, PotentialBounds(-3.0, 2.0),
                beta=beta, tau_s=tau,
            )
            expected = v * u * torch.sigmoid(beta * u)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            assert torch.all(actual >= bounds.min) and torch.all(actual <= bounds.max)
            largest_error = max(largest_error, (actual - expected).abs().max().item())
            count += actual.numel()
        print(f"{dtype}: {count} values, maximum absolute error {largest_error:.9g}")


def main() -> None:
    set_clock_driven(enabled=False)
    set_gaussian_time_noise(enabled=False)
    try:
        for check in (
            verify_symbolic_identities,
            verify_signed_accumulators,
            verify_exponential_difference_form,
            verify_swiglu_form,
        ):
            check()
            print(f"PASS {check.__name__}", flush=True)
    finally:
        set_clock_driven(enabled=False)
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
