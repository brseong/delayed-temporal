"""CPU checks for fixed GELU output bounds and output clamping."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis import gelu_cubic_phi_nl_vit as cubic_module
from scripts.analysis import gelu_operator_ablation_vit as ablation_module
from utils.transforms import functions
from utils.transforms.functions import (
    GELU_OUTPUT_MIN,
    clamp_gelu_output,
    gelu_output_bounds,
)
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import Potential, PotentialBounds


def _variants():
    """Return each GELU implementation and the module owning its output helper."""
    return (
        (functions, functions.gelu_approximation, {}, "tanh"),
        (functions, functions.gelu_approximation_sigmoid, {}, "sigmoid"),
        (cubic_module, cubic_module.gelu_with_multiplication_cube, {}, "tanh"),
        (
            ablation_module,
            ablation_module.gelu_operator_ablation,
            {"dense_operators": frozenset()},
            "tanh",
        ),
        (
            ablation_module,
            ablation_module.gelu_operator_ablation,
            {"dense_operators": frozenset({"multiplication", "exponential", "division"})},
            "tanh",
        ),
    )


def verify_fixed_gelu_output_bounds() -> None:
    """Check the fixed lower endpoint and the declared input upper endpoint."""
    assert GELU_OUTPUT_MIN == -0.170041

    # Locate the tanh approximation minimum independently with its derivative.
    # Bracketing avoids making an empirical activation minimum the bounds source.
    def derivative(value: float) -> float:
        argument = math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)
        gate = math.tanh(argument)
        slope = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * value**2)
        return 0.5 * (1.0 + gate) + 0.5 * value * (1.0 - gate**2) * slope

    lower, upper = -1.0, -0.5
    assert derivative(lower) < 0.0 < derivative(upper)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if derivative(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    minimum_input = 0.5 * (lower + upper)
    argument = math.sqrt(2.0 / math.pi) * (
        minimum_input + 0.044715 * minimum_input**3
    )
    minimum = 0.5 * minimum_input * (1.0 + math.tanh(argument))
    assert GELU_OUTPUT_MIN < minimum < GELU_OUTPUT_MIN + 1.0e-6

    for input_domain, expected_upper in (
        (PotentialBounds(-163.3133, 163.3133), 163.3133),
        (PotentialBounds(1.0, 3.0), 3.0),
        (PotentialBounds(-4.0, 0.0), 0.0),
        (PotentialBounds(-4.0, -2.0), 0.0),
        (PotentialBounds(0.0, 0.0), 0.0),
    ):
        expected = PotentialBounds(GELU_OUTPUT_MIN, expected_upper)
        assert gelu_output_bounds(input_domain) == expected
        for dtype in (torch.float32, torch.float64):
            first, first_domain = clamp_gelu_output(
                torch.tensor([-3.0, 0.0, 4.0], dtype=dtype), input_domain
            )
            second, second_domain = clamp_gelu_output(
                torch.tensor([0.0, 0.01], dtype=dtype), input_domain
            )
            assert first_domain == second_domain == expected
            assert first.dtype == second.dtype == dtype
            assert bool((first >= expected.min).all())
            assert bool((first <= expected.max).all())

    # This API takes only immutable interval metadata, never activation tensors.
    assert tuple(inspect.signature(gelu_output_bounds).parameters) == ("input_domain",)
    tree = ast.parse(inspect.getsource(gelu_output_bounds))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"min", "max", "amin", "amax", "quantile"}
        for node in ast.walk(tree)
    )


def verify_gelu_output_clamp_counts() -> None:
    """Check strict clipping counts without introducing events or random draws."""
    domain = PotentialBounds(-2.0, 2.0)
    raw = torch.tensor([-0.5, GELU_OUTPUT_MIN, 0.0, 2.0, 2.5], dtype=torch.float64)
    set_gaussian_time_noise(enabled=True, time_std_fraction=0.1, seed=91, device="cpu")
    try:
        generator = get_gaussian_time_noise().generator
        before = generator.get_state().clone()
        output, output_domain = clamp_gelu_output(raw, domain)
        assert torch.equal(before, generator.get_state())
        assert output_domain == PotentialBounds(GELU_OUTPUT_MIN, 2.0)
        assert torch.equal(output, raw.clamp(GELU_OUTPUT_MIN, 2.0))
        stats = get_gaussian_noise_stats()
        assert set(stats) == {"gelu.output"}
        assert stats["gelu.output"]["outputs"] == 5
        assert stats["gelu.output"]["output_underflows"] == 1
        assert stats["gelu.output"]["output_overflows"] == 1
        assert stats["gelu.output"]["events"] == 0
        assert stats["gelu.output"]["misses"] == 0
    finally:
        set_gaussian_time_noise(enabled=False)
    clean, _ = clamp_gelu_output(raw, domain)
    assert torch.equal(clean, output)
    assert get_gaussian_noise_stats() == {}


def verify_gelu_variants() -> None:
    """Check clean parity and isolate the final clamp from internal event draws."""
    domain = PotentialBounds(-3.0, 3.0)
    for owner, evaluate, options, approximation in _variants():
        for dtype in (torch.float32, torch.float64):
            value = torch.linspace(-3.0, 3.0, 257, dtype=dtype)
            expected = (
                value * torch.sigmoid(1.702 * value)
                if approximation == "sigmoid"
                else torch.nn.functional.gelu(value, approximate="tanh")
            )
            set_gaussian_time_noise(enabled=False)
            with patch.object(
                owner, "clamp_gelu_output", lambda raw, bounds: (raw, gelu_output_bounds(bounds))
            ):
                prior, _ = evaluate(value, domain, **options)
            clean, clean_domain = evaluate(value, domain, **options)
            assert torch.equal(clean, prior)
            # Identity-code subtraction retains its existing float32 roundoff.
            tolerance = 2.0e-5 if dtype == torch.float32 else 2.0e-12
            torch.testing.assert_close(clean, expected, rtol=tolerance, atol=tolerance)
            assert clean_domain == PotentialBounds(GELU_OUTPUT_MIN, 3.0)
            set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=92)
            zero, zero_domain = evaluate(value, domain, **options)
            torch.testing.assert_close(zero, clean, rtol=tolerance, atol=tolerance)
            assert zero_domain == clean_domain
            set_gaussian_time_noise(enabled=False)
            replay, replay_domain = evaluate(value, domain, **options)
            assert torch.equal(replay, clean) and replay_domain == clean_domain

        value = torch.linspace(-3.0, 3.0, 257, dtype=torch.float64)

        def unclamped(raw, input_domain):
            return raw, gelu_output_bounds(input_domain)

        try:
            # Both calls see the same random stream. Bypassing only the new output
            # clamp must retain every prior event counter and the final RNG state.
            set_gaussian_time_noise(enabled=True, time_std_fraction=0.2, seed=93)
            with patch.object(owner, "clamp_gelu_output", unclamped):
                raw, raw_domain = evaluate(value, domain, **options)
            raw_stats = get_gaussian_noise_stats()
            raw_rng = get_gaussian_time_noise().generator.get_state().clone()
            set_gaussian_time_noise(enabled=True, time_std_fraction=0.2, seed=93)
            noisy, noisy_domain = evaluate(value, domain, **options)
            stats = get_gaussian_noise_stats()
            assert torch.equal(raw_rng, get_gaussian_time_noise().generator.get_state())
            output_stats = stats.pop("gelu.output")
            assert stats == raw_stats
            assert output_stats["events"] == output_stats["misses"] == 0
            assert output_stats["outputs"] == value.numel()
            assert output_stats["output_underflows"] == int((raw < GELU_OUTPUT_MIN).sum())
            assert output_stats["output_overflows"] == int((raw > domain.max).sum())
            assert noisy_domain == raw_domain == clean_domain
            assert torch.equal(noisy, raw.clamp(GELU_OUTPUT_MIN, domain.max))
        finally:
            set_gaussian_time_noise(enabled=False)


def verify_vit_gelu_output_bounds() -> None:
    """Exercise dense, direct, and composed ViT GELU output metadata on CPU."""
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTIntermediate

    domain = PotentialBounds(-2.0, 2.0)
    for use_spiking_mlp, exact_gelu in ((False, False), (True, True), (True, False)):
        config = ViTConfig(
            hidden_size=4,
            intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            use_spiking_mlp=use_spiking_mlp,
            spiking_mlp_exact_gelu=exact_gelu,
            hidden_act="gelu",
        )
        module = ViTIntermediate(config).double().eval()
        with torch.no_grad():
            module.dense.weight.copy_(torch.eye(4, dtype=torch.float64))
            module.dense.bias.zero_()
            first = module(Potential(torch.tensor([[[-1.0, -0.75, 0.5, 1.0]]], dtype=torch.float64), domain))
            second = module(Potential(torch.tensor([[[0.25, 0.5, 0.75, 1.0]]], dtype=torch.float64), domain))
        assert first.domain == second.domain == PotentialBounds(GELU_OUTPUT_MIN, 2.0)
        assert first.value.shape == second.value.shape == (1, 1, 4)
        assert bool(torch.isfinite(first.value).all())
        assert bool((first.value >= GELU_OUTPUT_MIN).all())


def main() -> None:
    """Run all checks without loading a checkpoint, dataset, or GPU."""
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    try:
        for verify in (
            verify_fixed_gelu_output_bounds,
            verify_gelu_output_clamp_counts,
            verify_gelu_variants,
            verify_vit_gelu_output_bounds,
        ):
            verify()
            print(f"PASS {verify.__name__}")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
