"""CPU checks for LayerNorm final affine bounds, clamping, and cache reuse."""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transformers.models import spiking_ops
from utils.transformers.models.spiking_ops import SpikingLayerNorm
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import Potential, PotentialBounds


def _layer(flags: tuple[bool, bool, bool], dtype=torch.float64) -> SpikingLayerNorm:
    layer = SpikingLayerNorm(
        4,
        theta=4.0,
        clip_margin=0.1,
        use_spiking_mul=flags[0],
        use_spiking_log=flags[1],
        use_spiking_expdiff=flags[2],
    ).to(dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([6.0, 0.2, -0.1, -1.0], dtype=dtype))
        layer.bias.copy_(torch.tensor([0.0, 6.0, -6.0, 0.0], dtype=dtype))
    return layer


def _input(dtype=torch.float64) -> Potential:
    return Potential(
        torch.tensor(
            [[-1.5, -0.25, 0.75, 1.0], [0.5, -1.0, 1.5, -0.5]],
            dtype=dtype,
        ),
        PotentialBounds(-2.0, 2.0),
    )


# @lat: [[bounds-audit#2026-09-14 Bound Corrections#LayerNorm Affine Bounds]]
def verify_paired_bounds_and_parity() -> None:
    """Check every ablation, signed scales, theta clipping, and clean parity."""
    original_clamp = spiking_ops.clamp_gaussian_output

    def skip_final(value, domain, *, site, name):
        if site == "layernorm.affine_output":
            return value
        return original_clamp(value, domain, site=site, name=name)

    for flags, dtype in product(product((False, True), repeat=3), (torch.float32, torch.float64)):
        layer = _layer(flags, dtype)
        value = _input(dtype)
        weight = layer.weight.detach().to(torch.float64)
        bias = layer.bias.detach().to(torch.float64)
        limit = math.sqrt(4 if any(flags) else 3)
        effective_weight = weight.clamp(-4.0, 4.0) if flags[2] else weight
        radius = effective_weight.abs() * limit
        expected = PotentialBounds(
            (bias - radius).min().item(), (bias + radius).max().item()
        )
        frozen = layer.freeze_parameter_bounds()
        assert frozen[2] == expected
        assert layer.freeze_parameter_bounds() is frozen
        if flags[2]:
            assert expected == PotentialBounds(-8.0, 8.0)
            old_lower = -radius.max().item() + bias.min().item()
            old_upper = radius.max().item() + bias.max().item()
            assert old_lower == -14.0 and old_upper == 14.0
            assert expected.min > old_lower and expected.max < old_upper

        set_gaussian_time_noise(enabled=False)
        with patch.object(spiking_ops, "clamp_gaussian_output", skip_final):
            prior = layer(value)
        clean = layer(value)
        assert torch.equal(clean.value, prior.value.clamp(expected.min, expected.max))
        torch.testing.assert_close(clean.value, prior.value, rtol=0, atol=2.0e-6)
        assert clean.domain is frozen[2]
        assert get_gaussian_noise_stats() == {}
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=804)
        zero = layer(value)
        tolerance = 2.0e-5 if dtype == torch.float32 else 2.0e-12
        torch.testing.assert_close(zero.value, clean.value, rtol=tolerance, atol=tolerance)
        assert zero.domain is frozen[2]
        assert bool(((zero.value >= expected.min) & (zero.value <= expected.max)).all())
        if not any(flags):
            reference = torch.nn.functional.layer_norm(
                value.value, (4,), layer.weight, layer.bias, layer.eps
            )
            assert torch.equal(zero.value, reference)
            assert get_gaussian_noise_stats() == {}
        else:
            stats = get_gaussian_noise_stats()["layernorm.affine_output"]
            assert stats["outputs"] == value.value.numel()
            assert stats["events"] == stats["misses"] == 0
        set_gaussian_time_noise(enabled=False)
        assert torch.equal(layer(value).value, clean.value)


def verify_noisy_clamp_and_random_stream() -> None:
    """Compare identical event streams with and without the final output clamp."""
    original_clamp = spiking_ops.clamp_gaussian_output

    def skip_final(value, domain, *, site, name):
        if site == "layernorm.affine_output":
            return value
        return original_clamp(value, domain, site=site, name=name)

    for flags in product((False, True), repeat=3):
        if not any(flags):
            continue
        layer = _layer(flags)
        value = _input()
        set_gaussian_time_noise(enabled=True, time_std=0.4, seed=805)
        with patch.object(spiking_ops, "clamp_gaussian_output", skip_final):
            raw = layer(value)
        prior_stats = get_gaussian_noise_stats()
        prior_state = get_gaussian_time_noise().generator.get_state().clone()
        set_gaussian_time_noise(enabled=True, time_std=0.4, seed=805)
        bounded = layer(value)
        stats = get_gaussian_noise_stats()
        final = stats.pop("layernorm.affine_output")
        assert stats == prior_stats
        assert torch.equal(prior_state, get_gaussian_time_noise().generator.get_state())
        assert bounded.domain is raw.domain
        assert torch.equal(
            bounded.value, raw.value.clamp(raw.domain.min, raw.domain.max)
        )
        assert final["outputs"] == raw.value.numel()
        assert final["output_underflows"] == int((raw.value < raw.domain.min).sum())
        assert final["output_overflows"] == int((raw.value > raw.domain.max).sum())
        assert final["events"] == final["misses"] == 0
    set_gaussian_time_noise(enabled=False)


def verify_forced_output_violations() -> None:
    """Force a bounded shared product outside the tighter final affine interval."""
    layer = _layer((False, True, True))
    value = _input()
    original_multiplication = spiking_ops.multiplication_operator

    def forced_product(first, first_domain, second, second_domain, theta):
        if second is layer.weight:
            # These products remain inside [-8, 8], the shared multiplication
            # interval. Adding each matching bias creates two strict violations.
            scaled = first.new_tensor([0.0, 8.0, -8.0, 0.0]).expand_as(first)
            return scaled, PotentialBounds(-8.0, 8.0)
        return original_multiplication(first, first_domain, second, second_domain, theta)

    for enabled in (False, True):
        set_gaussian_time_noise(enabled=enabled, time_std=0.0, seed=806)
        with patch.object(spiking_ops, "multiplication_operator", forced_product):
            output = layer(value)
        expected = output.value.new_tensor([0.0, 8.0, -8.0, 0.0]).expand_as(output.value)
        assert torch.equal(output.value, expected)
        if enabled:
            final = get_gaussian_noise_stats()["layernorm.affine_output"]
            assert final["outputs"] == 8
            assert final["output_underflows"] == 2
            assert final["output_overflows"] == 2
            assert final["events"] == final["misses"] == 0
        else:
            assert get_gaussian_noise_stats() == {}
    set_gaussian_time_noise(enabled=False)


def verify_cache_and_single_feature() -> None:
    """Preserve cache invalidation and the zero-width single-feature dense bound."""
    layer = _layer((True, True, True))
    before = layer.freeze_parameter_bounds()
    with torch.no_grad():
        layer.bias.add_(0.25)
    try:
        layer.freeze_parameter_bounds()
    except RuntimeError:
        pass
    else:
        raise AssertionError("LayerNorm accepted changed parameters without refresh")
    after = layer.freeze_parameter_bounds(refresh=True)
    assert after is not before
    assert after[2] == PotentialBounds(-7.75, 8.25)
    layer.theta = 8.0
    try:
        layer(_input())
    except RuntimeError:
        pass
    else:
        raise AssertionError("LayerNorm accepted changed theta without refresh")
    assert layer.freeze_parameter_bounds(refresh=True)[2] == PotentialBounds(-11.75, 12.25)
    assert not any("bounds" in name for name in layer.state_dict())

    dense = SpikingLayerNorm(
        1, use_spiking_mul=False, use_spiking_log=False, use_spiking_expdiff=False
    ).to(torch.float64)
    with torch.no_grad():
        dense.bias.fill_(0.5)
    for enabled in (False, True):
        set_gaussian_time_noise(enabled=enabled, time_std=0.1, seed=807)
        output = dense(Potential(torch.tensor([[3.0], [-2.0]], dtype=torch.float64), PotentialBounds(-4, 4)))
        assert output.domain == PotentialBounds(0.5, 0.5)
        assert torch.equal(output.value, torch.full_like(output.value, 0.5))
        assert get_gaussian_noise_stats() == {}
    set_gaussian_time_noise(enabled=False)


def main() -> None:
    try:
        verify_paired_bounds_and_parity()
        verify_noisy_clamp_and_random_stream()
        verify_forced_output_violations()
        verify_cache_and_single_feature()
    finally:
        set_gaussian_time_noise(enabled=False)
    print("LayerNorm affine bounds: 4 verification groups passed")


if __name__ == "__main__":
    main()
