"""CPU checks for the fixed attention output interval and its final clamp."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transformers.integrations import spiking_sdpa_attention as attention
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds


def _previous_bounds(theta: float, source_length_max: int) -> PotentialBounds:
    return PotentialBounds(-theta * source_length_max, theta * source_length_max)


# @lat: [[bounds-audit#2026-09-14 Bound Corrections#Attention Output Bounds]]
def verify_attention_output_bounds() -> None:
    """Keep the output interval independent of source count and random seed."""
    for theta in (2.0, 40.0, 1.0e308):
        for capacity in (1, 5, 197):
            result = attention.attention_output_bounds(theta, capacity)
            assert result == PotentialBounds(-theta, theta)
            assert result is attention.attention_output_bounds(theta, capacity)
    for capacity in (0, -1, True, False, 1.0, 5.0, "5", None):
        try:
            attention.attention_output_bounds(40.0, capacity)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid attention capacity accepted: {capacity!r}")
    for theta in (0.0, -1.0, math.nan, math.inf, -math.inf):
        try:
            attention.attention_output_bounds(theta, 5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid attention theta accepted: {theta!r}")


def verify_clean_attention_parity() -> None:
    """Preserve deterministic normalized attention and capacity validation."""
    generator = torch.Generator().manual_seed(51)
    query = 0.2 * torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
    key = 0.2 * torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    random_value = torch.randn(1, 2, 5, 3, generator=generator, dtype=torch.float64)
    weights = torch.softmax(query @ key.transpose(-2, -1) / 2.0, dim=-1)
    set_gaussian_time_noise(enabled=False)
    for value in (random_value, torch.full_like(random_value, 40.0),
                  torch.full_like(random_value, -40.0)):
        for capacity in (5, 197):
            result = attention.spiking_scaled_dot_product_attention(
                query, key, value, theta=40.0, source_length_max=capacity
            )
            with patch.object(attention, "attention_output_bounds", _previous_bounds):
                previous = attention.spiking_scaled_dot_product_attention(
                    query, key, value, theta=40.0, source_length_max=capacity
                )
            torch.testing.assert_close(result, previous, atol=1.0e-12, rtol=1.0e-12)
            torch.testing.assert_close(result, weights @ value, atol=1.0e-12, rtol=1.0e-12)
            assert bool((result.abs() <= 40.0).all())
            assert get_gaussian_noise_stats() == {}
    for options in ({}, {"source_length_max": 4}):
        try:
            attention.spiking_scaled_dot_product_attention(query, key, random_value,
                                                          theta=40.0, **options)
        except ValueError:
            pass
        else:
            raise AssertionError("attention accepted missing or insufficient capacity")


def verify_noisy_attention_clamp_and_events() -> None:
    """Clamp a reproduced overshoot without changing events or random draws."""
    query = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    key = torch.zeros(1, 1, 4, 1, dtype=torch.float64)
    value = torch.full((1, 1, 4, 1), 40.0, dtype=torch.float64)
    try:
        for seed in (0, 1, 2):
            set_gaussian_time_noise(enabled=True, time_std=0.1, deadline_margin=0.4,
                                    seed=seed, device="cpu")
            with patch.object(attention, "attention_output_bounds", _previous_bounds):
                previous = attention.spiking_scaled_dot_product_attention(
                    query, key, value, theta=40.0, source_length_max=4
                )
            previous_stats = deepcopy(get_gaussian_noise_stats())
            previous_state = get_gaussian_time_noise().generator.get_state().clone()

            set_gaussian_time_noise(enabled=True, time_std=0.1, deadline_margin=0.4,
                                    seed=seed, device="cpu")
            result = attention.spiking_scaled_dot_product_attention(
                query, key, value, theta=40.0, source_length_max=4
            )
            stats = deepcopy(get_gaussian_noise_stats())
            assert torch.equal(result, previous.clamp(-40.0, 40.0))
            assert torch.equal(previous_state, get_gaussian_time_noise().generator.get_state())
            output_stats = stats.pop("attention.value_output")
            previous_output_stats = previous_stats.pop("attention.value_output")
            assert stats == previous_stats
            assert output_stats["events"] == previous_output_stats["events"] == 0
            assert output_stats["misses"] == previous_output_stats["misses"] == 0
            assert output_stats["outputs"] == result.numel()
            assert output_stats["output_overflows"] == int((previous > 40.0).sum())
            assert output_stats["output_underflows"] == int((previous < -40.0).sum())
            assert stats["attention.value"]["events"] == value.numel()
            assert stats["attention.value_reference"]["events"] == 1
            assert attention.attention_output_bounds(40.0, 4) == PotentialBounds(-40.0, 40.0)
            if seed == 0:
                assert bool((previous > 40.0).all())
                assert output_stats["output_overflows"] == result.numel()
                assert torch.equal(result, torch.full_like(result, 40.0))
    finally:
        set_gaussian_time_noise(enabled=False)


def verify_mask_and_dropout_compatibility() -> None:
    """Retain mask, dropout, and stochastic evaluation before the final clamp."""
    query = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    key = torch.zeros(1, 1, 4, 1, dtype=torch.float64)
    value = torch.full((1, 1, 4, 1), 40.0, dtype=torch.float64)
    masks = (
        {"attn_mask": torch.tensor([[[[True, False, True, False]]]])},
        {"attn_mask": torch.tensor([[[[0.0, -100.0, 0.0, -100.0]]]])},
        {"is_causal": True},
    )
    set_gaussian_time_noise(enabled=False)
    for options in masks:
        for dropout in (0.0, 0.5):
            torch.manual_seed(23)
            with patch.object(attention, "attention_output_bounds", _previous_bounds):
                previous = attention.spiking_scaled_dot_product_attention(
                    query, key, value, theta=40.0, source_length_max=4,
                    dropout_p=dropout, **options
                )
            previous_state = torch.random.get_rng_state().clone()
            torch.manual_seed(23)
            result = attention.spiking_scaled_dot_product_attention(
                query, key, value, theta=40.0, source_length_max=4,
                dropout_p=dropout, **options
            )
            assert torch.equal(result, previous.clamp(-40.0, 40.0))
            assert torch.equal(previous_state, torch.random.get_rng_state())


if __name__ == "__main__":
    for verify in (verify_attention_output_bounds, verify_clean_attention_parity,
                   verify_noisy_attention_clamp_and_events, verify_mask_and_dropout_compatibility):
        verify()
        print(f"PASS {verify.__name__}")
