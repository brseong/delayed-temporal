"""CPU checks for attention-local ranges and its final output clamp."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.transformers.integrations import spiking_sdpa_attention as attention
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms.types import PotentialBounds


# @lat: [[bounds-audit#2026-09-14 Bound Corrections#Attention Output Bounds]]
def verify_attention_output_bounds() -> None:
    """Use the declared value range for output and dtype limits for scores."""
    for dtype in (torch.float32, torch.float64):
        for capacity in (1, 5, 197):
            bounds = attention.attention_score_representability_bounds(1.0, capacity, dtype)
            expected = 0.5 * (
                -math.log(torch.finfo(dtype).tiny) - math.log(capacity) - 2.0
            )
            assert math.isclose(bounds.max, expected)
            assert bounds.min == -bounds.max
            assert bounds is attention.attention_score_representability_bounds(
                1.0, capacity, dtype
            )


def verify_clean_attention_parity() -> None:
    """Preserve normalized attention while requiring all three local ranges."""
    generator = torch.Generator().manual_seed(51)
    query = 0.2 * torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
    key = 0.2 * torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(1, 2, 5, 3, generator=generator, dtype=torch.float64)
    options = {
        "source_length_max": 5,
        "query_bounds": PotentialBounds(-0.8, 0.8),
        "key_bounds": PotentialBounds(-0.8, 0.8),
        "value_bounds": PotentialBounds(-3.0, 3.0),
    }
    set_gaussian_time_noise(enabled=False)
    result = attention.spiking_scaled_dot_product_attention(query, key, value, **options)
    expected = torch.softmax(query @ key.transpose(-2, -1) / 2.0, dim=-1) @ value
    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)
    assert bool(((result >= -3.0) & (result <= 3.0)).all())
    assert get_gaussian_noise_stats() == {}
    for missing in options:
        if not missing.endswith("_bounds"):
            continue
        incomplete = {name: item for name, item in options.items() if name != missing}
        try:
            attention.spiking_scaled_dot_product_attention(query, key, value, **incomplete)
        except ValueError:
            pass
        else:
            raise AssertionError(f"attention accepted missing {missing}")


def verify_noisy_attention_local_clamp() -> None:
    """Replay seeded timing noise without widening the declared value range."""
    query = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    key = torch.zeros(1, 1, 4, 1, dtype=torch.float64)
    value = torch.full((1, 1, 4, 1), 7.0, dtype=torch.float64)
    options = {
        "source_length_max": 4,
        "query_bounds": PotentialBounds(-1.0, 1.0),
        "key_bounds": PotentialBounds(-1.0, 1.0),
        "value_bounds": PotentialBounds(-7.0, 7.0),
    }
    try:
        outputs = []
        for seed in (11, 11, 12):
            set_gaussian_time_noise(
                enabled=True, time_std_fraction=0.05,
                deadline_margin_std_ratio=4.0, seed=seed, device="cpu",
            )
            output = attention.spiking_scaled_dot_product_attention(
                query, key, value, **options
            )
            assert bool(torch.isfinite(output).all())
            assert bool(((output >= -7.0) & (output <= 7.0)).all())
            outputs.append(output)
        assert torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[0], outputs[2])
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    for verification in (
        verify_attention_output_bounds,
        verify_clean_attention_parity,
        verify_noisy_attention_local_clamp,
    ):
        verification()
        print(f"PASS {verification.__name__}")
