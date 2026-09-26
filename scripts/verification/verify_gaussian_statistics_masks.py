"""Verify active signed-branch counts without changing numerical execution."""

from contextlib import nullcontext
from itertools import product
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.transforms import noise
from utils.transforms.functions import gelu_cubic_power_operator
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.types import Potential, PotentialBounds
from scripts.verification.verify_layernorm_calibrated_bounds import _bind, _layer


def replay(call, *, unmasked=False, std=1e-5):
    noise.set_gaussian_time_noise(enabled=True, time_std_fraction=std, seed=0)
    # Recover prior accounting only; do not replace any model computation.
    with patch.object(noise, "_statistics_mask_for", return_value=None) if unmasked else nullcontext():
        result = call()
    value = result.value if isinstance(result, Potential) else result[0]
    return value, noise.get_gaussian_time_noise().generator.get_state(), noise.get_gaussian_noise_stats()


def check_parity(call, *, std=1e-5):
    masked = replay(call, std=std)
    original = replay(call, unmasked=True, std=std)
    assert torch.equal(masked[0], original[0]), "accounting changed output"
    assert torch.equal(masked[1], original[1]), "accounting changed RNG state"
    return masked[2], original[2]


def verify_signed_branch_counts():
    domain = PotentialBounds(-2.0, 2.0)
    for values in (
        torch.ones(10000, dtype=torch.float64),
        -torch.ones(10000, dtype=torch.float64),
        torch.zeros(8, dtype=torch.float64),
        torch.tensor([-1.0, -1e-5, -1e-6, 0.0, 1e-6, 1e-5, 1.0], dtype=torch.float64),
    ):
        stats, old = check_parity(lambda: gelu_cubic_power_operator(values, domain, tau_s=1.0))
        pos, neg = int((values >= 1e-5).sum()), int((values <= -1e-5).sum())
        assert stats["gelu.cubic.log_positive"]["events"] == pos
        assert stats["gelu.cubic.log_negative"]["events"] == neg
        assert stats["gelu.cubic.log_reference"]["events"] == int(pos + neg > 0)
        assert stats["exponential_difference.internal"]["events"] == pos + neg
        assert stats["exponential_difference.output"]["outputs"] == pos + neg
        for site, active in (("gelu.cubic.log_positive", pos), ("gelu.cubic.log_negative", neg)):
            assert stats[site]["misses"] <= active
            assert stats[site]["deadline_events"] <= active
            if active == 0:
                assert stats[site]["misses"] == stats[site]["deadline_events"] == 0
        if values.numel() == 10000 and pos == 10000:
            assert old["gelu.cubic.log_negative"]["misses"] == 4930

    # Include a constant row, exact-zero centered features, and both active signs.
    values = torch.tensor([[-1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    potential = Potential(values, domain)
    for flags in product((False, True), repeat=3):
        layer = _layer(flags)
        _bind(layer, 2.0)
        for std in (0.0, 0.03):
            stats, _ = check_parity(lambda: layer(potential), std=std)
            if flags[1]:
                assert stats["layernorm.log_positive"]["events"] == 1
                assert stats["layernorm.log_negative"]["events"] == 1
                assert stats["layernorm.log_sigma"]["events"] == 1
            else:
                assert "layernorm.log_positive" not in stats
            if flags[2]:
                assert stats["exponential_difference.internal"]["events"] == 2
                assert stats["exponential_difference.output"]["outputs"] == 2


def verify_mask_contract():
    value = torch.tensor([[-3.0, 3.0], [-3.0, 3.0]], dtype=torch.float64)
    domain = PotentialBounds(-2.0, 2.0)
    noise.set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, time_mean=10.0)
    with noise.gaussian_noise_statistics_mask(torch.tensor([[True], [False]])):
        with noise.gaussian_noise_statistics_mask(torch.tensor([False, True])):
            result = noise.clamp_gaussian_output(value, domain, site="test.output", name="test")
            event = neg_identity_transform(value.clamp(-2, 2), domain,
                return_spike_sample=True, noise_site="test.event")
    assert torch.equal(result, value.clamp(-2, 2))
    assert not event.fired.any()
    stats = noise.get_gaussian_noise_stats()
    assert stats["test.output"]["outputs"] == 1
    assert stats["test.output"]["output_underflows"] == 0
    assert stats["test.output"]["output_overflows"] == 1
    assert stats["test.event"]["events"] == stats["test.event"]["misses"] == 1
    assert stats["test.event"]["deadline_events"] == 0
    try:
        with noise.gaussian_noise_statistics_mask(torch.tensor(False)):
            raise RuntimeError("test exit")
    except RuntimeError:
        pass
    assert noise._statistics_mask_for(value) is None
    try:
        with noise.gaussian_noise_statistics_mask(torch.ones(2)):
            pass
    except TypeError:
        pass
    else:
        raise AssertionError("non-boolean mask accepted")
    try:
        with noise.gaussian_noise_statistics_mask(torch.ones(2, dtype=torch.bool)):
            noise._statistics_mask_for(torch.tensor(0.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("scalar event expanded into consumer shape")


if __name__ == "__main__":
    torch.set_num_threads(1)
    try:
        verify_signed_branch_counts()
        verify_mask_contract()
        print("Gaussian signed-branch statistics verification passed.")
    finally:
        noise.set_gaussian_time_noise(enabled=False)
