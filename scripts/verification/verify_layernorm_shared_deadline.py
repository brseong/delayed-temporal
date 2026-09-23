"""CPU checks for one shared LayerNorm logarithmic deadline."""

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

from scripts.verification.verify_layernorm_calibrated_bounds import (
    FLAGS, _bind, _layer, _reference,
)
from utils.transformers.models import spiking_ops
from utils.transforms import noise
from utils.transforms.noise import get_gaussian_time_noise, set_gaussian_time_noise
from utils.transforms.potential_to_spike import neg_log_transform
from utils.transforms.primitive import (
    signed_pulse_width_modulation_operator,
    unsigned_pulse_width_modulation_operator,
)
from utils.transforms.types import Potential, PotentialBounds, SpikeSample, TimeBounds


RADII = (40.007, 40.172)


def _expect_error(callback, text=None):
    try:
        callback()
    except (TypeError, ValueError) as error:
        if text is not None:
            assert text in str(error), str(error)
    else:
        raise AssertionError("invalid deadline was accepted")


def _capture(layer, potential, *, enabled, std=0.0, seed=19):
    records, sampled = [], []
    active = []
    original_log = spiking_ops.neg_log_transform
    original_sampler = noise._sample_gaussian_spike_time

    def log(value, domain, **kwargs):
        supplied = kwargs.get("shared_time_bounds")
        active.append(supplied)
        try:
            result = original_log(value, domain, **kwargs)
        finally:
            active.pop()
        returned = result.domain if isinstance(result, SpikeSample) else result[1]
        records.append((domain, supplied, returned))
        return result

    def sample(value, **kwargs):
        if active:
            assert kwargs["domain"] is active[-1]
            sampled.append(kwargs["domain"])
        return original_sampler(value, **kwargs)

    set_gaussian_time_noise(enabled=enabled, time_std_fraction=std, seed=seed)
    with patch.object(spiking_ops, "neg_log_transform", log), \
         patch.object(noise, "_sample_gaussian_spike_time", sample):
        output = layer(potential)
    return output, records, sampled


def verify_rounding_directions():
    """The common epsilon-aware radius gives both encoders one exact deadline."""
    directions = []
    for radius in RADII:
        log_radius = math.sqrt(radius**2 + 1.0e-12)
        magnitude = math.log(log_radius) - math.log(1e-5)
        variance = 0.5 * (math.log(log_radius**2) - math.log(1e-5**2))
        directions.append((magnitude - variance) / math.ulp(magnitude))
    assert directions == [0.0, 0.0], directions


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Shared Log Deadline]]
def verify_layernorm_shared_deadline():
    """All three encoders and their samplers receive the same frozen object."""
    for flags, radius, dtype in product(FLAGS, RADII, (torch.float32, torch.float64)):
        layer = _layer(flags, dtype=dtype)
        _bind(layer, radius)
        output_domain = layer.freeze_parameter_bounds()[2]
        value = torch.tensor([
            [-radius * 1.5, -0.2, 0.2, radius * 1.5],
            [-1e-6, 0.0, 0.0, 1e-6],
            [-radius * 0.75, -radius * 0.25, radius * 0.25, radius * 0.75],
        ], dtype=dtype)
        potential = Potential(value, PotentialBounds(-100, 100))
        tolerance = 3e-4 if dtype == torch.float32 else 2e-10
        clean = None
        for enabled, std in ((False, 0.0), (True, 0.0), (True, 1e-6)):
            output, records, sampled = _capture(
                layer, potential, enabled=enabled, std=std
            )
            assert output.domain is output_domain
            assert torch.isfinite(output.value).all()
            assert layer.eps == 1e-12 and layer.clip_margin == 1e-5
            if not enabled:
                clean = output.value
                torch.testing.assert_close(
                    output.value, _reference(layer, value, radius),
                    rtol=tolerance, atol=tolerance,
                )
            elif std == 0.0:
                torch.testing.assert_close(output.value, clean, rtol=tolerance, atol=tolerance)
            else:
                replay, replay_records, _ = _capture(
                    layer, potential, enabled=True, std=std
                )
                assert torch.equal(output.value, replay.value)
                assert replay.domain is output_domain
                assert [row[2] for row in records] == [row[2] for row in replay_records]

            if flags[1]:
                assert len(records) == 3
                shared = records[0][1]
                assert isinstance(shared, TimeBounds)
                assert shared.min == 0.0
                log_radius = math.sqrt(radius**2 + layer.eps)
                assert shared.max == math.log(log_radius) - math.log(layer.clip_margin)
                assert all(supplied is returned is shared for _, supplied, returned in records)
                assert [row[0] for row in records] == [
                    PotentialBounds(layer.clip_margin**2, log_radius**2),
                    PotentialBounds(layer.clip_margin, log_radius),
                    PotentialBounds(layer.clip_margin, log_radius),
                ]
                assert len(sampled) == (3 if enabled else 0)
                assert all(domain is shared for domain in sampled)
            else:
                assert not records and not sampled


def verify_encoder_validation_and_rng():
    """Explicit sharing permits rounding only and rejects invalid values before sampling."""
    radius = RADII[0]
    domain = PotentialBounds(1e-5**2, radius**2)
    value = torch.tensor([domain.min, 1.0, domain.max], dtype=torch.float64)
    shared = TimeBounds(0.0, math.log(radius) - math.log(1e-5))
    set_gaussian_time_noise(enabled=False)
    native, native_domain = neg_log_transform(value, domain, tau_s=0.5)
    explicit, explicit_domain = neg_log_transform(
        value, domain, tau_s=0.5, shared_time_bounds=shared
    )
    assert native_domain.max != shared.max
    assert explicit_domain is shared
    torch.testing.assert_close(native, explicit, rtol=0, atol=math.ulp(shared.max))

    set_gaussian_time_noise(enabled=True, time_std_fraction=1e-6, seed=23)
    generator = get_gaussian_time_noise().generator
    for invalid in (
        "invalid", PotentialBounds(0.0, shared.max),
        TimeBounds(0.01, shared.max), TimeBounds(0.0, shared.max + 0.01),
        TimeBounds(0.0, shared.max - 0.01), TimeBounds(0.0, 0.0),
    ):
        before = generator.get_state().clone()
        with patch.object(noise, "_sample_gaussian_spike_time") as sampler:
            _expect_error(lambda: neg_log_transform(
                value, domain, tau_s=0.5, shared_time_bounds=invalid,
                return_spike_sample=True,
            ))
            sampler.assert_not_called()
        assert torch.equal(before, generator.get_state())
    set_gaussian_time_noise(enabled=False)


def verify_primitive_deadline_stays_strict():
    """The generic PWM operators still reject even one ULP of deadline underflow."""
    deadline = 2.0
    wider = TimeBounds(0.0, math.nextafter(deadline, math.inf))
    normal = TimeBounds(0.0, deadline)
    time = torch.tensor([0.5], dtype=torch.float64)
    drive = torch.ones(1, dtype=torch.float64)
    drive_domain = PotentialBounds(1.0, 1.0)
    _expect_error(lambda: unsigned_pulse_width_modulation_operator(
        time, wider, drive, drive_domain, observation_deadline=deadline
    ), "must not precede")
    _expect_error(lambda: signed_pulse_width_modulation_operator(
        time, normal, time, wider, drive, drive_domain,
        observation_deadline=deadline,
    ), "must not precede")
    event = SpikeSample(time=time, fired=torch.ones_like(time, dtype=torch.bool), domain=normal)
    _expect_error(lambda: signed_pulse_width_modulation_operator(
        event, normal, time, normal, drive, drive_domain,
        observation_deadline=math.nextafter(deadline, math.inf),
    ), "must equal")


def main():
    torch.set_num_threads(1)
    try:
        for verification in (
            verify_rounding_directions, verify_layernorm_shared_deadline,
            verify_encoder_validation_and_rng, verify_primitive_deadline_stays_strict,
        ):
            set_gaussian_time_noise(enabled=False)
            verification()
        print("LayerNorm shared logarithmic deadline: four verification groups passed")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
