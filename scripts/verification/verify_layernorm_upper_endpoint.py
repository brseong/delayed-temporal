"""CPU checks for the LayerNorm positive input range and upper endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds, SpikeSample, TimeBounds

FLAGS = tuple(product((False, True), repeat=3))
DTYPES = (torch.float32, torch.float64)
MODES = ((False, 0.0), (True, 0.0), (True, 0.2))


@dataclass
class _Trace:
    clamps: dict[str, tuple[PotentialBounds, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    logs: list[tuple[PotentialBounds, float, TimeBounds, torch.Tensor]] = field(
        default_factory=list
    )
    differences: list[tuple[TimeBounds, TimeBounds]] = field(default_factory=list)


def _layer(
    flags: tuple[bool, bool, bool],
    dtype: torch.dtype,
    *,
    size: int = 4,
    margin: float = 0.1,
    tau_s: float = 1.0,
) -> SpikingLayerNorm:
    layer = SpikingLayerNorm(
        size,
        theta=4.0,
        tau_s=tau_s,
        clip_margin=margin,
        use_spiking_mul=flags[0],
        use_spiking_log=flags[1],
        use_spiking_expdiff=flags[2],
    ).to(dtype=dtype)
    with torch.no_grad():
        layer.bias.copy_(torch.linspace(-0.25, 0.25, size, dtype=dtype))
    return layer


def _capture(
    layer: SpikingLayerNorm,
    value: torch.Tensor,
    *,
    enabled: bool,
    std: float,
) -> tuple[Potential, _Trace]:
    trace = _Trace()
    original_clamp = PotentialBounds.clamp
    original_log = spiking_ops.neg_log_transform
    original_difference = spiking_ops.exponential_difference_operator

    def capture_clamp(domain, tensor, name=None):
        result = original_clamp(domain, tensor, name=name)
        if name in {
            "x_err_pos_magnitude", "x_err_neg_magnitude",
            "x_err_pos_log_carrier", "x_err_neg_log_carrier",
            "var_x", "layernorm_normalized",
        }:
            trace.clamps[name] = (domain, tensor.detach().clone(), result.detach().clone())
        return result

    def capture_log(tensor, domain, *, tau_s=1.0, **kwargs):
        result = original_log(tensor, domain, tau_s=tau_s, **kwargs)
        time_domain = result.domain if isinstance(result, SpikeSample) else result[1]
        trace.logs.append((domain, tau_s, time_domain, tensor.detach().clone()))
        return result

    def capture_difference(first, first_domain, second, second_domain, **kwargs):
        trace.differences.append((first_domain, second_domain))
        return original_difference(first, first_domain, second, second_domain, **kwargs)

    set_gaussian_time_noise(enabled=enabled, time_std=std, seed=4107)
    with (
        patch.object(PotentialBounds, "clamp", capture_clamp),
        patch.object(spiking_ops, "neg_log_transform", capture_log),
        patch.object(spiking_ops, "exponential_difference_operator", capture_difference),
    ):
        output = layer(Potential(value, PotentialBounds(-8.0, 8.0)))
    return output, trace


def _assert_interval(domain, lower: float, upper: float) -> None:
    assert math.isclose(float(domain.min), lower, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(float(domain.max), upper, rel_tol=1e-12, abs_tol=1e-15)


def _assert_ranges(layer: SpikingLayerNorm, trace: _Trace) -> None:
    for name in ("x_err_pos_magnitude", "x_err_neg_magnitude"):
        _assert_interval(trace.clamps[name][0], 0.0, 4.0)
    for name in ("x_err_pos_log_carrier", "x_err_neg_log_carrier"):
        _assert_interval(trace.clamps[name][0], layer.clip_margin, 4.0)
    _assert_interval(trace.clamps["var_x"][0], layer.clip_margin**2, 16.0)
    deadline = layer.tau_s * math.log(4.0 / layer.clip_margin)
    if layer.use_spiking_log:
        assert len(trace.logs) == 3
        variance, positive, negative = trace.logs
        _assert_interval(variance[0], layer.clip_margin**2, 16.0)
        assert variance[1] == layer.tau_s / 2.0
        for entry in (positive, negative):
            _assert_interval(entry[0], layer.clip_margin, 4.0)
            assert entry[1] == layer.tau_s
        for entry in trace.logs:
            _assert_interval(entry[2], 0.0, deadline)
    else:
        assert trace.logs == []
    assert len(trace.differences) == (2 if layer.use_spiking_expdiff else 0)
    for first, second in trace.differences:
        _assert_interval(first, 0.0, deadline)
        _assert_interval(second, 0.0, deadline)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Positive Input Range]]
def verify_upper_endpoint_and_ablations() -> None:
    """Observe actual domains at and beyond theta in all eight ablations."""
    noisy_misses = 0
    for flags, dtype, tau_s in product(FLAGS, DTYPES, (1.0, 0.75)):
        layer = _layer(flags, dtype, tau_s=tau_s)
        frozen = layer.freeze_parameter_bounds()[2]
        # Unequal feature magnitudes prevent common clipping from cancelling out.
        value = torch.tensor(
            [[-4.0, -1.0, 1.0, 4.0], [-5.0, -1.0, 1.0, 5.0],
             [-4.0, -4.0, 4.0, 4.0]],
            dtype=dtype,
        )
        clean = None
        for enabled, std in MODES:
            output, trace = _capture(layer, value, enabled=enabled, std=std)
            assert output.domain is frozen
            assert bool(torch.isfinite(output.value).all())
            assert bool(((output.value >= frozen.min) & (output.value <= frozen.max)).all())
            if not enabled:
                clean = output.value
            elif std == 0:
                tolerance = 3e-5 if dtype == torch.float32 else 3e-12
                torch.testing.assert_close(output.value, clean, rtol=tolerance, atol=tolerance)
            stats = get_gaussian_noise_stats()
            if not any(flags):
                assert trace.clamps == {} and trace.logs == [] and trace.differences == []
                reference = torch.nn.functional.layer_norm(
                    value, (4,), layer.weight, layer.bias, layer.eps
                )
                assert torch.equal(output.value, reference)
                assert stats == {}
                continue
            _assert_ranges(layer, trace)
            for suffix, expected in (
                ("pos", [[0, 0, 1, 4], [0, 0, 1, 4], [0, 0, 4, 4]]),
                ("neg", [[4, 1, 0, 0], [4, 1, 0, 0], [4, 4, 0, 0]]),
            ):
                magnitude = trace.clamps[f"x_err_{suffix}_magnitude"]
                carrier = trace.clamps[f"x_err_{suffix}_log_carrier"]
                assert torch.equal(magnitude[2], value.new_tensor(expected))
                assert magnitude[1].max().item() == 5.0
                assert magnitude[2].max().item() == 4.0
                assert carrier[2].max().item() == 4.0
            if not enabled or std == 0:
                # The final row reaches theta squared before epsilon is clamped.
                torch.testing.assert_close(
                    trace.clamps["var_x"][2][-1], value.new_tensor([16.0]),
                    rtol=0, atol=3e-5 if dtype == torch.float32 else 3e-12,
                )
            if enabled:
                assert sum(site["events"] for site in stats.values()) > 0
                for site in stats.values():
                    assert 0 <= site["misses"] <= site["events"]
                if std == 0:
                    assert all(site["misses"] == 0 for site in stats.values())
                else:
                    noisy_misses += sum(site["misses"] for site in stats.values())
            else:
                assert stats == {}
        # With direct variance and division, the endpoint effect has a simple reference.
        if flags == (False, True, False):
            centered = value.clamp(-4.0, 4.0)
            variance = (centered.square().mean(-1, keepdim=True) + layer.eps).clamp(
                layer.clip_margin**2, 16.0
            )
            expected = centered / variance.sqrt() + layer.bias
            tolerance = 3e-5 if dtype == torch.float32 else 3e-12
            torch.testing.assert_close(clean, expected, rtol=tolerance, atol=tolerance)
    assert noisy_misses > 0


def verify_positive_floor_and_inactive_values() -> None:
    """Keep zero magnitudes separate from positive logarithmic input copies."""
    for flags, dtype in product(FLAGS, DTYPES):
        # Binary-exact values isolate equality at the floor from centering roundoff.
        layer = _layer(flags, dtype, size=8, margin=0.125)
        value = torch.tensor(
            [[-0.125, -0.0625, 0, 0, 0, 0, 0.0625, 0.125]], dtype=dtype
        )
        assert value.mean().item() == 0.0
        for enabled, std in MODES:
            _, trace = _capture(layer, value, enabled=enabled, std=std)
            if not any(flags):
                continue
            _assert_ranges(layer, trace)
            for suffix in ("pos", "neg"):
                magnitude = trace.clamps[f"x_err_{suffix}_magnitude"][2]
                carrier = trace.clamps[f"x_err_{suffix}_log_carrier"][2]
                assert bool((magnitude == 0).any())
                assert bool((magnitude == 0.0625).any())
                assert bool((magnitude == 0.125).any())
                assert torch.equal(carrier, torch.full_like(carrier, 0.125))
            normalized = trace.clamps["layernorm_normalized"][1]
            assert torch.equal(normalized[:, 1:-1], torch.zeros_like(normalized[:, 1:-1]))
            if not enabled or std == 0:
                assert normalized[0, 0] < 0 and normalized[0, -1] > 0
                expected_variance = value.square().mean(-1, keepdim=True) + layer.eps
                torch.testing.assert_close(
                    trace.clamps["var_x"][1], expected_variance,
                    rtol=0, atol=3e-6 if dtype == torch.float32 else 3e-12,
                )

        for enabled in (False, True):
            constant = torch.full_like(value, 1.25)
            output, trace = _capture(layer, constant, enabled=enabled, std=0.0)
            tolerance = 3e-5 if dtype == torch.float32 else 3e-12
            torch.testing.assert_close(
                output.value, layer.bias.expand_as(value), rtol=0, atol=tolerance
            )
            if any(flags):
                for suffix in ("pos", "neg"):
                    magnitude = trace.clamps[f"x_err_{suffix}_magnitude"][2]
                    assert torch.equal(magnitude, torch.zeros_like(magnitude))
                torch.testing.assert_close(
                    trace.clamps["var_x"][1], value.new_full((1, 1), layer.eps),
                    rtol=0, atol=3e-6 if dtype == torch.float32 else 3e-12,
                )


def verify_margin_validation_and_cache() -> None:
    """Allow every finite positive floor below theta and reject stale caches."""
    for margin in (2.0, 3.0):
        layer = _layer((True, True, True), torch.float64, margin=margin)
        layer.freeze_parameter_bounds()
        for enabled in (False, True):
            output, trace = _capture(
                layer, torch.tensor([[-4.0, -1.0, 1.0, 4.0]], dtype=torch.float64),
                enabled=enabled, std=0.0,
            )
            assert bool(torch.isfinite(output.value).all())
            _assert_ranges(layer, trace)

    for margin in (0.0, -0.1, 4.0, 4.1, math.nan, math.inf, -math.inf):
        try:
            _layer((True, True, True), torch.float64, margin=margin)
        except ValueError:
            pass
        else:
            raise AssertionError("LayerNorm constructor accepted an invalid clip_margin")
        layer = _layer((True, True, True), torch.float64)
        layer.freeze_parameter_bounds()
        layer.clip_margin = margin
        for refresh in (False, True):
            try:
                layer.freeze_parameter_bounds(refresh=refresh)
            except ValueError:
                pass
            else:
                raise AssertionError("LayerNorm bounds accepted an invalid clip_margin")

    layer = _layer((True, True, True), torch.float64)
    original = layer.freeze_parameter_bounds()
    layer.clip_margin = 3.0
    try:
        layer.freeze_parameter_bounds()
    except RuntimeError:
        pass
    else:
        raise AssertionError("LayerNorm reused bounds after clip_margin changed")
    refreshed = layer.freeze_parameter_bounds(refresh=True)
    assert refreshed is not original
    assert refreshed[2] == original[2]
    assert layer.freeze_parameter_bounds() is refreshed
    assert SpikingLayerNorm(4).clip_margin == 1e-5


def main() -> None:
    try:
        verify_upper_endpoint_and_ablations()
        verify_positive_floor_and_inactive_values()
        verify_margin_validation_and_cache()
    finally:
        set_gaussian_time_noise(enabled=False)
    print("LayerNorm positive input range: 3 verification groups passed")


if __name__ == "__main__":
    main()
