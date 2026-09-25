"""CPU verification of ViT LayerNorm internal calibration."""

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
from utils.transformers.calibration import (
    OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
    VIT_CALIBRATION_POLICY_VERSION,
    bind_model_calibration,
    clear_model_calibration,
)
from utils.transforms.calibration import (
    CalibrationMetadata,
    CalibrationMode,
    CalibrationRangePolicy,
    LayerCalibrationSpec,
    calibration_table_from_dict,
    calibration_table_to_dict,
    create_calibration_collector,
    create_calibration_runtime,
    finalize_calibration_collection,
    observe_calibration_activation,
    start_histogram_calibration_pass,
)
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds, SpikeSample


FLAGS = tuple(product((False, True), repeat=3))
KEY = ("", "centered_input")


def _metadata(*, policy: int | None = VIT_CALIBRATION_POLICY_VERSION, dtype: str = "float64") -> CalibrationMetadata:
    return CalibrationMetadata(
        model_family="vit", model_id="layernorm-test", dataset_id="fixture",
        dataset_split="train", preprocessing="none", dtype=dtype,
        tau_s=1.0, tau_m=1.0, clip_margin=1e-5,
        max_sequence_length=None, input_shape=(4,),
        model_options=(("layer_norm_clip_margin", 1e-5), ("layer_norm_eps", 1e-12))
        + (() if policy is None else (
            ("operator_backed_output_head_version", OPERATOR_BACKED_OUTPUT_HEAD_VERSION),
            ("vit_calibration_policy_version", policy),
        )),
    )


def _collector(*, margin: float = 0.0, policy: int = VIT_CALIBRATION_POLICY_VERSION, dtype: str = "float64"):
    return create_calibration_collector(
        _metadata(policy=policy, dtype=dtype),
        (LayerCalibrationSpec("", "centered_input", CalibrationRangePolicy.SIGNED_SYMMETRIC,
                              0.0, 1.0, margin),),
        bin_count=32,
    )


def _table(radius: float, *, policy: int = VIT_CALIBRATION_POLICY_VERSION):
    collector = _collector(policy=policy)
    values = torch.tensor([-radius, radius], dtype=torch.float64)
    observe_calibration_activation(collector, *KEY, values)
    start_histogram_calibration_pass(collector)
    observe_calibration_activation(collector, *KEY, values)
    return finalize_calibration_collection(collector)


def _layer(flags=(True, True, True), *, dtype=torch.float64):
    layer = SpikingLayerNorm(
        4, eps=1e-12, clip_margin=1e-5,
        use_spiking_mul=flags[0], use_spiking_log=flags[1],
        use_spiking_expdiff=flags[2],
    ).to(dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([3, 4, 5, 6], dtype=dtype))
        layer.bias.copy_(torch.tensor([-0.2, 0, 0.1, 0.3], dtype=dtype))
    return layer


def _bind(layer, radius: float, *, policy: int = VIT_CALIBRATION_POLICY_VERSION):
    table = _table(radius, policy=policy)
    restored = calibration_table_from_dict(calibration_table_to_dict(table))
    assert restored == table
    runtime = create_calibration_runtime(
        CalibrationMode.VALIDATE, restored, expected_metadata=table.metadata
    )
    bind_model_calibration(layer, runtime)
    return runtime


def _reference(layer, value, radius):
    if not any((layer.use_spiking_mul, layer.use_spiking_log, layer.use_spiking_expdiff)):
        return torch.nn.functional.layer_norm(value, (4,), layer.weight, layer.bias, layer.eps)
    centered = (value - value.mean(-1, keepdim=True)).clamp(-radius, radius)
    variance = (centered.square().mean(-1, keepdim=True) + layer.eps).clamp(
        layer.clip_margin**2, radius**2 + layer.eps
    )
    centered = torch.where(centered.abs() >= layer.clip_margin, centered, 0.0)
    return centered / variance.sqrt() * layer.weight + layer.bias


def _capture(layer, pot, *, enabled=False, std=0.0, seed=171):
    logs, multiplications, clamps = [], [], {}
    original_log = spiking_ops.neg_log_transform
    original_mul = spiking_ops.multiplication_operator
    original_clamp = PotentialBounds.clamp

    def log(value, bounds, **kwargs):
        result = original_log(value, bounds, **kwargs)
        time_bounds = result.domain if isinstance(result, SpikeSample) else result[1]
        logs.append((bounds, kwargs["tau_s"], time_bounds))
        return result

    def multiply(value, bounds, factor, factor_bounds):
        multiplications.append(float(factor_bounds.max))
        return original_mul(value, bounds, factor, factor_bounds)

    def clamp(bounds, value, name=None):
        result = original_clamp(bounds, value, name=name)
        if name in {"x_err_pos_magnitude", "x_err_neg_magnitude", "var_x"}:
            clamps[name] = (bounds, result.detach().clone())
        return result

    set_gaussian_time_noise(enabled=enabled, time_std_fraction=std, seed=seed)
    with patch.object(spiking_ops, "neg_log_transform", log), \
         patch.object(spiking_ops, "multiplication_operator", multiply), \
         patch.object(PotentialBounds, "clamp", clamp):
        result = layer(pot)
    return result, logs, multiplications, clamps


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT LayerNorm Internal Calibration]]
def verify_selected_ranges_and_ablations():
    """Selected bounds affect internal encoding without changing gamma's encoder."""
    values = {
        60.0: [[-50, -10, 10, 50], [-80, -20, 20, 80]],
        1.0: [[-0.8, -0.2, 0.2, 0.8], [-2, -0.5, 0.5, 2]],
    }
    for flags, radius, dtype in product(FLAGS, (60.0, 1.0), (torch.float32, torch.float64)):
        layer = _layer(flags, dtype=dtype)
        runtime = _bind(layer, radius)
        output_domain = layer.freeze_parameter_bounds()[2]
        value = torch.tensor(values[radius], dtype=dtype)
        pot = Potential(value, PotentialBounds(-100, 100))
        clean = None
        for enabled, std in ((False, 0.0), (True, 0.0), (True, 1e-9)):
            out, logs, multiplications, clamps = _capture(layer, pot, enabled=enabled, std=std)
            assert out.domain is output_domain and torch.isfinite(out.value).all()
            assert layer.eps == 1e-12 and layer.clip_margin == 1e-5
            if not enabled:
                clean = out.value
                tolerance = 3e-4 if dtype == torch.float32 else 2e-10
                torch.testing.assert_close(clean, _reference(layer, value, radius),
                                           rtol=tolerance, atol=tolerance)
            elif std == 0:
                tolerance = 3e-4 if dtype == torch.float32 else 2e-10
                torch.testing.assert_close(out.value, clean, rtol=tolerance, atol=tolerance)
            if not any(flags):
                assert not logs and not multiplications and not clamps
                assert runtime.clipping_counts[KEY].num_values == 0
                assert get_gaussian_noise_stats() == {}
                continue
            expected_multiplications = ([radius, radius] if flags[0] else [])
            if flags[2]:
                expected_multiplications.append(6.0)
            assert multiplications == expected_multiplications
            for name in ("x_err_pos_magnitude", "x_err_neg_magnitude"):
                assert clamps[name][0] == PotentialBounds(0, radius)
            log_radius = math.sqrt(radius**2 + layer.eps)
            assert clamps["var_x"][0] == PotentialBounds(layer.clip_margin**2, log_radius**2)
            if flags[1]:
                assert [entry[0] for entry in logs] == [
                    PotentialBounds(layer.clip_margin**2, log_radius**2),
                    PotentialBounds(layer.clip_margin, log_radius),
                    PotentialBounds(layer.clip_margin, log_radius),
                ]
                assert [entry[1] for entry in logs] == [0.5, 1.0, 1.0]
                for entry in logs:
                    assert math.isclose(entry[2].max, math.log(log_radius/layer.clip_margin))
            else:
                assert not logs
        if any(flags):
            counts = runtime.clipping_counts[KEY]
            assert (counts.num_values, counts.underflows, counts.overflows) == (24, 3, 3)


def verify_collection_and_partitioning():
    """Collect signed values before clipping using one analytic width in both passes."""
    value = torch.tensor([[-30, 30, 30, 30], [-10, 10, -10, 10]], dtype=torch.float64)
    tables = []
    for partitions in ((value,), (value[:1], value[1:])):
        layer = _layer()
        collector = _collector(margin=0.05)
        bind_model_calibration(layer, collector)
        for pass_index in range(2):
            if pass_index:
                start_histogram_calibration_pass(collector)
            for part in partitions:
                out, logs, multiplications, _ = _capture(
                    layer, Potential(part, PotentialBounds(-30, 30))
                )
                assert multiplications == [60, 60, 6]
                assert logs[1][0] == PotentialBounds(
                    1e-5,
                    math.sqrt(60**2 + layer.eps),
                )
                torch.testing.assert_close(out.value, _reference(layer, part, 60), rtol=2e-10, atol=2e-10)
        observer = collector.min_max_states[KEY]
        assert (observer.observed_min, observer.observed_max, observer.num_values) == (-45, 15, 8)
        table = finalize_calibration_collection(collector)
        clear_model_calibration(layer, expected_state=collector)
        tables.append(table)
        assert table.layers[0].bounds.min == -49.5 and table.layers[0].bounds.max == 49.5
        assert table.layers[0].histogram.num_values == 8
    assert tables[0] == tables[1]

    table = tables[0]
    eval_value = torch.tensor([[-90, 30, 30, 30], [-10, 10, -10, 10]], dtype=torch.float64)
    outputs, counts = [], []
    for partitions in ((eval_value,), (eval_value[:1], eval_value[1:])):
        layer = _layer()
        runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table,
                                             expected_metadata=table.metadata)
        bind_model_calibration(layer, runtime)
        outputs.append(torch.cat([layer(Potential(part, PotentialBounds(-100, 100))).value
                                  for part in partitions]))
        counts.append(runtime.clipping_counts[KEY])
        assert runtime.table == table
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-10, atol=2e-10)
    assert counts[0] == counts[1]
    assert (counts[0].num_values, counts[0].underflows, counts[0].overflows) == (8, 1, 0)


def verify_legacy_and_dense_parity():
    """Unbound and old-policy execution use the incoming range; dense stays dense."""
    value = torch.tensor([[-30, 30, 30, 30]], dtype=torch.float64)
    pot = Potential(value, PotentialBounds(-30, 30))
    for flags in FLAGS:
        layer = _layer(flags)
        clean = layer(pot)
        runtime = _bind(layer, 1, policy=None)
        bound, logs, multiplications, _ = _capture(layer, pot)
        assert torch.equal(clean.value, bound.value) and clean.domain is bound.domain
        assert runtime.clipping_counts[KEY].num_values == 0
        assert all(bound_max in {6.0, 60.0} for bound_max in multiplications)
        if any(flags):
            torch.testing.assert_close(bound.value, _reference(layer, value, 60), rtol=2e-10, atol=2e-10)
        if flags[1]:
            assert logs[1][0] == PotentialBounds(
                1e-5,
                math.sqrt(60**2 + layer.eps),
            )


def verify_floor_and_epsilon():
    """Changing the internal upper bound does not alter either fixed positive floor."""
    for flags in FLAGS:
        layer = _layer(flags)
        _bind(layer, 1)
        value = torch.tensor([[-1e-6, 0, 0, 1e-6]], dtype=torch.float64)
        out, _, _, clamps = _capture(layer, Potential(value, PotentialBounds(-2, 2)))
        if any(flags):
            torch.testing.assert_close(out.value, layer.bias.unsqueeze(0), rtol=0, atol=2e-12)
            assert clamps["var_x"][1].item() == layer.clip_margin**2
        else:
            torch.testing.assert_close(out.value, _reference(layer, value, 1), rtol=0, atol=0)


def verify_invalid_ranges():
    """Invalid static or persisted bounds fail without inventing a replacement."""
    for radius in (1e-6, 1e-5, 1e160):
        layer = _layer()
        try:
            _bind(layer, radius)
            layer(Potential(torch.tensor([[-1, 0, 0, 1]], dtype=torch.float64), PotentialBounds(-2, 2)))
        except ValueError as error:
            assert "centered_input" in str(error) or "incoming interval width" in str(error)
        else:
            raise AssertionError(f"accepted invalid centered_input radius {radius}")
    for bounds in (PotentialBounds(0, 0), PotentialBounds(-1e308, 1e308),
                   PotentialBounds(-1e160, 1e160)):
        layer = _layer()
        collector = _collector()
        bind_model_calibration(layer, collector)
        try:
            layer(Potential(torch.zeros(1, 4, dtype=torch.float64), bounds))
        except ValueError as error:
            assert "centered_input" in str(error) or "incoming interval width" in str(error)
        else:
            raise AssertionError(f"accepted invalid collection bounds {bounds}")
        assert collector.min_max_states == {}


def verify_seeded_noise_and_output_cache():
    """Seeded delivery changes values, never the fixed range or affine cache."""
    layer = _layer()
    _bind(layer, 60)
    domain = layer.freeze_parameter_bounds()[2]
    value = torch.tensor([[-50, -10, 10, 50]], dtype=torch.float64)
    pot = Potential(value, PotentialBounds(-100, 100))
    first = _capture(layer, pot, enabled=True, std=1e-5, seed=3)[0]
    second = _capture(layer, pot, enabled=True, std=1e-5, seed=3)[0]
    other = _capture(layer, pot, enabled=True, std=1e-5, seed=4)[0]
    assert torch.equal(first.value, second.value)
    assert not torch.equal(first.value, other.value)
    assert first.domain is second.domain is other.domain is domain
    assert torch.isfinite(other.value).all()
    clear_model_calibration(layer)
    _bind(layer, 1)
    assert layer.freeze_parameter_bounds()[2] is domain


def main():
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    try:
        for verification in (
            verify_selected_ranges_and_ablations, verify_collection_and_partitioning,
            verify_legacy_and_dense_parity, verify_floor_and_epsilon,
            verify_invalid_ranges, verify_seeded_noise_and_output_cache,
        ):
            set_gaussian_time_noise(enabled=False)
            verification()
        print("ViT LayerNorm internal calibration: six verification groups passed")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
