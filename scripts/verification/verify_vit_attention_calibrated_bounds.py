"""CPU checks for calibrated ViT attention input and output bounds."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import sys
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transforms.calibration import (
    CalibrationMetadata, CalibrationMode, CalibrationRangePolicy, LayerCalibrationSpec,
    create_calibration_collector, create_calibration_runtime,
    finalize_calibration_collection, start_histogram_calibration_pass,
)
from utils.transforms.noise import (
    get_gaussian_noise_stats, get_gaussian_time_noise, set_gaussian_time_noise,
)
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import (
    bind_model_calibration, clear_model_calibration, vit_calibration_uses_explicit_bounds,
)
from utils.transformers.integrations import spiking_sdpa_attention as attention


def _inputs():
    query = torch.tensor([[[[3.0, 0.5], [-1.0, 2.5]]]], dtype=torch.float64)
    key = torch.tensor([[[[6.0, -1.0], [-2.0, 5.0], [1.0, 1.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[9.0, -7.0], [-5.0, 8.0], [3.0, -1.0]]]], dtype=torch.float64)
    options = dict(
        theta=2.0, source_length_max=3,
        query_bounds=PotentialBounds(-4.0, 4.0),
        key_bounds=PotentialBounds(-8.0, 8.0),
        value_bounds=PotentialBounds(-12.0, 12.0),
    )
    return query, key, value, options


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT Attention Bound Transfer]]
def verify_explicit_attention_bounds() -> None:
    """Retain distinct bounds above global theta without an extra key clamp."""
    query, key, value, options = _inputs()
    expected = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(2.0), -1) @ value
    with patch.object(
        attention, "scaled_dot_product_function",
        wraps=attention.scaled_dot_product_function,
    ) as multiply:
        result = attention.spiking_scaled_dot_product_attention(query, key, value, **options)
    arguments = multiply.call_args.args
    assert arguments[1] == options["query_bounds"]
    assert arguments[3] == options["key_bounds"]
    assert arguments[4] == 8.0
    torch.testing.assert_close(result, expected, rtol=1.0e-11, atol=1.0e-11)
    assert bool((result.abs() > 2.0).any())
    legacy = attention.spiking_scaled_dot_product_attention(
        query, key, value, theta=2.0, source_length_max=3,
    )
    assert bool((legacy.abs() <= 2.0).all())
    assert not torch.allclose(result, legacy)
    assert get_gaussian_noise_stats() == {}


def verify_score_numeric_ceiling() -> None:
    """Remove only the new path's global threshold cap, retaining numeric safety."""
    for dtype in (torch.float32, torch.float64):
        legacy = attention.attention_score_representability_bounds(2.0, 1.0, 197, dtype)
        numerical = attention.attention_score_representability_bounds(
            2.0, 1.0, 197, dtype, cap_by_theta=False,
        )
        expected = (-math.log(torch.finfo(dtype).tiny) - math.log(197) - 2.0) / 2.0
        assert legacy == PotentialBounds(-2.0, 2.0)
        assert math.isclose(numerical.max, expected)
        assert numerical.min == -numerical.max
        assert numerical.max > legacy.max
        assert numerical is attention.attention_score_representability_bounds(
            2.0, 1.0, 197, dtype, cap_by_theta=False,
        )


def verify_explicit_gaussian_replay() -> None:
    """Use the selected value interval in both event paths without changing sigma."""
    query, key, value, options = _inputs()
    clean = attention.spiking_scaled_dot_product_attention(query, key, value, **options)
    try:
        set_gaussian_time_noise(enabled=True, time_std=0.0, deadline_margin=0.0,
                                seed=0, device="cpu")
        zero = attention.spiking_scaled_dot_product_attention(query, key, value, **options)
        torch.testing.assert_close(zero, clean, rtol=1.0e-10, atol=1.0e-10)
        for seed in (0, 1, 2):
            results, statistics, states = [], [], []
            for _ in range(2):
                set_gaussian_time_noise(enabled=True, time_std=0.01, deadline_margin=0.04,
                                        seed=seed, device="cpu")
                results.append(attention.spiking_scaled_dot_product_attention(
                    query, key, value, **options,
                ))
                statistics.append(deepcopy(get_gaussian_noise_stats()))
                states.append(get_gaussian_time_noise().generator.get_state().clone())
            assert torch.equal(results[0], results[1])
            assert statistics[0] == statistics[1]
            assert torch.equal(states[0], states[1])
            assert bool(torch.isfinite(results[0]).all())
            assert bool((results[0].abs() <= 12.0).all())
            assert statistics[0]["attention.value"]["events"] == value.numel()
            assert statistics[0]["attention.value_reference"]["events"] == 1
    finally:
        set_gaussian_time_noise(enabled=False)


def verify_invalid_input_bounds() -> None:
    """Reject partial, asymmetric and degenerate bounds before stochastic sampling."""
    query, key, value, options = _inputs()
    invalid = [
        {**options, "query_bounds": None},
        {**options, "key_bounds": PotentialBounds(-7.0, 8.0)},
        {**options, "value_bounds": PotentialBounds(0.0, 0.0)},
        {**options, "query_bounds": PotentialBounds(-1.0e308, 1.0e308)},
    ]
    try:
        for bad in invalid:
            set_gaussian_time_noise(enabled=True, time_std=0.01, seed=0, device="cpu")
            state = get_gaussian_time_noise().generator.get_state().clone()
            try:
                attention.spiking_scaled_dot_product_attention(query, key, value, **bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("invalid attention bounds were accepted")
            assert torch.equal(state, get_gaussian_time_noise().generator.get_state())
            assert get_gaussian_noise_stats() == {}
    finally:
        set_gaussian_time_noise(enabled=False)


def verify_vit_projection_collection_and_transfer() -> None:
    """Collect real projections, freeze their separate ranges and preserve legacy dispatch."""
    from utils.transformers.models.spiking_vit import modeling_spiking_vit as modeling
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig

    config = ViTConfig(hidden_size=2, num_attention_heads=1, image_size=4,
                       patch_size=2, theta=2.0, tau_s=1.0)
    config._attn_implementation = "spiking_sdpa"
    module = modeling.ViTSelfAttention(config).double().eval()
    with torch.no_grad():
        for projection, gain in ((module.query, 6.0), (module.key, 8.0),
                                 (module.value, 12.0)):
            projection.weight.copy_(torch.eye(2, dtype=torch.float64) * gain)
            projection.bias.zero_()
    metadata = CalibrationMetadata(
        model_family="vit", model_id="tiny-attention", dataset_id="test",
        dataset_split="train", preprocessing="fixed", dtype="float64",
        theta=2.0, tau_s=1.0, tau_m=1.0, clip_margin=1.0e-5,
        max_sequence_length=None, input_shape=(3, 4, 4),
        model_options=(("vit_calibration_policy_version", 2),),
    )
    numeric = attention.attention_score_representability_bounds(
        2.0, 1.0, 5, torch.float64, cap_by_theta=False,
    )
    specs = [
        LayerCalibrationSpec("", name, CalibrationRangePolicy.SIGNED_SYMMETRIC,
                             0.0, 1.0, 0.05)
        for name in ("query", "key", "value")
    ]
    specs.append(LayerCalibrationSpec(
        "", "attention_score", CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING,
        0.0, 1.0, 0.05, numeric.min, numeric.max,
    ))
    collector = create_calibration_collector(metadata, tuple(specs), bin_count=32)
    hidden = Potential(torch.tensor([[[0.5, 0.25], [-0.25, 0.5]]], dtype=torch.float64),
                       PotentialBounds(-1.0, 1.0))
    assert not vit_calibration_uses_explicit_bounds(module)
    with patch.object(modeling, "ALL_ATTENTION_FUNCTIONS", {
        "spiking_sdpa": attention.spiking_sdpa_attention_forward,
    }):
        bind_model_calibration(module, collector)
        try:
            assert vit_calibration_uses_explicit_bounds(module)
            first, _ = module(hidden)
            assert first.domain == PotentialBounds(-12.0, 12.0)
            start_histogram_calibration_pass(collector)
            second, _ = module(hidden)
            assert torch.equal(first.value, second.value)
            table = finalize_calibration_collection(collector)
        finally:
            clear_model_calibration(module, expected_state=collector)
        records = {record.tensor_name: record for record in table.layers}
        assert set(records) == {"query", "key", "value", "attention_score"}
        for name, radius in (("query", 3.3), ("key", 4.4), ("value", 6.6)):
            assert math.isclose(records[name].bounds.max, radius)
        runtime = create_calibration_runtime(CalibrationMode.VALIDATE, table,
                                             expected_metadata=metadata)
        bind_model_calibration(module, runtime)
        try:
            frozen, _ = module(hidden)
            assert math.isclose(frozen.domain.max, 6.6)
            torch.testing.assert_close(frozen.value, first.value, rtol=1.0e-10, atol=1.0e-10)
        finally:
            clear_model_calibration(module, expected_state=runtime)
        legacy, _ = module(hidden)
        assert legacy.domain == PotentialBounds(-2.0, 2.0)
        assert not vit_calibration_uses_explicit_bounds(module)


def main() -> None:
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    try:
        for verify in (
            verify_explicit_attention_bounds, verify_score_numeric_ceiling,
            verify_explicit_gaussian_replay, verify_invalid_input_bounds,
            verify_vit_projection_collection_and_transfer,
        ):
            verify()
            print(f"PASS {verify.__name__}")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
