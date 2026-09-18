#!/usr/bin/env python3
"""Dataset-independent checks for the clock-driven TTFS execution contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluation.error_analysis_vit import (
    evaluation_shard_bounds,
    validate_vit_runtime_arguments,
)
from scripts.experiments.run_clock_driven_vit import (
    calibration_compatibility_paths,
    initially_idle_gpus,
    parse_result,
    prepare_evaluation_subset,
    write_summary,
)
from utils.transforms.clock import (
    clock_step_indices,
    get_clock_driven_stats,
    get_clock_update_stats,
    set_clock_driven,
)
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.primitive import signed_pulse_width_duration
from utils.transforms.spike_to_potential import (
    exp_operator,
    normalized_exp_operator,
)
from utils.transforms.types import PotentialBounds, TimeBounds
from utils.transforms.types import Potential
from utils.transformers.integrations import spiking_sdpa_attention as attention
from utils.transformers.models.spiking_ops import SpikingLinear


def _explicit_signed_pwm(
    event_a_step: int,
    event_b_step: int,
    deadline_step: int,
    *,
    time_step: float,
) -> float:
    accumulator_a = 0.0
    accumulator_b = 0.0
    for step in range(deadline_step):
        if step >= event_a_step:
            accumulator_a += time_step
        if step >= event_b_step:
            accumulator_b += time_step
    return accumulator_a - accumulator_b


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Causal Encoder Clocking]]
def verify_causal_encoder_clocking() -> None:
    set_clock_driven(enabled=True, time_step=0.25)
    values = torch.tensor([1.0, 0.6, 0.0, -0.6, -1.0], dtype=torch.float64)
    times, domain = neg_identity_transform(values, PotentialBounds(-1.0, 1.0))
    torch.testing.assert_close(
        times,
        torch.tensor([0.0, 0.5, 1.0, 1.75, 2.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert domain == TimeBounds(0.0, 2.0)
    stats = get_clock_driven_stats()["neg_linear_transform"]
    assert stats["events"] == values.numel()
    assert stats["maximum_window_steps"] == 8
    updates = get_clock_update_stats()["encoder"]
    assert updates == {"calls": 1, "time_steps": 9, "element_updates": 45}
    set_clock_driven(enabled=False)


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#PWM State Updates]]
def verify_pwm_state_updates() -> None:
    time_step = 0.25
    set_clock_driven(enabled=True, time_step=time_step)
    event_a = torch.tensor([0.5, 1.75], dtype=torch.float64)
    event_b = torch.tensor(1.0, dtype=torch.float64)
    result = signed_pulse_width_duration(
        event_a,
        event_b,
        observation_deadline=2.0,
    )
    expected = torch.tensor(
        [
            _explicit_signed_pwm(2, 4, 8, time_step=time_step),
            _explicit_signed_pwm(7, 4, 8, time_step=time_step),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected, atol=0.0, rtol=0.0)
    updates = get_clock_update_stats()["pwm"]
    assert updates == {"calls": 1, "time_steps": 8, "element_updates": 32}

    set_clock_driven(enabled=True, time_step=0.1)
    accumulated = torch.zeros((), dtype=torch.float64)
    for _ in range(400):
        accumulated = accumulated + accumulated.new_tensor(0.1)
    assert clock_step_indices(accumulated).item() == 400
    try:
        clock_step_indices(torch.tensor(40.05, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("quarter-step misalignment was accepted")
    set_clock_driven(enabled=False)


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Exponential State Updates]]
def verify_exponential_state_updates() -> None:
    time_step = 0.125
    tau = 1.0
    set_clock_driven(enabled=True, time_step=time_step)
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    with patch.object(torch, "pow", side_effect=AssertionError("closed-form power")):
        decay, _ = exp_operator(times, TimeBounds(0.0, 1.0), tau_m=tau)
    beta = math.exp(-time_step / tau)
    expected_decay = []
    for elapsed_steps in (8, 4, 0):
        value = 1.0
        for _ in range(elapsed_steps):
            value *= beta
        expected_decay.append(value)
    torch.testing.assert_close(
        decay,
        torch.tensor(expected_decay, dtype=torch.float64),
        atol=1.0e-15,
        rtol=1.0e-15,
    )

    signed_times = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
    with patch.object(torch, "pow", side_effect=AssertionError("closed-form power")):
        growth, _ = normalized_exp_operator(
            signed_times,
            TimeBounds(-0.5, 0.5),
            tau_m=tau,
        )
    alpha = math.exp(time_step / tau)
    expected_growth = torch.tensor(
        [alpha**-4, 1.0, alpha**4], dtype=torch.float64
    )
    torch.testing.assert_close(growth, expected_growth, atol=1.0e-15, rtol=1.0e-15)
    updates = get_clock_update_stats()["exponential"]
    assert updates["calls"] == 4
    assert updates["time_steps"] == 24
    assert updates["element_updates"] == 60
    set_clock_driven(enabled=False)


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Optimized Model Kernels]]
def verify_optimized_model_kernels() -> None:
    set_clock_driven(enabled=True, time_step=0.25)
    linear = SpikingLinear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float64))
    linear_output = linear(
        Potential(
            torch.tensor([[0.6, -0.6]], dtype=torch.float64),
            PotentialBounds(-1.0, 1.0),
        )
    )
    torch.testing.assert_close(
        linear_output.value,
        torch.tensor([[1.75]], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )

    query = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    key = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    value = torch.tensor([[[[0.6], [-0.6]]]], dtype=torch.float64)
    fixed_weights = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float64)
    with patch.object(
        attention,
        "softmin_function",
        return_value=(fixed_weights, PotentialBounds(0.0, 1.0)),
    ):
        attention_output = attention.spiking_scaled_dot_product_attention(
            query,
            key,
            value,
            theta=1.0,
            source_length_max=2,
        )
    torch.testing.assert_close(
        attention_output,
        torch.tensor([[[[-0.125]]]], dtype=torch.float64),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    updates = get_clock_update_stats()
    assert updates["encoder"]["calls"] > 0
    assert updates["pwm"]["calls"] > 0
    assert updates["pwm"]["time_steps"] > 0
    set_clock_driven(enabled=False)


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Disabled-Mode Parity]]
def verify_disabled_mode_parity() -> None:
    set_clock_driven(enabled=False)
    values = torch.tensor([0.75, 0.0, -0.75], dtype=torch.float64)
    domain = PotentialBounds(-1.0, 1.0)
    times, time_domain = neg_identity_transform(values, domain)
    torch.testing.assert_close(
        times,
        torch.tensor([0.25, 1.0, 1.75], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    duration = signed_pulse_width_duration(
        times,
        torch.tensor(1.0, dtype=torch.float64),
        observation_deadline=2.0,
    )
    torch.testing.assert_close(duration, 1.0 - times, atol=0.0, rtol=0.0)
    decoded, _ = exp_operator(times, time_domain, tau_m=1.0)
    torch.testing.assert_close(decoded, torch.exp(times - 2.0), atol=0.0, rtol=0.0)
    assert get_clock_update_stats() == {}


def _runtime_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "max_eval_batches": 0,
        "benchmark_warmup_batches": 0,
        "benchmark_measure_batches": 0,
        "evaluation_dataset_path": "",
        "evaluation_shard_count": 1,
        "evaluation_shard_index": 0,
        "image_preprocessing_config": "",
        "clock_driven": True,
        "clock_time_step": 0.25,
        "model_backend": "spiking",
        "gaussian_time_noise": False,
        "time_noise_std_frac": 0.0,
        "time_noise_mean": 0.0,
        "time_noise_deadline_margin_std": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#ViT Runtime Isolation]]
def verify_vit_runtime_isolation() -> None:
    validate_vit_runtime_arguments(_runtime_args())
    invalid = (
        _runtime_args(clock_time_step=0.0),
        _runtime_args(model_backend="hf"),
        _runtime_args(gaussian_time_noise=True),
        _runtime_args(time_noise_std_frac=1.0e-5),
        _runtime_args(clock_driven=False, clock_time_step=0.25),
        _runtime_args(evaluation_shard_count=0),
        _runtime_args(evaluation_shard_count=4, evaluation_shard_index=4),
    )
    for args in invalid:
        try:
            validate_vit_runtime_arguments(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid clock-driven configuration accepted: {args}")

    snapshot_target = "scripts.experiments.run_clock_driven_vit.gpu_snapshot"
    sleep_target = "scripts.experiments.run_clock_driven_vit.time.sleep"
    snapshot = {gpu: (0, 0) for gpu in range(8)}
    with (
        patch(snapshot_target, return_value=snapshot),
        patch(sleep_target),
    ):
        assert initially_idle_gpus((0, 4, 7), allowed=tuple(range(8))) == (0, 4, 7)
    with (
        patch(snapshot_target, return_value=snapshot),
        patch(sleep_target),
    ):
        try:
            initially_idle_gpus((0, 4), allowed=(4, 5, 6, 7))
        except ValueError:
            pass
        else:
            raise AssertionError("default GPU policy accepted GPU 0")

    run_target = "scripts.experiments.run_clock_driven_vit.subprocess.run"
    output_target = (
        "scripts.experiments.run_clock_driven_vit.subprocess.check_output"
    )
    with (
        patch(run_target, return_value=SimpleNamespace(returncode=0)),
        patch(
            output_target,
            return_value="utils/transforms/calibration.py\nutils/transforms/clock.py\n",
        ),
    ):
        assert calibration_compatibility_paths(
            REPOSITORY_ROOT, "a" * 40, "b" * 40
        ) == ["utils/transforms/calibration.py", "utils/transforms/clock.py"]
    with (
        patch(run_target, return_value=SimpleNamespace(returncode=0)),
        patch(output_target, return_value="utils/transforms/functions.py\n"),
    ):
        try:
            calibration_compatibility_paths(
                REPOSITORY_ROOT, "a" * 40, "b" * 40
            )
        except ValueError:
            pass
        else:
            raise AssertionError("calibration-relevant source change was accepted")


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Contiguous Evaluation Shards]]
def verify_contiguous_evaluation_shards() -> None:
    ranges = [evaluation_shard_bounds(500, 4, index) for index in range(4)]
    assert ranges == [(0, 125), (125, 250), (250, 375), (375, 500)]
    uneven = [evaluation_shard_bounds(10, 3, index) for index in range(3)]
    assert uneven == [(0, 4), (4, 7), (7, 10)]

    rows = []
    for index, (start, stop) in enumerate(ranges):
        samples = stop - start
        rows.append({
            "run_id": f"continuous_shard_{index:02d}",
            "clock_driven": False,
            "time_step": None,
            "shard_index": index,
            "shard_count": 4,
            "shard_start": start,
            "shard_stop": stop,
            "correct": 100,
            "samples": samples,
            "accuracy": 100 / samples,
            "prediction_sha256": f"{index:064x}",
            "gpu": index + 4,
            "gpu_model": "NVIDIA RTX A6000",
            "elapsed_seconds": 1.0,
            "source_commit": "a" * 40,
            "calibration_sha256": "b" * 64,
            "log_sha256": f"{index + 4:064x}",
        })
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_summary(
            root,
            rows,
            tag="test_population_500",
            time_steps=(),
            shard_count=4,
            expected_population=500,
        )
        summary = json.loads((root / "summary.json").read_text())
    assert summary["tag"] == "test_population_500"
    assert summary["conditions"][0]["samples"] == 500
    assert summary["conditions"][0]["correct"] == 400

    from datasets import Dataset

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source"
        subset_path = root / "subset"
        metadata_path = root / "subset.json"
        Dataset.from_dict({"value": list(range(10))}).save_to_disk(str(source_path))
        metadata = prepare_evaluation_subset(
            source_path, subset_path, metadata_path, sample_count=5
        )
        replay = prepare_evaluation_subset(
            source_path, subset_path, metadata_path, sample_count=5
        )
        assert metadata == replay
        assert metadata["sample_count"] == 5
        assert Dataset.load_from_disk(str(subset_path))["value"] == list(range(5))


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Verification#Composed Encoder Statistics]]
def verify_composed_encoder_statistics() -> None:
    log = """GPU model: NVIDIA RTX A6000
Evaluation metadata — model: checkpoint, dataset: imagenet-1k, split: validation, samples: 62, theta: 20.0, precision: float64, source: disk:/tmp/validation_first_500, fingerprint: abcdef
Evaluation shard — index: 4, count: 8, start: 252, stop: 314, population: 500
Correct: 52
Evaluated samples: 62
Prediction SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Accuracy: 0.83870968
ClockUpdates[encoder] calls=1, time_steps=1, element_updates=1
ClockUpdates[exponential] calls=1, time_steps=1, element_updates=1
ClockUpdates[pwm] calls=1, time_steps=1, element_updates=1
Clock[neg_linear_transform] events=1, rounded_events=1, mean_absolute_error=0.1, maximum_absolute_error=0.1, windows=1, minimum_window_steps=1, maximum_window_steps=1
Clock[neg_log_transform] events=1, rounded_events=1, mean_absolute_error=0.1, maximum_absolute_error=0.1, windows=1, minimum_window_steps=1, maximum_window_steps=1
Clock[gelu.cubic.log_positive] events=1, rounded_events=1, mean_absolute_error=0.1, maximum_absolute_error=0.1, windows=1, minimum_window_steps=1, maximum_window_steps=1
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clock.log"
        path.write_text(log)
        result = parse_result(
            path,
            run_id="dt_1_shard_04",
            time_step=1.0,
            shard_index=4,
            shard_count=8,
            expected_population=500,
            evaluation_dataset_path=Path("/tmp/validation_first_500"),
            gpu=4,
            commit="a" * 40,
            calibration_sha256="b" * 64,
            elapsed_seconds=1.0,
        )
    assert set(result["clock_sites"]) == {
        "neg_linear_transform",
        "neg_log_transform",
        "gelu.cubic.log_positive",
    }
    assert result["accuracy"] == 52 / 62


if __name__ == "__main__":
    checks = (
        verify_causal_encoder_clocking,
        verify_pwm_state_updates,
        verify_exponential_state_updates,
        verify_optimized_model_kernels,
        verify_disabled_mode_parity,
        verify_vit_runtime_isolation,
        verify_contiguous_evaluation_shards,
        verify_composed_encoder_statistics,
    )
    try:
        for check in checks:
            check()
            print(f"PASS {check.__name__}")
    finally:
        set_clock_driven(enabled=False)
