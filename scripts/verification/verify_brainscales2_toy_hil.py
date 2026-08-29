#!/usr/bin/env python3
"""Pure-Python regression checks for the toy BrainScaleS-2 HIL path."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import ast
import csv
import json
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.hagen import HagenConfig, HagenPWMBackend
from utils.hardware.brainscales2.toy import (
    ARCHITECTURES,
    ToyMLP,
    convert_float_model,
    deserialize_converted_model,
    make_yin_yang_split,
    parameter_sha256,
    serialize_converted_model,
)
from utils.hardware.brainscales2.toy_artifacts import (
    ToyConditionEvaluation,
    summarize_toy_evaluations,
    write_toy_artifacts,
)
from utils.hardware.brainscales2.toy_pooling import (
    MockToyPoolBackend,
    ReplayToyPoolBackend,
    TimingCalibration,
    ToyPoolConfig,
    decode_pool_observations,
    resolve_grouped_physical_coordinates,
)


def verify_deterministic_yin_yang_splits() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Deterministic datasets and frozen conversion]]
    first_x, first_y = make_yin_yang_split(120, 42)
    second_x, second_y = make_yin_yang_split(120, 42)
    other_x, _ = make_yin_yang_split(120, 41)
    torch.testing.assert_close(first_x, second_x)
    torch.testing.assert_close(first_y, second_y)
    assert first_x.shape == (120, 4)
    assert first_y.bincount(minlength=3).tolist() == [40, 40, 40]
    assert not torch.equal(first_x, other_x)
    assert bool(((first_x >= 0.0) & (first_x <= 1.0)).all())


def _converted_fixture() -> tuple[ToyMLP, object, torch.Tensor]:
    torch.manual_seed(7)
    model = ToyMLP(ARCHITECTURES["yy-30"])
    calibration_x, _ = make_yin_yang_split(96, 41)
    converted = convert_float_model(model, calibration_x)
    return model, converted, calibration_x


def verify_frozen_integer_conversion() -> None:
    model, converted, calibration_x = _converted_fixture()
    before = parameter_sha256(model)
    forward = converted.forward(calibration_x[:8])
    after = parameter_sha256(model)
    assert before == after == converted.manifest.source_parameter_sha256
    assert int(forward.input_uint5.min()) >= 0 and int(forward.input_uint5.max()) <= 31
    assert int(forward.hidden_uint5.min()) >= 0 and int(forward.hidden_uint5.max()) <= 31
    assert int(converted.first.weight_with_bias.min()) >= -63
    assert int(converted.first.weight_with_bias.max()) <= 63
    assert int(forward.logits_int8.min()) >= -128
    assert int(forward.logits_int8.max()) <= 127
    assert converted.first.weight_with_bias.shape == (30, 5)
    assert converted.second.weight_with_bias.shape == (3, 31)
    restored = deserialize_converted_model(serialize_converted_model(converted))
    restored_forward = restored.forward(calibration_x[:8])
    torch.testing.assert_close(forward.hidden_uint5, restored_forward.hidden_uint5)
    torch.testing.assert_close(forward.logits_int8, restored_forward.logits_int8)


def verify_grouped_placement() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Physical pool allocation]]
    local = resolve_grouped_physical_coordinates(30, 16, "local-pool", "dedicated")
    cross = resolve_grouped_physical_coordinates(30, 16, "cross-quadrant", "dedicated")
    assert local.shape == (30, 16) and cross.shape == (30, 16)
    assert torch.unique(local).numel() == 480
    assert torch.unique(cross).numel() == 480
    assert int(local.min()) >= 0 and int(local.max()) < 512
    for logical in range(30):
        assert torch.unique(local[logical] // 128).numel() == 1
        assert torch.unique(cross[logical] // 128).numel() == 4
    reused = resolve_grouped_physical_coordinates(
        128, 16, "cross-quadrant", "time-multiplexed"
    )
    assert bool((reused == reused[0]).all())
    try:
        ToyPoolConfig(pool_size=16, logical_neurons=128, mapping="dedicated")
    except ValueError as error:
        assert "512" in str(error)
    else:
        raise AssertionError("oversized dedicated mapping was accepted")


def verify_mock_and_all_miss_policy() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Miss-aware temporal decoding]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(4,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    hidden = torch.tensor([[0, 15], [31, 8]], dtype=torch.int32)
    config = ToyPoolConfig(
        pool_size=4,
        logical_neurons=2,
        inference_trials=3,
        calibration_trials=4,
        seed=19,
        miss_probability=0.0,
    )
    first = MockToyPoolBackend().run_uint5(hidden, config, spiking)
    second = MockToyPoolBackend().run_uint5(hidden, config, spiking)
    torch.testing.assert_close(first.first_spike_s, second.first_spike_s)
    torch.testing.assert_close(first.decoded_uint5, second.decoded_uint5)
    assert float((first.decoded_uint5 - hidden).abs().float().mean()) < 2.0

    coordinates = resolve_grouped_physical_coordinates(2, 4, "local-pool", "dedicated")
    all_miss = torch.full((1, 2, 2, 4), torch.nan, dtype=torch.float64)
    nominal = torch.full((2, 2), 15.0e-6, dtype=torch.float64)
    decoded = decode_pool_observations(
        all_miss,
        nominal,
        TimingCalibration(5.0e-6, torch.zeros((2, 4), dtype=torch.float64), 4),
        coordinates,
        ToyPoolConfig(
            pool_size=4,
            logical_neurons=2,
            inference_trials=1,
            calibration_trials=4,
        ),
        spiking,
    )
    assert bool(decoded.all_miss.all())
    assert int(decoded.decoded_uint5.sum()) == 0


def verify_replay_split_and_reproducibility() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Held-out hardware replay]]
    path = (
        REPOSITORY_ROOT
        / "artifacts/brainscales2/20260829T084007Z/full_pooling/events.pt"
    )
    if not path.is_file():
        raise AssertionError("allowlisted accepted replay artifact is missing")
    backend = ReplayToyPoolBackend(path)
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(4,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=4,
        logical_neurons=3,
        inference_trials=2,
        calibration_trials=4,
        seed=23,
    )
    hidden = torch.tensor([[0, 15, 31], [7, 21, 4]], dtype=torch.int32)
    first = backend.run_uint5(hidden, config, spiking)
    second = backend.run_uint5(hidden, config, spiking)
    torch.testing.assert_close(first.first_spike_s, second.first_spike_s, equal_nan=True)
    assert first.metadata["held_out_trials"] == 128
    assert first.metadata["scope"] == "rough-model-only"


def verify_hagen_host_tiling() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Hagen tiling contract]]
    _, converted, _ = _converted_fixture()

    class FakePerceptron:
        @staticmethod
        def matmul(input_value: torch.Tensor, weight: torch.Tensor, **_: object) -> torch.Tensor:
            return input_value @ weight

    fake_hxtorch = SimpleNamespace(perceptron=FakePerceptron())
    backend = HagenPWMBackend(HagenConfig(mode="mock", tiling="host-128", tile_size=2))
    value = torch.tensor([[1, 2, 3, 4, 31]], dtype=torch.float32)
    result, schedule = backend._host_tiled_linear(  # noqa: SLF001 - contract probe
        fake_hxtorch,
        value,
        converted.first,
        avg=1,
    )
    expected = value.to(torch.int32) @ converted.first.weight_with_bias.T.to(torch.int32)
    torch.testing.assert_close(result.to(torch.int32), expected.clamp(-128, 127))
    assert [
        {"start": row["start"], "stop": row["stop"]} for row in schedule
    ] == [
        {"start": 0, "stop": 2},
        {"start": 2, "stop": 4},
        {"start": 4, "stop": 5},
    ]
    assert all(0.0 <= row["saturation_rate"] <= 1.0 for row in schedule)


def verify_metrics_and_artifact_schema() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Network artifact contract]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(2,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    hidden = torch.tensor([[0, 15], [31, 8]], dtype=torch.int32)
    result = MockToyPoolBackend().run_uint5(
        hidden,
        ToyPoolConfig(
            pool_size=2,
            logical_neurons=2,
            inference_trials=2,
            calibration_trials=4,
            seed=3,
            miss_probability=0.0,
        ),
        spiking,
    )
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]]
    )
    evaluation = ToyConditionEvaluation(
        key="ttfs_M2_local-pool_dedicated",
        pool_size=2,
        pooling_domain="ttfs",
        pool_result=result,
        logits=logits,
        pwm_metadata={"backend": "torch"},
    )
    labels = torch.tensor([0, 1])
    rows = summarize_toy_evaluations(labels, logits[0], logits[0], [evaluation], bootstrap_iterations=10)
    assert rows[-1]["accuracy"] == 1.0
    with TemporaryDirectory() as directory:
        output = Path(directory)
        write_toy_artifacts(
            output,
            labels=labels,
            float_logits=logits[0],
            ideal_logits=logits[0],
            ideal_hidden_uint5=hidden,
            evaluations=[evaluation],
            manifest={"task": "fixture"},
            runtime={"elapsed_s": 0.1},
            bootstrap_iterations=10,
        )
        expected = {
            "manifest.json",
            "runtime.json",
            "metrics.csv",
            "predictions.csv",
            "events.csv",
            "intermediates.pt",
        }
        assert expected.issubset({path.name for path in output.iterdir()})
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["event_csv_coverage"]["full_raw_tensor"] == "intermediates.pt"
        with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 2 * 2 * 2 * 2


def verify_python311_and_notebook_contract() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#EBRAINS launcher contract]]
    source_paths = [
        REPOSITORY_ROOT / "utils/hardware/brainscales2/toy.py",
        REPOSITORY_ROOT / "utils/hardware/brainscales2/toy_pooling.py",
        REPOSITORY_ROOT / "utils/hardware/brainscales2/hagen.py",
        REPOSITORY_ROOT / "scripts/evaluation/brainscales2_toy_hil.py",
    ]
    for path in source_paths:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 11))
    notebook = REPOSITORY_ROOT / "scripts/notebooks/ebrains_brainscales2_toy_hil.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    version = payload["metadata"]["language_info"]["version"]
    assert version == "3.11" or version.startswith("3.11.")
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "RUN_HARDWARE_SMOKE = False" in source
    assert "RUN_YINYANG_FULL = False" in source
    assert "brainscales2_toy_hil.py" in source
    assert "--max-test-samples" in source
    assert "save_nightly_calibration" in source
    assert "hagen_cocolist.pbin" in source
    assert "spiking_cocolist.pbin" in source
    assert "HAGEN_CALIBRATION_PATH is None" in source
    assert "SPIKING_CALIBRATION_PATH is None" in source
    assert "pipeline_status.json" in source
    assert "Using probe-selected HAGEN_HIDDEN_SHIFT" in source
    assert "formal stages require a passing same-run smoke gate" in source
    assert "SMOKE_MAX_MULTI_SPIKE_RATE" in source
    assert source.index("'--phase', 'train'") < source.index(
        "setup_hardware_client()"
    )
    assert source.index("if RUN_HARDWARE_SMOKE:") < source.index(
        "if RUN_YINYANG_FULL:"
    )


def main() -> None:
    verify_deterministic_yin_yang_splits()
    verify_frozen_integer_conversion()
    verify_grouped_placement()
    verify_mock_and_all_miss_policy()
    verify_replay_split_and_reproducibility()
    verify_hagen_host_tiling()
    verify_metrics_and_artifact_schema()
    verify_python311_and_notebook_contract()
    print("BrainScaleS-2 toy ANN2SNN verification passed")


if __name__ == "__main__":
    main()
