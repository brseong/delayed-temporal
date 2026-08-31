#!/usr/bin/env python3
"""Pure-Python regression checks for the toy BrainScaleS-2 HIL path."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import ast
import csv
import json
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluation.brainscales2_toy_hil import (
    _aggregate_isolated_conditions,
    _apply_condition_worker_config,
    _load_isolated_condition,
    _run_hagen_output,
    _run_isolated_pool_chunks,
    _run_temporal_pool,
    _run_worker_command_with_retries,
)
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.hagen import HagenConfig, HagenPWMBackend, HagenResult
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
    _configure_grouped_synapse_weights,
    _grouped_input_channel_slice,
    MockToyPoolBackend,
    ReplayToyPoolBackend,
    TimingCalibration,
    ToyPoolConfig,
    ToyPoolResult,
    concatenate_toy_pool_results,
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


def verify_grouped_broadcast_fan_in() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Grouped broadcast fan-in]]
    dedicated = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        mapping="dedicated",
        inference_trials=1,
        calibration_trials=2,
    )
    assert _grouped_input_channel_slice(0, dedicated, 3) == slice(0, 3)
    assert _grouped_input_channel_slice(1, dedicated, 3) == slice(3, 6)
    weight = torch.full((4, 6), -1.0)
    _configure_grouped_synapse_weights(
        weight,
        dedicated,
        input_fan_in=3,
        synaptic_weight=7.0,
    )
    expected = torch.tensor(
        [
            [7, 7, 7, 0, 0, 0],
            [7, 7, 7, 0, 0, 0],
            [0, 0, 0, 7, 7, 7],
            [0, 0, 0, 7, 7, 7],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(weight, expected)

    reused = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        mapping="time-multiplexed",
        inference_trials=1,
        calibration_trials=2,
    )
    reused_weight = torch.zeros((2, 3))
    _configure_grouped_synapse_weights(
        reused_weight, reused, input_fan_in=3, synaptic_weight=5.0
    )
    torch.testing.assert_close(reused_weight, torch.full((2, 3), 5.0))


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


def verify_chunked_pool_aggregation() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Chunked pool aggregation]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(2,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=29,
        miss_probability=0.0,
    )
    backend = MockToyPoolBackend()
    first = backend.run_uint5(
        torch.tensor([[0, 31]], dtype=torch.int32), config, spiking
    )
    second = backend.run_uint5(
        torch.tensor([[7, 19]], dtype=torch.int32), config, spiking
    )
    joined = concatenate_toy_pool_results([first, second])
    assert joined.first_spike_s.shape == (2, 2, 2, 2)
    assert joined.decoded_uint5.shape == (2, 2, 2)
    torch.testing.assert_close(
        joined.nominal_input_s,
        torch.cat((first.nominal_input_s, second.nominal_input_s), dim=0),
    )
    assert joined.metadata["chunked"] is True
    assert joined.metadata["sample_chunk_count"] == 2
    assert joined.metadata["calibration_strategy"] == "per-sample-chunk"


def verify_pool_size_aware_hardware_chunk_cap() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Pool-size-aware hardware chunk cap]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(8,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=8,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=31,
        miss_probability=0.0,
    )

    class RecordingBackend:
        def __init__(self) -> None:
            self.sizes: list[int] = []
            self.delegate = MockToyPoolBackend()

        def run_uint5(
            self,
            hidden_uint5: torch.Tensor,
            pool_config: ToyPoolConfig,
            spiking_config: BrainScaleS2PoolConfig,
        ) -> object:
            self.sizes.append(hidden_uint5.shape[0])
            return self.delegate.run_uint5(hidden_uint5, pool_config, spiking_config)

    backend = RecordingBackend()
    hidden = torch.arange(65 * 2, dtype=torch.int32).reshape(65, 2).remainder(32)
    result = _run_temporal_pool(
        SimpleNamespace(
            pool_backend="hardware",
            pool_sample_chunk_size=64,
            pool_replica_sample_budget=256,
        ),
        backend,
        hidden,
        config,
        spiking,
    )
    assert backend.sizes == [32, 32, 1]
    assert result.decoded_uint5.shape[1] == 65
    assert result.metadata["requested_pool_sample_chunk_size"] == 64
    assert result.metadata["effective_pool_sample_chunk_size"] == 32
    assert result.metadata["pool_replica_sample_budget"] == 256


def verify_pool_chunk_process_isolation() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Pool chunk process isolation]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(8,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=8,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=37,
        miss_probability=0.0,
    )
    hidden = torch.arange(35 * 2, dtype=torch.int32).reshape(35, 2).remainder(32)
    launched: list[tuple[str, int]] = []
    backend = MockToyPoolBackend()

    def launch(
        args: SimpleNamespace,
        chunk_dir: Path,
        hidden_chunk: torch.Tensor,
        pool_config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> ToyPoolResult:
        chunk_dir.mkdir(parents=True)
        launched.append((chunk_dir.name, hidden_chunk.shape[0]))
        (chunk_dir / "worker_status.json").write_text(
            json.dumps({"status": "passed", "attempts": [{"attempt": 1}]}),
            encoding="utf-8",
        )
        return backend.run_uint5(hidden_chunk, pool_config, spiking_config)

    with TemporaryDirectory() as directory:
        result = _run_isolated_pool_chunks(
            SimpleNamespace(output_dir=Path(directory)),
            hidden,
            config,
            spiking,
            16,
            launcher=launch,
        )
    assert launched == [
        ("chunk_000000_000016", 16),
        ("chunk_000016_000032", 16),
        ("chunk_000032_000035", 3),
    ]
    assert result.decoded_uint5.shape[1] == 35
    assert result.metadata["chunk_process_isolation"] is True
    assert len(result.metadata["chunk_worker_dirs"]) == 3
    assert len(result.metadata["chunk_worker_status"]) == 3


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


class _FakeChunkedHagen:
    def __init__(self) -> None:
        self.config = SimpleNamespace(mode="hardware")
        self.row_counts: list[int] = []

    def output_layer(self, converted: object, hidden: torch.Tensor) -> HagenResult:
        del converted
        self.row_counts.append(hidden.shape[0])
        value = hidden.sum(dim=1, keepdim=True).to(torch.int8)
        return HagenResult(
            value=value,
            metadata={
                "backend": "fake-hardware",
                "input_shape": list(hidden.shape),
                "output_shape": list(value.shape),
                "elapsed_s": 0.25,
            },
        )


def verify_hagen_output_row_chunking() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Hagen output row chunking]]
    hagen = _FakeChunkedHagen()
    hidden = torch.arange(14, dtype=torch.int32).reshape(7, 2)
    result = _run_hagen_output(
        SimpleNamespace(hagen_row_chunk_size=3),
        hagen,
        None,
        hidden,
    )
    assert hagen.row_counts == [3, 3, 1]
    torch.testing.assert_close(
        result.value, hidden.sum(dim=1, keepdim=True).to(torch.int8)
    )
    assert result.metadata["chunked"] is True
    assert result.metadata["row_chunk_count"] == 3
    assert result.metadata["input_shape"] == [7, 2]
    assert result.metadata["output_shape"] == [7, 1]
    assert result.metadata["elapsed_s"] == 0.75


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


def verify_host_mediated_implicit_relu_boundary() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Host-mediated implicit ReLU boundary]]
    hidden, metadata = HagenPWMBackend._implicit_lower_bound_uint5(  # noqa: SLF001
        torch.tensor([[-128.0, -3.0, 0.0, 3.0, 62.0, 127.0]]),
        shift=1,
    )
    torch.testing.assert_close(
        hidden,
        torch.tensor([[0, 0, 0, 2, 31, 31]], dtype=torch.int32),
    )
    assert metadata["relu_boundary"] == "implicit-lower-bound-host"
    assert metadata["converting_relu"] is None
    assert metadata["host_mediated_lower_bound"] is True
    assert metadata["lower_bound_v"] == 0.0
    assert metadata["upper_bound_v"] == 31.0
    assert metadata["lower_bound_clamped_values"] == 2
    assert metadata["upper_bound_clamped_values"] == 1


def verify_condition_process_isolation_contract() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Condition process isolation]]
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config_path = output / "worker_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(output / "checkpoint.pt"),
                    "output_dir": str(output / "worker"),
                    "condition_worker": True,
                }
            ),
            encoding="utf-8",
        )
        loaded_args = _apply_condition_worker_config(
            SimpleNamespace(
                condition_worker_config=config_path,
                checkpoint=None,
                output_dir=output,
                condition_worker=False,
            )
        )
        assert loaded_args.checkpoint == output / "checkpoint.pt"
        assert loaded_args.output_dir == output / "worker"
        assert loaded_args.condition_worker is True

        labels = torch.tensor([0, 1])
        baseline_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        hidden = torch.tensor([[0, 31], [15, 8]], dtype=torch.int32)
        spiking = BrainScaleS2PoolConfig(
            trials=4,
            pool_sizes=(1, 2),
            placements=("same-quadrant",),
            routings=("broadcast",),
        )
        worker_dirs: list[Path] = []
        for pool_size in (1, 2):
            result = MockToyPoolBackend().run_uint5(
                hidden,
                ToyPoolConfig(
                    pool_size=pool_size,
                    logical_neurons=2,
                    inference_trials=2,
                    calibration_trials=4,
                    seed=40 + pool_size,
                    miss_probability=0.0,
                ),
                spiking,
            )
            key = f"ttfs_M{pool_size}_local-pool_dedicated"
            evaluation = ToyConditionEvaluation(
                key=key,
                pool_size=pool_size,
                pooling_domain="ttfs",
                pool_result=result,
                logits=baseline_logits.reshape(1, 2, 2).repeat(2, 1, 1),
                pwm_metadata={"first": {"shared": True}, "output": {}},
            )
            worker_dir = output / "condition_workers" / key
            write_toy_artifacts(
                worker_dir,
                labels=labels,
                float_logits=baseline_logits,
                ideal_logits=baseline_logits,
                ideal_hidden_uint5=hidden,
                evaluations=[evaluation],
                manifest={
                    "pool_sizes": [pool_size],
                    "placements": ["local-pool"],
                    "conditions": [
                        {
                            "key": key,
                            "pool_size": pool_size,
                            "pooling_domain": "ttfs",
                            "placement": result.placement,
                            "mapping": result.mapping,
                            "physical_coordinates": result.physical_coordinates,
                            "pool_metadata": result.metadata,
                            "pwm_metadata": evaluation.pwm_metadata,
                        }
                    ],
                },
                runtime={"elapsed_s": float(pool_size)},
                bootstrap_iterations=10,
            )
            (worker_dir / "worker_status.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "attempts": [{"attempt": 1, "status": "passed"}],
                    }
                ),
                encoding="utf-8",
            )
            restored, _, _, _ = _load_isolated_condition(worker_dir)
            assert restored.key == key
            worker_dirs.append(worker_dir)

        first_hidden_dir = output / "condition_workers" / "first_hidden_avg1"
        first_hidden_dir.mkdir(parents=True)
        _aggregate_isolated_conditions(
            SimpleNamespace(
                output_dir=output,
                pool_sizes=[1, 2],
                placements=["local-pool"],
                bootstrap_iterations=10,
                seed=0,
            ),
            worker_dirs,
            {1: first_hidden_dir},
        )
        master = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        isolation = master["condition_process_isolation"]
        assert isolation["enabled"] is True
        assert isolation["worker_count"] == 2
        assert isolation["resumable"] is True
        assert set(isolation["worker_status"]) == {
            "condition_workers/ttfs_M1_local-pool_dedicated",
            "condition_workers/ttfs_M2_local-pool_dedicated",
        }
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            conditions = {row["condition"] for row in csv.DictReader(handle)}
        assert "ttfs_M1_local-pool_dedicated" in conditions
        assert "ttfs_M2_local-pool_dedicated" in conditions


def verify_transient_worker_retry() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Transient worker retry]]
    with TemporaryDirectory() as directory:
        worker_dir = Path(directory)
        calls: list[list[str]] = []
        delays: list[float] = []

        def flaky_runner(
            command: list[str], *, cwd: Path, check: bool
        ) -> subprocess.CompletedProcess[str]:
            assert cwd == REPOSITORY_ROOT
            assert check is True
            calls.append(command)
            if len(calls) < 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        status = _run_worker_command_with_retries(
            ["worker"],
            worker_dir,
            max_attempts=3,
            retry_backoff_s=2.0,
            idle_timeout_s=10.0,
            runner=flaky_runner,
            sleeper=delays.append,
        )
        assert len(calls) == 3
        assert delays == [2.0, 4.0]
        assert status["status"] == "passed"
        assert [attempt["status"] for attempt in status["attempts"]] == [
            "failed",
            "failed",
            "passed",
        ]
        saved = json.loads(
            (worker_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert saved == status

        exhausted_dir = worker_dir / "exhausted"
        exhausted_dir.mkdir()

        def failing_runner(
            command: list[str], *, cwd: Path, check: bool
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(7, command)

        try:
            _run_worker_command_with_retries(
                ["worker"],
                exhausted_dir,
                max_attempts=2,
                retry_backoff_s=0.0,
                idle_timeout_s=10.0,
                runner=failing_runner,
                sleeper=lambda _: None,
            )
        except subprocess.CalledProcessError as error:
            assert error.returncode == 7
        else:
            raise AssertionError("exhausted worker retry must re-raise")
        exhausted = json.loads(
            (exhausted_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert exhausted["status"] == "failed"
        assert len(exhausted["attempts"]) == 2

        silent_dir = worker_dir / "silent"
        silent_dir.mkdir()
        try:
            _run_worker_command_with_retries(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                silent_dir,
                max_attempts=1,
                retry_backoff_s=0.0,
                idle_timeout_s=0.25,
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("silent worker must hit its idle watchdog")
        silent = json.loads(
            (silent_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert silent["status"] == "failed"
        assert silent["attempts"][0]["failure"] == "idle-timeout"
        assert (silent_dir / "worker_attempt_1.log").is_file()


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
    assert "SPIKING_INPUT_FAN_IN = 4" in source
    assert "'--input-fan-in', SPIKING_INPUT_FAN_IN" in source
    assert "RELU_BOUNDARY = 'implicit-lower-bound-host'" in source
    assert "'--relu-boundary', RELU_BOUNDARY" in source
    assert "POOL_SAMPLE_CHUNK_SIZE = 64" in source
    assert "'--pool-sample-chunk-size', POOL_SAMPLE_CHUNK_SIZE" in source
    assert "POOL_REPLICA_SAMPLE_BUDGET = 128" in source
    assert "'--pool-replica-sample-budget', POOL_REPLICA_SAMPLE_BUDGET" in source
    assert "HAGEN_ROW_CHUNK_SIZE = 512" in source
    assert "CONDITION_WORKER_MAX_ATTEMPTS = 3" in source
    assert "CONDITION_WORKER_RETRY_BACKOFF_S = 20.0" in source
    assert "CONDITION_WORKER_IDLE_TIMEOUT_S = 180.0" in source
    assert "ARTIFACT_ROOT = None" in source
    assert "if ARTIFACT_ROOT is not None" in source
    assert "'--hagen-row-chunk-size', HAGEN_ROW_CHUNK_SIZE" in source
    assert "'--condition-worker-max-attempts', CONDITION_WORKER_MAX_ATTEMPTS" in source
    assert (
        "'--condition-worker-retry-backoff-s', "
        "CONDITION_WORKER_RETRY_BACKOFF_S"
    ) in source
    assert (
        "'--condition-worker-idle-timeout-s', "
        "CONDITION_WORKER_IDLE_TIMEOUT_S"
    ) in source
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
    verify_grouped_broadcast_fan_in()
    verify_mock_and_all_miss_policy()
    verify_chunked_pool_aggregation()
    verify_pool_size_aware_hardware_chunk_cap()
    verify_pool_chunk_process_isolation()
    verify_replay_split_and_reproducibility()
    verify_hagen_output_row_chunking()
    verify_hagen_host_tiling()
    verify_host_mediated_implicit_relu_boundary()
    verify_condition_process_isolation_contract()
    verify_transient_worker_retry()
    verify_metrics_and_artifact_schema()
    verify_python311_and_notebook_contract()
    print("BrainScaleS-2 toy ANN2SNN verification passed")


if __name__ == "__main__":
    main()
