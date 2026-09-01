#!/usr/bin/env python3
"""Train, convert, and evaluate toy ANN2SNN classifiers with BSS-2 HIL stages."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import monotonic, perf_counter, sleep
from typing import Any, Callable
import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_REPLICA_SAMPLE_BUDGET = 128
DEFAULT_POOL_CALIBRATION_TRIAL_CHUNK_SIZE = 4
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.hagen import (
    HagenConfig,
    HagenPWMBackend,
    HagenResult,
    file_sha256,
)
from utils.hardware.brainscales2.toy import (
    ARCHITECTURES,
    ConvertedToyModel,
    ToyMLP,
    TrainingConfig,
    classification_metrics,
    convert_float_model,
    deserialize_converted_model,
    load_dataset_bundle,
    parameter_sha256,
    serialize_converted_model,
    train_float_model,
)
from utils.hardware.brainscales2.toy_artifacts import (
    ToyConditionEvaluation,
    write_toy_artifacts,
)
from utils.hardware.brainscales2.toy_pooling import (
    GroupedHardwarePoolBackend,
    MockToyPoolBackend,
    ReplayToyPoolBackend,
    TimingCalibration,
    TimingCalibrationObservation,
    ToyPoolConfig,
    ToyPoolResult,
    calibrate_timing,
    concatenate_timing_calibration_observations,
    concatenate_toy_pool_results,
)


DEFAULT_REPLAY = (
    REPOSITORY_ROOT
    / "artifacts/brainscales2/20260829T084007Z/full_pooling/events.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toy ANN2SNN accuracy robustness on BrainScaleS-2",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "train",
            "convert",
            "local-eval",
            "probe-hagen",
            "hardware-smoke",
            "hardware-eval",
        ),
        default="local-eval",
    )
    parser.add_argument("--task", choices=("yinyang", "mnist"), default="yinyang")
    parser.add_argument(
        "--architecture",
        choices=tuple(ARCHITECTURES),
        default="yy-30",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "sigmoid"),
        default="relu",
        help=(
            "hidden activation trained into the checkpoint; sigmoid uses an explicit "
            "host UInt5 adapter between Hagen and the TTFS pool"
        ),
    )
    parser.add_argument(
        "--pwm-backend",
        choices=("torch", "hagen-mock", "hagen-hardware"),
        default="torch",
    )
    parser.add_argument(
        "--pool-backend",
        choices=("mock", "replay", "hardware"),
        default="mock",
    )
    parser.add_argument(
        "--pooling-domain",
        choices=("ttfs", "potential"),
        default="ttfs",
    )
    parser.add_argument(
        "--pool-mapping",
        choices=("dedicated", "time-multiplexed"),
        default="dedicated",
    )
    parser.add_argument(
        "--placements",
        choices=("local-pool", "cross-quadrant"),
        nargs="+",
        default=["local-pool", "cross-quadrant"],
    )
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--pool-trials", type=int, default=8)
    parser.add_argument("--pool-calibration-trials", type=int, default=32)
    parser.add_argument(
        "--pool-calibration-trial-chunk-size",
        type=int,
        default=DEFAULT_POOL_CALIBRATION_TRIAL_CHUNK_SIZE,
        help="calibration trials per disposable hardware worker",
    )
    parser.add_argument("--pool-sample-chunk-size", type=int, default=64)
    parser.add_argument(
        "--pool-replica-sample-budget",
        type=int,
        default=DEFAULT_POOL_REPLICA_SAMPLE_BUDGET,
        help="hardware cap for pool_size times samples in one spiking graph",
    )
    parser.add_argument("--hagen-row-chunk-size", type=int, default=512)
    parser.add_argument("--condition-worker-max-attempts", type=int, default=3)
    parser.add_argument("--condition-worker-retry-backoff-s", type=float, default=20.0)
    parser.add_argument("--condition-worker-idle-timeout-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--converted-checkpoint", type=Path)
    parser.add_argument("--replay-events", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)

    parser.add_argument("--hagen-calibration", type=Path)
    parser.add_argument("--spiking-calibration", type=Path)
    parser.add_argument("--allow-environment-calibration", action="store_true")
    parser.add_argument(
        "--hagen-tiling",
        choices=("auto", "high-level", "host-128"),
        default="auto",
    )
    parser.add_argument("--hagen-tile-size", type=int, default=128)
    parser.add_argument("--hagen-num-sends", type=int)
    parser.add_argument("--hagen-wait-between-events", type=int, default=5)
    parser.add_argument("--hagen-hidden-shift", type=int, default=1)
    parser.add_argument(
        "--relu-boundary",
        choices=("implicit-lower-bound-host", "hagen-converting-relu"),
        default="implicit-lower-bound-host",
        help=(
            "hidden activation boundary: default maps raw Hagen PWM output through "
            "a host-mediated V_lb=0 Potential before TTFS; the other choice is a "
            "Hagen ConvertingReLU baseline"
        ),
    )

    parser.add_argument("--dt-s", type=float, default=1.0e-6)
    parser.add_argument("--input-early-s", type=float, default=5.0e-6)
    parser.add_argument("--input-late-s", type=float, default=25.0e-6)
    parser.add_argument("--deadline-s", type=float, default=60.0e-6)
    parser.add_argument("--inter-batch-wait-s", type=float, default=50.0e-6)
    parser.add_argument("--tau-m-s", type=float, default=20.0e-6)
    parser.add_argument("--tau-syn-s", type=float, default=1.0e-6)
    parser.add_argument("--leak", type=float, default=80.0)
    parser.add_argument("--reset", type=float, default=80.0)
    parser.add_argument("--threshold", type=float, default=125.0)
    parser.add_argument("--refractory-time-s", type=float, default=1.0e-6)
    parser.add_argument("--i-synin-gm", type=float, default=500.0)
    parser.add_argument("--synapse-dac-bias", type=float, default=600.0)
    parser.add_argument("--synaptic-weight", type=float, default=63.0)
    parser.add_argument("--input-fan-in", type=int, default=4)
    parser.add_argument("--raw-time-scale-s", type=float)
    parser.add_argument("--condition-worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pool-chunk-worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--condition-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-first-hidden", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--first-hidden-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--condition-hagen-avg", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--condition-code-revision", help=argparse.SUPPRESS)
    parser.add_argument("--condition-hagen-calibration-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--condition-spiking-calibration-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--condition-checkpoint-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--condition-converted-sha256", help=argparse.SUPPRESS)
    return parser.parse_args()


_WORKER_PATH_FIELDS = {
    "dataset_cache",
    "checkpoint",
    "converted_checkpoint",
    "replay_events",
    "output_dir",
    "hagen_calibration",
    "spiking_calibration",
    "condition_worker_config",
    "pool_chunk_worker_config",
    "first_hidden_cache",
}


def _apply_condition_worker_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.condition_worker_config is None:
        return args
    config_path = args.condition_worker_config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - set(vars(args)))
    if unknown:
        raise ValueError(f"condition worker config has unknown keys: {unknown}")
    for key, value in payload.items():
        if key in _WORKER_PATH_FIELDS and value is not None:
            value = Path(value)
        setattr(args, key, value)
    args.condition_worker_config = config_path
    args.condition_worker = True
    return args


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_normalize(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(child) for child in value]
    return value


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_normalize(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_architecture(args: argparse.Namespace) -> None:
    architecture = ARCHITECTURES[args.architecture]
    if args.pool_sample_chunk_size <= 0:
        raise ValueError("pool_sample_chunk_size must be positive")
    if args.pool_replica_sample_budget <= 0:
        raise ValueError("pool_replica_sample_budget must be positive")
    if args.pool_calibration_trial_chunk_size <= 0:
        raise ValueError("pool_calibration_trial_chunk_size must be positive")
    if args.hagen_row_chunk_size <= 0:
        raise ValueError("hagen_row_chunk_size must be positive")
    if args.condition_worker_max_attempts <= 0:
        raise ValueError("condition_worker_max_attempts must be positive")
    if args.condition_worker_retry_backoff_s < 0:
        raise ValueError("condition_worker_retry_backoff_s must be non-negative")
    if args.condition_worker_idle_timeout_s <= 0:
        raise ValueError("condition_worker_idle_timeout_s must be positive")
    if architecture.task != args.task:
        raise ValueError(
            f"architecture {architecture.name} belongs to {architecture.task}, not {args.task}"
        )
    if architecture.hidden_features == 128 and args.pool_mapping == "dedicated":
        raise ValueError("mnist-128 requires --pool-mapping time-multiplexed")
    if args.pooling_domain == "potential" and args.pwm_backend == "torch":
        raise ValueError("potential-domain pooling requires a Hagen PWM backend")
    if args.phase.startswith("hardware"):
        if args.pwm_backend != "hagen-hardware" or args.pool_backend != "hardware":
            raise ValueError(
                "hardware phases require --pwm-backend hagen-hardware and --pool-backend hardware"
            )


def _training_config(args: argparse.Namespace, seed: int) -> TrainingConfig:
    config = TrainingConfig.for_architecture(ARCHITECTURES[args.architecture], seed)
    if args.quick:
        config = replace(config, epochs=min(config.epochs, 20))
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    return config


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint or (args.output_dir / "checkpoint.pt")


def _converted_path(args: argparse.Namespace) -> Path:
    return args.converted_checkpoint or (args.output_dir / "converted_checkpoint.pt")


def _load_float_checkpoint(args: argparse.Namespace) -> ToyMLP:
    path = _checkpoint_path(args)
    if not path.is_file():
        raise FileNotFoundError(
            f"float checkpoint is missing: {path}; run --phase train first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = ARCHITECTURES[payload["architecture"]]
    checkpoint_activation = payload.get("activation", "relu")
    if checkpoint_activation != args.activation:
        raise ValueError(
            "checkpoint activation does not match --activation: "
            f"{checkpoint_activation} != {args.activation}"
        )
    model = ToyMLP(architecture, activation=checkpoint_activation)
    model.load_state_dict(payload["state_dict"])
    expected = payload.get("parameter_sha256")
    actual = parameter_sha256(model)
    if expected is not None and actual != expected:
        raise RuntimeError("float checkpoint parameter hash does not match its manifest")
    return model.eval()


def _load_or_convert(
    args: argparse.Namespace,
    model: ToyMLP,
    calibration_x: torch.Tensor,
) -> ConvertedToyModel:
    path = _converted_path(args)
    if path.is_file():
        converted = deserialize_converted_model(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        if converted.manifest.source_parameter_sha256 != parameter_sha256(model):
            raise RuntimeError("converted checkpoint does not belong to the float checkpoint")
        if converted.manifest.activation != model.activation:
            raise RuntimeError("converted checkpoint activation does not match float checkpoint")
        return converted
    converted = convert_float_model(model, calibration_x)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialize_converted_model(converted), path)
    _json_write(args.output_dir / "conversion.json", converted.manifest.to_dict())
    return converted


def train_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture = ARCHITECTURES[args.architecture]
    dataset = load_dataset_bundle(args.task, cache_dir=args.dataset_cache)
    if args.max_train_samples is not None:
        dataset = replace(
            dataset,
            train_x=dataset.train_x[: args.max_train_samples],
            train_y=dataset.train_y[: args.max_train_samples],
        )
    rows: list[dict[str, Any]] = []
    seed_zero_payload: dict[str, Any] | None = None
    for seed in args.train_seeds:
        config = _training_config(args, seed)
        started = perf_counter()
        model, history = train_float_model(
            architecture,
            dataset,
            config,
            activation=args.activation,
        )
        with torch.no_grad():
            test = classification_metrics(model(dataset.test_x), dataset.test_y)
        digest = parameter_sha256(model)
        final = history[-1]
        rows.append(
            {
                "seed": seed,
                **final,
                "test_accuracy": test["accuracy"],
                "test_nll": test["nll"],
                "elapsed_s": perf_counter() - started,
                "parameter_sha256": digest,
            }
        )
        payload = {
            "schema_version": 1,
            "architecture": architecture.name,
            "activation": args.activation,
            "seed": seed,
            "training_config": asdict(config),
            "dataset": dataset.metadata,
            "state_dict": model.state_dict(),
            "parameter_sha256": digest,
        }
        torch.save(payload, args.output_dir / f"checkpoint_seed{seed}.pt")
        if seed == 0:
            seed_zero_payload = payload
    if seed_zero_payload is None:
        raise ValueError("--train-seeds must include preregistered hardware seed 0")
    torch.save(seed_zero_payload, args.output_dir / "checkpoint.pt")
    _write_csv(args.output_dir / "training_metrics.csv", rows)
    print(f"Wrote float checkpoints to {args.output_dir}")


def convert_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_bundle(args.task, cache_dir=args.dataset_cache)
    model = _load_float_checkpoint(args)
    converted = convert_float_model(model, dataset.calibration_x)
    converted_path = _converted_path(args)
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialize_converted_model(converted), converted_path)
    _json_write(args.output_dir / "conversion.json", converted.manifest.to_dict())
    print(f"Wrote deterministic conversion to {args.output_dir}")


def _spiking_config(args: argparse.Namespace) -> BrainScaleS2PoolConfig:
    return BrainScaleS2PoolConfig(
        dt_s=args.dt_s,
        input_early_s=args.input_early_s,
        input_late_s=args.input_late_s,
        observation_deadline_s=args.deadline_s,
        inter_batch_wait_s=args.inter_batch_wait_s,
        tau_mem_s=args.tau_m_s,
        tau_syn_s=args.tau_syn_s,
        leak=args.leak,
        reset=args.reset,
        threshold=args.threshold,
        refractory_time_s=args.refractory_time_s,
        i_synin_gm=args.i_synin_gm,
        synapse_dac_bias=args.synapse_dac_bias,
        synaptic_weight=args.synaptic_weight,
        input_fan_in=args.input_fan_in,
        pool_sizes=tuple(args.pool_sizes),
        placements=("same-quadrant",),
        routings=("broadcast",),
        trials=max(2, args.pool_trials),
        seed=args.seed,
        calibration_path=args.spiking_calibration,
        allow_environment_calibration=args.allow_environment_calibration,
        raw_time_scale_s=args.raw_time_scale_s,
    )


def _hagen_backend(args: argparse.Namespace) -> HagenPWMBackend | None:
    if args.pwm_backend == "torch":
        return None
    return HagenPWMBackend(
        HagenConfig(
            mode="mock" if args.pwm_backend == "hagen-mock" else "hardware",
            calibration_path=args.hagen_calibration,
            allow_environment_calibration=args.allow_environment_calibration,
            tiling=args.hagen_tiling,
            tile_size=args.hagen_tile_size,
            wait_between_events=args.hagen_wait_between_events,
            num_sends=args.hagen_num_sends,
            hidden_shift=args.hagen_hidden_shift,
        )
    )


def _pool_backend(args: argparse.Namespace) -> Any:
    if args.pool_backend == "mock":
        return MockToyPoolBackend()
    if args.pool_backend == "replay":
        return ReplayToyPoolBackend(args.replay_events)
    if not GroupedHardwarePoolBackend.dependencies_available():
        raise RuntimeError("hardware pooling requires the EBRAINS hxtorch environment")
    return GroupedHardwarePoolBackend()


def _run_temporal_pool(
    args: argparse.Namespace,
    temporal_backend: Any,
    hidden_uint5: torch.Tensor,
    pool_config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
) -> Any:
    requested_chunk_size = args.pool_sample_chunk_size
    effective_chunk_size = requested_chunk_size
    if args.pool_backend == "hardware":
        effective_chunk_size = min(
            requested_chunk_size,
            max(1, args.pool_replica_sample_budget // pool_config.pool_size),
        )

    def with_chunk_metadata(result: Any) -> Any:
        return replace(
            result,
            metadata={
                **result.metadata,
                "requested_pool_sample_chunk_size": requested_chunk_size,
                "effective_pool_sample_chunk_size": effective_chunk_size,
                "pool_replica_sample_budget": args.pool_replica_sample_budget,
            },
        )

    if args.pool_backend == "hardware" and getattr(args, "condition_worker", False):
        return with_chunk_metadata(
            _run_isolated_pool_chunks(
                args,
                hidden_uint5,
                pool_config,
                spiking_config,
                effective_chunk_size,
            )
        )

    if (
        args.pool_backend != "hardware"
        or hidden_uint5.shape[0] <= effective_chunk_size
    ):
        return with_chunk_metadata(
            temporal_backend.run_uint5(hidden_uint5, pool_config, spiking_config)
        )
    results = []
    for start in range(0, hidden_uint5.shape[0], effective_chunk_size):
        stop = min(start + effective_chunk_size, hidden_uint5.shape[0])
        print(
            f"  pool sample chunk [{start}:{stop}) / {hidden_uint5.shape[0]} "
            f"(effective={effective_chunk_size}, requested={requested_chunk_size})",
            flush=True,
        )
        results.append(
            temporal_backend.run_uint5(
                hidden_uint5[start:stop],
                pool_config,
                spiking_config,
            )
        )
    return with_chunk_metadata(concatenate_toy_pool_results(results))


def _run_hagen_output(
    args: argparse.Namespace,
    hagen: HagenPWMBackend,
    converted: ConvertedToyModel,
    flat_hidden: torch.Tensor,
) -> HagenResult:
    if (
        hagen.config.mode != "hardware"
        or flat_hidden.shape[0] <= args.hagen_row_chunk_size
    ):
        return hagen.output_layer(converted, flat_hidden)
    chunk_results: list[HagenResult] = []
    row_chunks: list[dict[str, Any]] = []
    for start in range(0, flat_hidden.shape[0], args.hagen_row_chunk_size):
        stop = min(start + args.hagen_row_chunk_size, flat_hidden.shape[0])
        print(
            f"  Hagen output row chunk [{start}:{stop}) / {flat_hidden.shape[0]}",
            flush=True,
        )
        result = hagen.output_layer(converted, flat_hidden[start:stop])
        chunk_results.append(result)
        row_chunks.append(
            {
                "row_start": start,
                "row_stop": stop,
                "metadata": result.metadata,
            }
        )
    combined = torch.cat([result.value for result in chunk_results], dim=0)
    metadata = dict(chunk_results[0].metadata)
    for key in ("input_shape", "output_shape", "elapsed_s"):
        metadata.pop(key, None)
    metadata.update(
        {
            "chunked": True,
            "row_chunk_size": args.hagen_row_chunk_size,
            "row_chunk_count": len(chunk_results),
            "row_chunks": row_chunks,
            "input_shape": list(flat_hidden.shape),
            "output_shape": list(combined.shape),
            "elapsed_s": sum(
                float(result.metadata.get("elapsed_s", 0.0))
                for result in chunk_results
            ),
        }
    )
    return HagenResult(combined, metadata)


def _evaluate_readout_ablations(
    args: argparse.Namespace,
    hagen: HagenPWMBackend | None,
    converted: ConvertedToyModel,
    pool_result: ToyPoolResult,
    ideal_hidden_uint5: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Evaluate physical/torch readouts before and after position-wise miss repair."""
    decoded = pool_result.decoded_uint5
    if ideal_hidden_uint5.shape != decoded.shape[1:]:
        raise ValueError("ideal hidden tensor does not match pooled hidden activations")
    repaired = torch.where(
        pool_result.all_miss,
        ideal_hidden_uint5.unsqueeze(0).expand_as(decoded),
        decoded,
    )
    flat_hidden = decoded.reshape(-1, converted.architecture.hidden_features)
    flat_repaired = repaired.reshape(-1, converted.architecture.hidden_features)
    _, flat_torch_logits = converted.output_from_hidden(flat_hidden)
    _, flat_torch_oracle_logits = converted.output_from_hidden(flat_repaired)

    physical_oracle_reexecuted = bool(pool_result.all_miss.any()) and hagen is not None
    if hagen is None:
        flat_logits = flat_torch_logits
        flat_oracle_logits = flat_torch_oracle_logits
        output_metadata: dict[str, Any] = {"backend": "torch"}
    elif physical_oracle_reexecuted:
        combined_hidden = torch.cat((flat_hidden, flat_repaired), dim=0)
        output = _run_hagen_output(args, hagen, converted, combined_hidden)
        row_count = flat_hidden.shape[0]
        flat_logits = output.value[:row_count]
        flat_oracle_logits = output.value[row_count:]
        output_metadata = {
            **output.metadata,
            "row_segments": {
                "physical": [0, row_count],
                "physical_oracle_miss_repair": [row_count, 2 * row_count],
            },
        }
    else:
        output = _run_hagen_output(args, hagen, converted, flat_hidden)
        flat_logits = output.value
        flat_oracle_logits = output.value
        output_metadata = output.metadata

    shape = (
        decoded.shape[0],
        decoded.shape[1],
        converted.architecture.output_features,
    )
    output_metadata = {
        **output_metadata,
        "physical_oracle_miss_repair_reexecuted": physical_oracle_reexecuted,
        "torch_readout_ablation": True,
        "oracle_repair_policy": "replace-all-miss-position-with-ideal-hidden",
    }
    return (
        flat_logits.reshape(shape).to(torch.float32),
        flat_oracle_logits.reshape(shape).to(torch.float32),
        flat_torch_logits.reshape(shape).to(torch.float32),
        flat_torch_oracle_logits.reshape(shape).to(torch.float32),
        output_metadata,
    )


def _select_test_data(args: argparse.Namespace, dataset: Any) -> tuple[torch.Tensor, torch.Tensor]:
    limit = args.max_test_samples
    if args.phase == "hardware-smoke":
        limit = min(limit or 12, 12)
    elif args.task == "mnist" and args.phase == "hardware-eval" and limit is None:
        limit = 128
    if limit is None:
        return dataset.test_x, dataset.test_y
    return dataset.test_x[:limit], dataset.test_y[:limit]


def evaluation_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_started = perf_counter()
    dataset = load_dataset_bundle(args.task, cache_dir=args.dataset_cache)
    test_x, labels = _select_test_data(args, dataset)
    runtime: dict[str, Any] = {"dataset_load_s": perf_counter() - dataset_started}
    model = _load_float_checkpoint(args)
    converted = _load_or_convert(args, model, dataset.calibration_x)
    with torch.no_grad():
        float_logits = model(test_x).detach().cpu()
    ideal = converted.forward(test_x)
    ideal_logits = ideal.logits_int8.to(torch.float32)

    pool_sizes = [1, 4] if args.quick or args.phase == "hardware-smoke" else args.pool_sizes
    placements = [args.placements[0]] if args.quick or args.phase == "hardware-smoke" else args.placements
    pool_trials = min(args.pool_trials, 2) if args.quick else args.pool_trials
    calibration_trials = (
        min(args.pool_calibration_trials, 4) if args.quick else args.pool_calibration_trials
    )
    hagen = _hagen_backend(args)
    temporal_backend = _pool_backend(args)
    spiking_config = _spiking_config(args)
    evaluations: list[ToyConditionEvaluation] = []
    first_cache: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
    if args.first_hidden_cache is not None:
        cached = torch.load(args.first_hidden_cache, map_location="cpu", weights_only=False)
        cached_avg = int(cached["hagen_avg"])
        cached_hidden = cached["first_hidden"].to(torch.int32)
        if cached_hidden.shape != (test_x.shape[0], converted.architecture.hidden_features):
            raise ValueError("shared first-hidden cache does not match the evaluation shape")
        if cached.get("source_parameter_sha256") != parameter_sha256(model):
            raise ValueError("shared first-hidden cache belongs to a different checkpoint")
        if cached.get("activation", "relu") != args.activation:
            raise ValueError("shared first-hidden cache uses a different activation")
        if args.activation == "relu" and cached.get("relu_boundary") != args.relu_boundary:
            raise ValueError("shared first-hidden cache uses a different ReLU boundary")
        first_cache[cached_avg] = (cached_hidden, cached["metadata"])

    for placement in placements:
        for effective_pool_size in pool_sizes:
            if args.pooling_domain == "ttfs":
                hagen_avg = 1
                temporal_pool_size = effective_pool_size
            else:
                hagen_avg = effective_pool_size
                temporal_pool_size = 1
            if hagen_avg not in first_cache:
                if hagen is None:
                    first_hidden = ideal.hidden_uint5
                    first_metadata = {"backend": "torch", "avg": 1}
                else:
                    input_uint5 = converted.encode_input(test_x)
                    first = hagen.first_layer(
                        converted,
                        input_uint5,
                        avg=hagen_avg,
                        relu_boundary=args.relu_boundary,
                        activation=args.activation,
                    )
                    first_hidden = first.value
                    first_metadata = first.metadata
                first_cache[hagen_avg] = (first_hidden, first_metadata)
            first_hidden, first_metadata = first_cache[hagen_avg]
            pool_config = ToyPoolConfig(
                pool_size=temporal_pool_size,
                logical_neurons=converted.architecture.hidden_features,
                placement=placement,
                mapping=args.pool_mapping,
                inference_trials=pool_trials,
                calibration_trials=calibration_trials,
                seed=args.seed + effective_pool_size,
            )
            print(
                "Running",
                f"task={args.task}",
                f"M={effective_pool_size}",
                f"domain={args.pooling_domain}",
                f"placement={placement}",
                f"mapping={args.pool_mapping}",
                flush=True,
            )
            started = perf_counter()
            pool_result = _run_temporal_pool(
                args,
                temporal_backend,
                first_hidden,
                pool_config,
                spiking_config,
            )
            (
                logits,
                oracle_miss_repair_logits,
                torch_readout_logits,
                torch_oracle_miss_repair_logits,
                output_metadata,
            ) = _evaluate_readout_ablations(
                args,
                hagen,
                converted,
                pool_result,
                ideal.hidden_uint5,
            )
            key = (
                f"{args.pooling_domain}_M{effective_pool_size}_"
                f"{placement}_{args.pool_mapping}"
            )
            runtime[key] = {
                "elapsed_s": perf_counter() - started,
                "samples": test_x.shape[0],
                "trials": pool_trials,
            }
            evaluations.append(
                ToyConditionEvaluation(
                    key=key,
                    pool_size=effective_pool_size,
                    pooling_domain=args.pooling_domain,
                    pool_result=pool_result,
                    nominal_hidden_uint5=first_hidden,
                    logits=logits,
                    oracle_miss_repair_logits=oracle_miss_repair_logits,
                    torch_readout_logits=torch_readout_logits,
                    torch_oracle_miss_repair_logits=torch_oracle_miss_repair_logits,
                    pwm_metadata={"first": first_metadata, "output": output_metadata},
                )
            )

    total_condition_s = sum(
        float(value["elapsed_s"])
        for value in runtime.values()
        if isinstance(value, dict) and "elapsed_s" in value
    )
    if args.task == "mnist" and test_x.shape[0] > 0:
        per_sample = total_condition_s / test_x.shape[0]
        runtime["mnist_estimates"] = {
            "measured_samples": test_x.shape[0],
            "seconds_per_sample_all_conditions": per_sample,
            "estimated_1k_s": per_sample * 1_000,
            "estimated_10k_s": per_sample * 10_000,
            "formal_run_requires_explicit_max_test_samples": True,
        }

    manifest = {
        "phase": args.phase,
        "task": args.task,
        "architecture": args.architecture,
        "activation": args.activation,
        "dataset": dataset.metadata,
        "test_samples": test_x.shape[0],
        "pwm_backend": args.pwm_backend,
        "pool_backend": args.pool_backend,
        "pooling_domain": args.pooling_domain,
        "relu_boundary": args.relu_boundary,
        "pool_mapping": args.pool_mapping,
        "pool_sample_chunk_size": args.pool_sample_chunk_size,
        "pool_replica_sample_budget": args.pool_replica_sample_budget,
        "pool_calibration_trial_chunk_size": args.pool_calibration_trial_chunk_size,
        "hagen_row_chunk_size": args.hagen_row_chunk_size,
        "pool_sizes": pool_sizes,
        "placements": placements,
        "conversion": converted.manifest.to_dict(),
        "float_parameter_sha256": parameter_sha256(model),
        "hagen_calibration_sha256": file_sha256(args.hagen_calibration),
        "spiking_calibration_sha256": file_sha256(args.spiking_calibration),
        "spiking_config": spiking_config.to_manifest_dict(),
        "analyses": {
            "miss_stratification": "nominal-hidden-uint5-zero-vs-positive",
            "activation_error_support": "logical-pool-non-miss-only",
            "activation_error_units": "uint5-code",
            "oracle_miss_repair": "replace-all-miss-position-with-ideal-hidden",
            "oracle_readout_backend": args.pwm_backend,
            "readout_ablation": "same-pooled-hidden-through-integer-torch-readout",
        },
        "conditions": [
            {
                "key": item.key,
                "pool_size": item.pool_size,
                "pooling_domain": item.pooling_domain,
                "hagen_avg": item.pool_size if item.pooling_domain == "potential" else 1,
                "temporal_pool_size": item.pool_result.pool_size,
                "placement": item.pool_result.placement,
                "mapping": item.pool_result.mapping,
                "physical_coordinates": item.pool_result.physical_coordinates,
                "pool_metadata": item.pool_result.metadata,
                "pwm_metadata": item.pwm_metadata,
            }
            for item in evaluations
        ],
        "environment": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "ebrains_release": os.environ.get("EBRAINS_RELEASE"),
        },
        "claims": {
            "host_mediated": True,
            "host_free_latency_or_energy": False,
            "transformer_hardware_execution": False,
            "replay_is_hardware_evidence": False,
            "implicit_relu_continuous_on_chip": False,
            "implicit_relu_host_mediated": (
                args.activation == "relu"
                and
                args.relu_boundary == "implicit-lower-bound-host"
            ),
            "hagen_converting_relu_baseline": (
                args.activation == "relu"
                and
                args.relu_boundary == "hagen-converting-relu"
            ),
            "sigmoid_physical_subcircuit": False,
            "sigmoid_host_mediated_adapter": args.activation == "sigmoid",
        },
    }
    metrics = write_toy_artifacts(
        args.output_dir,
        labels=labels,
        float_logits=float_logits,
        ideal_logits=ideal_logits,
        ideal_hidden_uint5=ideal.hidden_uint5,
        evaluations=evaluations,
        manifest=manifest,
        runtime=runtime,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    float_accuracy = next(row["accuracy"] for row in metrics if row["condition"] == "float-ann")
    ideal_accuracy = next(
        row["accuracy"] for row in metrics if row["condition"] == "ideal-converted"
    )
    print(
        f"Wrote toy HIL artifacts to {args.output_dir}; "
        f"float_accuracy={float_accuracy:.4f}, ideal_accuracy={ideal_accuracy:.4f}"
    )


def _serialize_worker_config(args: argparse.Namespace) -> dict[str, Any]:
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (tuple, list)):
            return [normalize(child) for child in value]
        return value

    return {
        key: normalize(value)
        for key, value in vars(args).items()
        if key != "condition_worker_config"
    }


def _terminate_worker_process(
    process: subprocess.Popen[str],
    *,
    process_group: bool,
    grace_s: float = 5.0,
) -> None:
    """Terminate one worker or its explicitly isolated process group."""
    if process.poll() is not None:
        return
    try:
        if process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if process_group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait()


def _run_streamed_worker_command(
    command: list[str],
    *,
    cwd: Path,
    check: bool,
    idle_timeout_s: float,
    attempt_log_path: Path,
    isolate_process_group: bool,
) -> subprocess.CompletedProcess[str]:
    """Stream a child worker while enforcing a no-output timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=isolate_process_group,
    )
    if process.stdout is None:
        _terminate_worker_process(process, process_group=isolate_process_group)
        raise RuntimeError("condition worker stdout pipe was not created")

    messages: Queue[str | None] = Queue()

    def pump_output() -> None:
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    Thread(target=pump_output, name="condition-worker-output", daemon=True).start()
    attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with attempt_log_path.open("w", encoding="utf-8") as attempt_log:
            last_output_at = monotonic()
            while True:
                idle_s = monotonic() - last_output_at
                remaining_s = idle_timeout_s - idle_s
                if remaining_s <= 0:
                    _terminate_worker_process(
                        process,
                        process_group=isolate_process_group,
                    )
                    raise subprocess.TimeoutExpired(command, idle_timeout_s)
                try:
                    line = messages.get(timeout=min(30.0, remaining_s))
                except Empty:
                    idle_s = monotonic() - last_output_at
                    print(
                        f"Worker is still waiting for native output "
                        f"({idle_s:.0f}s idle; timeout {idle_timeout_s:g}s)",
                        flush=True,
                    )
                    continue
                if line is None:
                    break
                last_output_at = monotonic()
                print(line, end="", flush=True)
                attempt_log.write(line)
                attempt_log.flush()
        returncode = process.wait()
    except BaseException:
        _terminate_worker_process(process, process_group=isolate_process_group)
        raise
    finally:
        process.stdout.close()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


def _run_worker_command_with_retries(
    command: list[str],
    worker_dir: Path,
    *,
    max_attempts: int,
    retry_backoff_s: float,
    idle_timeout_s: float,
    isolate_process_group: bool = True,
    runner: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Run one isolated worker with bounded process-level reconnect attempts."""
    status_path = worker_dir / "worker_status.json"
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started_at = datetime.now(timezone.utc).isoformat()
        print(
            f"Worker {worker_dir.name} attempt {attempt}/{max_attempts}",
            flush=True,
        )
        attempt_log_path = worker_dir / f"worker_attempt_{attempt}.log"
        try:
            if runner is None:
                _run_streamed_worker_command(
                    command,
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    idle_timeout_s=idle_timeout_s,
                    attempt_log_path=attempt_log_path,
                    isolate_process_group=isolate_process_group,
                )
            else:
                runner(command, cwd=REPOSITORY_ROOT, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            attempts.append(
                {
                    "attempt": attempt,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "returncode": 124 if timed_out else error.returncode,
                    "status": "failed",
                    "failure": "idle-timeout" if timed_out else "process-exit",
                    "attempt_log": attempt_log_path.name if runner is None else None,
                }
            )
            final = attempt == max_attempts
            _json_write(
                status_path,
                {
                    "status": "failed" if final else "retrying",
                    "max_attempts": max_attempts,
                    "retry_backoff_s": retry_backoff_s,
                    "idle_timeout_s": idle_timeout_s,
                    "attempts": attempts,
                },
            )
            if final:
                raise
            delay_s = retry_backoff_s * attempt
            failure = "idle timeout" if timed_out else f"exit {error.returncode}"
            print(
                f"Worker {worker_dir.name} failed with {failure}; "
                f"retrying in {delay_s:g}s",
                flush=True,
            )
            sleeper(delay_s)
            continue
        attempts.append(
            {
                "attempt": attempt,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": 0,
                "status": "passed",
                "attempt_log": attempt_log_path.name if runner is None else None,
            }
        )
        status = {
            "status": "passed",
            "max_attempts": max_attempts,
            "retry_backoff_s": retry_backoff_s,
            "idle_timeout_s": idle_timeout_s,
            "attempts": attempts,
        }
        _json_write(status_path, status)
        return status
    raise AssertionError("unreachable worker retry state")


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _timing_calibration_sha256(calibration: TimingCalibration) -> str:
    digest = hashlib.sha256()
    digest.update(repr(float(calibration.response_delay_s)).encode("ascii"))
    digest.update(str(int(calibration.calibration_trials)).encode("ascii"))
    digest.update(
        calibration.neuron_offset_s.detach().cpu().contiguous().numpy().tobytes()
    )
    return digest.hexdigest()


def _load_timing_calibration(path: Path) -> TimingCalibration:
    calibration = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(calibration, TimingCalibration):
        raise TypeError(f"timing calibration at {path} has an invalid type")
    return calibration


def _load_timing_observation(path: Path) -> TimingCalibrationObservation:
    observation = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(observation, TimingCalibrationObservation):
        raise TypeError(f"timing observation at {path} has an invalid type")
    return observation


def _load_pool_chunk_result(path: Path) -> ToyPoolResult:
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, ToyPoolResult):
        raise TypeError(f"pool chunk result at {path} has an invalid type")
    return result


def _launch_pool_calibration_worker(
    args: argparse.Namespace,
    chunk_dir: Path,
    pool_config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
) -> TimingCalibrationObservation:
    """Run or resume a bounded calibration-trial chunk in its own process."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    result_path = chunk_dir / "calibration_observation.pt"
    config_path = chunk_dir / "pool_chunk_config.json"
    payload = _json_normalize({
        "schema_version": 2,
        "worker_kind": "timing-calibration",
        "code_revision": _git_revision(),
        "pool_config": asdict(pool_config),
        "spiking_config": spiking_config.to_manifest_dict(),
        "result_file": result_path.name,
    })
    same_config = False
    if config_path.is_file():
        try:
            same_config = json.loads(config_path.read_text(encoding="utf-8")) == payload
        except (OSError, json.JSONDecodeError):
            same_config = False
    if same_config and result_path.is_file():
        try:
            observation = _load_timing_observation(result_path)
        except (OSError, RuntimeError, TypeError):
            pass
        else:
            print(f"  Reusing calibration chunk {chunk_dir.name}", flush=True)
            return observation

    result_path.unlink(missing_ok=True)
    _json_write(config_path, payload)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(chunk_dir.resolve()),
        "--pool-chunk-worker-config",
        str(config_path.resolve()),
    ]
    _run_worker_command_with_retries(
        command,
        chunk_dir,
        max_attempts=args.condition_worker_max_attempts,
        retry_backoff_s=args.condition_worker_retry_backoff_s,
        idle_timeout_s=args.condition_worker_idle_timeout_s,
        isolate_process_group=False,
    )
    if not result_path.is_file():
        raise RuntimeError(f"calibration worker {chunk_dir.name} missed its result")
    return _load_timing_observation(result_path)


def _launch_pool_chunk_worker(
    args: argparse.Namespace,
    chunk_dir: Path,
    hidden_chunk: torch.Tensor,
    pool_config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
    timing_calibration: TimingCalibration,
    timing_calibration_path: Path,
) -> ToyPoolResult:
    """Run or resume one physical sample chunk in its own process."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    input_path = chunk_dir / "pool_chunk_input.pt"
    result_path = chunk_dir / "pool_chunk_result.pt"
    config_path = chunk_dir / "pool_chunk_config.json"
    payload = _json_normalize({
        "schema_version": 2,
        "worker_kind": "inference",
        "code_revision": _git_revision(),
        "hidden_sha256": _tensor_sha256(hidden_chunk),
        "timing_calibration_file": str(timing_calibration_path.resolve()),
        "timing_calibration_sha256": _timing_calibration_sha256(
            timing_calibration
        ),
        "pool_config": asdict(pool_config),
        "spiking_config": spiking_config.to_manifest_dict(),
        "input_file": input_path.name,
        "result_file": result_path.name,
    })
    same_config = False
    if config_path.is_file():
        try:
            same_config = json.loads(config_path.read_text(encoding="utf-8")) == payload
        except (OSError, json.JSONDecodeError):
            same_config = False
    if same_config and result_path.is_file():
        try:
            result = _load_pool_chunk_result(result_path)
        except (OSError, RuntimeError, TypeError):
            pass
        else:
            print(f"  Reusing completed pool chunk {chunk_dir.name}", flush=True)
            return result

    result_path.unlink(missing_ok=True)
    torch.save(hidden_chunk.detach().cpu().to(torch.int32), input_path)
    _json_write(config_path, payload)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(chunk_dir.resolve()),
        "--pool-chunk-worker-config",
        str(config_path.resolve()),
    ]
    _run_worker_command_with_retries(
        command,
        chunk_dir,
        max_attempts=args.condition_worker_max_attempts,
        retry_backoff_s=args.condition_worker_retry_backoff_s,
        idle_timeout_s=args.condition_worker_idle_timeout_s,
        isolate_process_group=False,
    )
    if not result_path.is_file():
        raise RuntimeError(f"pool chunk worker {chunk_dir.name} missed its result")
    return _load_pool_chunk_result(result_path)


def _run_shared_timing_calibration(
    args: argparse.Namespace,
    pool_config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
    *,
    launcher: Callable[
        [argparse.Namespace, Path, ToyPoolConfig, BrainScaleS2PoolConfig],
        TimingCalibrationObservation,
    ]
    | None = None,
) -> tuple[TimingCalibration, Path, dict[str, Any]]:
    """Acquire one resumable split calibration for all inference chunks."""
    launch = launcher or _launch_pool_calibration_worker
    calibration_root = args.output_dir / "pool_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    timing_path = calibration_root / "timing_calibration.pt"
    manifest_path = calibration_root / "calibration_manifest.json"
    effective_chunk_size = min(
        args.pool_calibration_trial_chunk_size,
        pool_config.calibration_trials,
    )
    expected = _json_normalize({
        "schema_version": 1,
        "code_revision": _git_revision(),
        "pool_config": asdict(pool_config),
        "spiking_config": spiking_config.to_manifest_dict(),
        "requested_trial_chunk_size": args.pool_calibration_trial_chunk_size,
        "effective_trial_chunk_size": effective_chunk_size,
    })
    if manifest_path.is_file() and timing_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            calibration = _load_timing_calibration(timing_path)
        except (OSError, RuntimeError, TypeError, json.JSONDecodeError):
            pass
        else:
            if (
                manifest.get("configuration") == expected
                and manifest.get("timing_calibration_sha256")
                == _timing_calibration_sha256(calibration)
            ):
                print("  Reusing shared timing calibration", flush=True)
                return calibration, timing_path, manifest

    observations: list[TimingCalibrationObservation] = []
    worker_dirs: list[str] = []
    for start in range(0, pool_config.calibration_trials, effective_chunk_size):
        stop = min(start + effective_chunk_size, pool_config.calibration_trials)
        chunk_dir = calibration_root / f"trials_{start:04d}_{stop:04d}"
        print(
            f"  isolated calibration trials [{start}:{stop}) / "
            f"{pool_config.calibration_trials}",
            flush=True,
        )
        observations.append(
            launch(
                args,
                chunk_dir,
                replace(pool_config, calibration_trials=stop - start),
                spiking_config,
            )
        )
        worker_dirs.append(str(chunk_dir.relative_to(args.output_dir)))
    joined = concatenate_timing_calibration_observations(observations)
    calibration = calibrate_timing(
        joined.first_spike_s,
        joined.nominal_input_s,
    )
    temporary_path = timing_path.with_suffix(timing_path.suffix + ".tmp")
    torch.save(calibration, temporary_path)
    os.replace(temporary_path, timing_path)
    worker_status = {
        worker_dir: json.loads(
            (args.output_dir / worker_dir / "worker_status.json").read_text(
                encoding="utf-8"
            )
        )
        for worker_dir in worker_dirs
        if (args.output_dir / worker_dir / "worker_status.json").is_file()
    }
    manifest = {
        "configuration": expected,
        "timing_calibration_sha256": _timing_calibration_sha256(calibration),
        "response_delay_s": calibration.response_delay_s,
        "calibration_trials": calibration.calibration_trials,
        "physical_coordinates": joined.physical_coordinates.tolist(),
        "calibration_observation": joined.metadata,
        "calibration_worker_dirs": worker_dirs,
        "calibration_worker_status": worker_status,
    }
    _json_write(manifest_path, manifest)
    return calibration, timing_path, manifest


def _run_isolated_pool_chunks(
    args: argparse.Namespace,
    hidden_uint5: torch.Tensor,
    pool_config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
    effective_chunk_size: int,
    *,
    launcher: Callable[
        [
            argparse.Namespace,
            Path,
            torch.Tensor,
            ToyPoolConfig,
            BrainScaleS2PoolConfig,
            TimingCalibration,
            Path,
        ],
        ToyPoolResult,
    ]
    | None = None,
    calibration_launcher: Callable[
        [argparse.Namespace, Path, ToyPoolConfig, BrainScaleS2PoolConfig],
        TimingCalibrationObservation,
    ]
    | None = None,
) -> ToyPoolResult:
    """Share split calibration, then isolate every inference sample chunk."""
    launch = launcher or _launch_pool_chunk_worker
    timing, timing_path, calibration_manifest = _run_shared_timing_calibration(
        args,
        pool_config,
        spiking_config,
        launcher=calibration_launcher,
    )
    chunk_root = args.output_dir / "pool_chunks"
    results: list[ToyPoolResult] = []
    worker_dirs: list[str] = []
    for start in range(0, hidden_uint5.shape[0], effective_chunk_size):
        stop = min(start + effective_chunk_size, hidden_uint5.shape[0])
        chunk_dir = chunk_root / f"chunk_{start:06d}_{stop:06d}"
        print(
            f"  isolated pool sample chunk [{start}:{stop}) / "
            f"{hidden_uint5.shape[0]}",
            flush=True,
        )
        results.append(
            launch(
                args,
                chunk_dir,
                hidden_uint5[start:stop],
                pool_config,
                spiking_config,
                timing,
                timing_path,
            )
        )
        worker_dirs.append(str(chunk_dir.relative_to(args.output_dir)))
    joined = concatenate_toy_pool_results(results)
    worker_status = {
        worker_dir: json.loads(
            (args.output_dir / worker_dir / "worker_status.json").read_text(
                encoding="utf-8"
            )
        )
        for worker_dir in worker_dirs
        if (args.output_dir / worker_dir / "worker_status.json").is_file()
    }
    return replace(
        joined,
        metadata={
            **joined.metadata,
            "chunk_process_isolation": True,
            "chunk_worker_dirs": worker_dirs,
            "chunk_worker_status": worker_status,
            "calibration_strategy": "shared-split",
            "shared_timing_calibration": {
                "file": str(timing_path.relative_to(args.output_dir)),
                **calibration_manifest,
            },
        },
    )


def pool_chunk_worker_phase(args: argparse.Namespace) -> None:
    """Execute one serialized calibration or inference hardware chunk."""
    config_path = args.pool_chunk_worker_config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported pool chunk worker schema")
    result_path = config_path.parent / payload["result_file"]
    pool_config = ToyPoolConfig(**payload["pool_config"])
    spiking_payload = dict(payload["spiking_config"])
    for key in ("pool_sizes", "placements", "routings"):
        spiking_payload[key] = tuple(spiking_payload[key])
    if spiking_payload["calibration_path"] is not None:
        spiking_payload["calibration_path"] = Path(spiking_payload["calibration_path"])
    spiking_config = BrainScaleS2PoolConfig(**spiking_payload)
    backend = GroupedHardwarePoolBackend()
    if payload.get("worker_kind") == "timing-calibration":
        result = backend.observe_timing_calibration(pool_config, spiking_config)
        description = "calibration observation"
    elif payload.get("worker_kind") == "inference":
        input_path = config_path.parent / payload["input_file"]
        hidden_uint5 = torch.load(input_path, map_location="cpu", weights_only=True)
        if _tensor_sha256(hidden_uint5) != payload["hidden_sha256"]:
            raise ValueError("pool chunk input checksum mismatch")
        timing_path = Path(payload["timing_calibration_file"])
        timing = _load_timing_calibration(timing_path)
        if _timing_calibration_sha256(timing) != payload["timing_calibration_sha256"]:
            raise ValueError("pool chunk timing calibration checksum mismatch")
        result = backend.run_uint5(
            hidden_uint5,
            pool_config,
            spiking_config,
            timing_calibration=timing,
        )
        description = "pool inference chunk"
    else:
        raise ValueError("unsupported pool chunk worker kind")
    temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
    torch.save(result, temporary_path)
    os.replace(temporary_path, result_path)
    print(f"Wrote isolated {description} to {result_path}", flush=True)


def _run_condition_subprocess(
    args: argparse.Namespace,
    worker_dir: Path,
    *,
    required: tuple[str, ...],
    overrides: dict[str, Any],
) -> None:
    worker_dir.mkdir(parents=True, exist_ok=True)
    config_path = worker_dir / "worker_config.json"
    payload = _serialize_worker_config(args)
    payload.update(overrides)
    payload.update(
        {
            "condition_code_revision": _git_revision(),
            "condition_hagen_calibration_sha256": file_sha256(args.hagen_calibration),
            "condition_spiking_calibration_sha256": file_sha256(args.spiking_calibration),
            "condition_checkpoint_sha256": file_sha256(args.checkpoint),
            "condition_converted_sha256": file_sha256(args.converted_checkpoint),
        }
    )
    payload.update(
        {
            "condition_worker": True,
            "condition_worker_config": None,
            "output_dir": str(worker_dir.resolve()),
        }
    )
    same_config = False
    if config_path.is_file():
        try:
            same_config = json.loads(config_path.read_text(encoding="utf-8")) == payload
        except (OSError, json.JSONDecodeError):
            same_config = False
    if same_config and all((worker_dir / name).is_file() for name in required):
        print(f"Reusing completed condition worker {worker_dir.name}", flush=True)
        return
    _json_write(config_path, payload)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(worker_dir.resolve()),
        "--condition-worker-config",
        str(config_path.resolve()),
    ]
    print("Launching isolated worker:", worker_dir.name, flush=True)
    _run_worker_command_with_retries(
        command,
        worker_dir,
        max_attempts=args.condition_worker_max_attempts,
        retry_backoff_s=args.condition_worker_retry_backoff_s,
        idle_timeout_s=args.condition_worker_idle_timeout_s,
    )
    missing = [name for name in required if not (worker_dir / name).is_file()]
    if missing:
        status_path = worker_dir / "worker_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        _json_write(
            status_path,
            {
                **status,
                "status": "failed",
                "reason": "missing-required-artifacts",
                "missing": missing,
            },
        )
        raise RuntimeError(f"condition worker {worker_dir.name} missed artifacts: {missing}")


def prepare_first_hidden_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_bundle(args.task, cache_dir=args.dataset_cache)
    test_x, _ = _select_test_data(args, dataset)
    model = _load_float_checkpoint(args)
    converted = _load_or_convert(args, model, dataset.calibration_x)
    hagen = _hagen_backend(args)
    if hagen is None:
        raise ValueError("physical first-hidden preparation requires a Hagen backend")
    input_uint5 = converted.encode_input(test_x)
    first = hagen.first_layer(
        converted,
        input_uint5,
        avg=args.condition_hagen_avg,
        relu_boundary=args.relu_boundary,
        activation=args.activation,
    )
    payload = {
        "hagen_avg": args.condition_hagen_avg,
        "activation": args.activation,
        "relu_boundary": args.relu_boundary,
        "first_hidden": first.value.to(torch.int32),
        "metadata": first.metadata,
        "test_samples": test_x.shape[0],
        "source_parameter_sha256": parameter_sha256(model),
    }
    torch.save(payload, args.output_dir / "first_hidden.pt")
    _json_write(
        args.output_dir / "first_hidden_manifest.json",
        {
            key: value
            for key, value in payload.items()
            if key != "first_hidden"
        },
    )
    print(f"Wrote shared physical first hidden to {args.output_dir}", flush=True)


def _load_isolated_condition(
    worker_dir: Path,
) -> tuple[ToyConditionEvaluation, dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((worker_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads((worker_dir / "runtime.json").read_text(encoding="utf-8"))
    archive = torch.load(
        worker_dir / "intermediates.pt",
        map_location="cpu",
        weights_only=False,
    )
    if len(archive["conditions"]) != 1 or len(manifest["conditions"]) != 1:
        raise ValueError(f"isolated worker {worker_dir} must contain exactly one condition")
    key, tensors = next(iter(archive["conditions"].items()))
    condition_manifest = manifest["conditions"][0]
    if condition_manifest["key"] != key:
        raise ValueError(f"isolated worker {worker_dir} condition keys disagree")
    pool_result = ToyPoolResult(
        first_spike_s=tensors["first_spike_s"],
        fired=tensors["fired"],
        spike_count=tensors["spike_count"],
        nominal_input_s=tensors["nominal_input_s"],
        pooled_first_spike_s=tensors["pooled_first_spike_s"],
        decoded_uint5=tensors["decoded_uint5"],
        all_miss=tensors["all_miss"],
        physical_coordinates=tensors["physical_coordinates"],
        pool_size=int(tensors["first_spike_s"].shape[-1]),
        placement=condition_manifest["placement"],
        mapping=condition_manifest["mapping"],
        metadata=condition_manifest["pool_metadata"],
    )
    evaluation = ToyConditionEvaluation(
        key=key,
        pool_size=int(condition_manifest["pool_size"]),
        pooling_domain=condition_manifest["pooling_domain"],
        pool_result=pool_result,
        nominal_hidden_uint5=tensors["nominal_hidden_uint5"],
        logits=tensors["logits"],
        oracle_miss_repair_logits=tensors["oracle_miss_repair_logits"],
        torch_readout_logits=tensors["torch_readout_logits"],
        torch_oracle_miss_repair_logits=tensors[
            "torch_oracle_miss_repair_logits"
        ],
        pwm_metadata=condition_manifest["pwm_metadata"],
    )
    return evaluation, archive, manifest, runtime


def _aggregate_isolated_conditions(
    args: argparse.Namespace,
    worker_dirs: list[Path],
    first_hidden_dirs: dict[int, Path],
) -> None:
    evaluations: list[ToyConditionEvaluation] = []
    manifests: list[dict[str, Any]] = []
    runtimes: dict[str, Any] = {}
    reference_archive: dict[str, Any] | None = None
    for worker_dir in worker_dirs:
        evaluation, archive, manifest, runtime = _load_isolated_condition(worker_dir)
        if reference_archive is None:
            reference_archive = archive
        else:
            for key in ("labels", "float_logits", "ideal_logits", "ideal_hidden_uint5"):
                if not torch.equal(reference_archive[key], archive[key]):
                    raise ValueError(f"isolated workers disagree on {key}")
        evaluations.append(evaluation)
        manifests.append(manifest)
        runtimes[evaluation.key] = runtime
    if reference_archive is None:
        raise ValueError("no isolated condition workers were produced")
    base_manifest = {
        key: value
        for key, value in manifests[0].items()
        if key not in {"schema_version", "event_csv_coverage", "conditions"}
    }
    base_manifest.update(
        {
            "pool_sizes": sorted({item.pool_size for item in evaluations}),
            "placements": list(dict.fromkeys(item.pool_result.placement for item in evaluations)),
            "conditions": [
                condition
                for manifest in manifests
                for condition in manifest["conditions"]
            ],
            "condition_process_isolation": {
                "enabled": True,
                "worker_count": len(worker_dirs),
                "worker_directories": [
                    str(path.relative_to(args.output_dir)) for path in worker_dirs
                ],
                "worker_environments": {
                    manifest["conditions"][0]["key"]: manifest.get("environment")
                    for manifest in manifests
                },
                "worker_status": {
                    str(path.relative_to(args.output_dir)): json.loads(
                        (path / "worker_status.json").read_text(encoding="utf-8")
                    )
                    for path in [*first_hidden_dirs.values(), *worker_dirs]
                    if (path / "worker_status.json").is_file()
                },
                "shared_first_hidden": {
                    str(avg): str(path.relative_to(args.output_dir))
                    for avg, path in first_hidden_dirs.items()
                },
                "resumable": True,
            },
        }
    )
    metrics = write_toy_artifacts(
        args.output_dir,
        labels=reference_archive["labels"],
        float_logits=reference_archive["float_logits"],
        ideal_logits=reference_archive["ideal_logits"],
        ideal_hidden_uint5=reference_archive["ideal_hidden_uint5"],
        evaluations=evaluations,
        manifest=base_manifest,
        runtime={
            "condition_process_isolation": True,
            "condition_workers": runtimes,
        },
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    float_accuracy = next(row["accuracy"] for row in metrics if row["condition"] == "float-ann")
    ideal_accuracy = next(
        row["accuracy"] for row in metrics if row["condition"] == "ideal-converted"
    )
    print(
        f"Aggregated isolated HIL artifacts at {args.output_dir}; "
        f"float_accuracy={float_accuracy:.4f}, ideal_accuracy={ideal_accuracy:.4f}",
        flush=True,
    )


def isolated_hardware_evaluation_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_sizes = [1, 4] if args.quick else list(args.pool_sizes)
    placements = [args.placements[0]] if args.quick else list(args.placements)
    condition_root = args.output_dir / "condition_workers"
    hagen_averages = (
        {1}
        if args.pooling_domain == "ttfs"
        else {int(pool_size) for pool_size in pool_sizes}
    )
    first_hidden_dirs: dict[int, Path] = {}
    for hagen_avg in sorted(hagen_averages):
        worker_dir = condition_root / f"first_hidden_avg{hagen_avg}"
        _run_condition_subprocess(
            args,
            worker_dir,
            required=("first_hidden.pt", "first_hidden_manifest.json"),
            overrides={
                "prepare_first_hidden": True,
                "condition_hagen_avg": hagen_avg,
                "first_hidden_cache": None,
            },
        )
        first_hidden_dirs[hagen_avg] = worker_dir

    worker_dirs: list[Path] = []
    for placement in placements:
        for pool_size in pool_sizes:
            hagen_avg = 1 if args.pooling_domain == "ttfs" else int(pool_size)
            key = f"{args.pooling_domain}_M{pool_size}_{placement}_{args.pool_mapping}"
            worker_dir = condition_root / key
            _run_condition_subprocess(
                args,
                worker_dir,
                required=("manifest.json", "runtime.json", "intermediates.pt"),
                overrides={
                    "prepare_first_hidden": False,
                    "first_hidden_cache": str(
                        (first_hidden_dirs[hagen_avg] / "first_hidden.pt").resolve()
                    ),
                    "placements": [placement],
                    "pool_sizes": [pool_size],
                },
            )
            worker_dirs.append(worker_dir)
    _aggregate_isolated_conditions(args, worker_dirs, first_hidden_dirs)


def probe_phase(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_bundle(args.task, cache_dir=args.dataset_cache)
    model = _load_float_checkpoint(args)
    converted = _load_or_convert(args, model, dataset.calibration_x)
    hagen = _hagen_backend(args)
    if hagen is None:
        raise ValueError("probe-hagen requires --pwm-backend hagen-mock or hagen-hardware")
    started = perf_counter()
    payload = hagen.probe(converted)
    calibration_input = converted.encode_input(dataset.calibration_x[:128])
    calibration_target = converted.hidden_from_input(
        dataset.calibration_x[:128]
    )[2]
    payload["hidden_shift_calibration"] = hagen.recommend_hidden_shift(
        converted,
        calibration_input,
        calibration_target,
        relu_boundary=args.relu_boundary,
        activation=args.activation,
    )
    payload["elapsed_s"] = perf_counter() - started
    _json_write(args.output_dir / "hagen_probe.json", payload)
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    args = _apply_condition_worker_config(parse_args())
    _validate_architecture(args)
    if args.pool_chunk_worker_config is not None:
        pool_chunk_worker_phase(args)
    elif args.prepare_first_hidden:
        prepare_first_hidden_phase(args)
    elif (
        args.phase == "hardware-eval"
        and args.pool_backend == "hardware"
        and not args.condition_worker
    ):
        isolated_hardware_evaluation_phase(args)
    elif args.phase == "train":
        train_phase(args)
    elif args.phase == "convert":
        convert_phase(args)
    elif args.phase == "probe-hagen":
        probe_phase(args)
    else:
        evaluation_phase(args)


if __name__ == "__main__":
    main()
