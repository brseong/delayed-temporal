#!/usr/bin/env python3
"""Train, convert, and evaluate toy ANN2SNN classifiers with BSS-2 HIL stages."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import argparse
import csv
import json
import os
import platform
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.hagen import HagenConfig, HagenPWMBackend, file_sha256
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
    ToyPoolConfig,
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

    parser.add_argument("--dt-s", type=float, default=1.0e-6)
    parser.add_argument("--input-early-s", type=float, default=5.0e-6)
    parser.add_argument("--input-late-s", type=float, default=25.0e-6)
    parser.add_argument("--deadline-s", type=float, default=60.0e-6)
    parser.add_argument("--inter-batch-wait-s", type=float, default=50.0e-6)
    parser.add_argument("--tau-m-s", type=float, default=20.0e-6)
    parser.add_argument("--tau-syn-s", type=float, default=1.0e-6)
    parser.add_argument("--leak", type=float, default=80.0)
    parser.add_argument("--reset", type=float, default=80.0)
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--refractory-time-s", type=float, default=1.0e-6)
    parser.add_argument("--i-synin-gm", type=float, default=500.0)
    parser.add_argument("--synapse-dac-bias", type=float, default=600.0)
    parser.add_argument("--synaptic-weight", type=float, default=63.0)
    parser.add_argument("--raw-time-scale-s", type=float)
    return parser.parse_args()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): normalize(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(child) for child in value]
        return value

    path.write_text(json.dumps(normalize(payload), indent=2, sort_keys=True), encoding="utf-8")


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
    model = ToyMLP(architecture)
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
        model, history = train_float_model(architecture, dataset, config)
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
                    first = hagen.first_layer(converted, input_uint5, avg=hagen_avg)
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
            pool_result = temporal_backend.run_uint5(
                first_hidden,
                pool_config,
                spiking_config,
            )
            flat_hidden = pool_result.decoded_uint5.reshape(
                -1, converted.architecture.hidden_features
            )
            if hagen is None:
                _, flat_logits = converted.output_from_hidden(flat_hidden)
                output_metadata = {"backend": "torch"}
            else:
                output = hagen.output_layer(converted, flat_hidden)
                flat_logits = output.value
                output_metadata = output.metadata
            logits = flat_logits.reshape(
                pool_result.decoded_uint5.shape[0],
                pool_result.decoded_uint5.shape[1],
                converted.architecture.output_features,
            ).to(torch.float32)
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
                    logits=logits,
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
        "dataset": dataset.metadata,
        "test_samples": test_x.shape[0],
        "pwm_backend": args.pwm_backend,
        "pool_backend": args.pool_backend,
        "pooling_domain": args.pooling_domain,
        "pool_mapping": args.pool_mapping,
        "pool_sizes": pool_sizes,
        "placements": placements,
        "conversion": converted.manifest.to_dict(),
        "float_parameter_sha256": parameter_sha256(model),
        "hagen_calibration_sha256": file_sha256(args.hagen_calibration),
        "spiking_calibration_sha256": file_sha256(args.spiking_calibration),
        "spiking_config": spiking_config.to_manifest_dict(),
        "conditions": [
            {
                "key": item.key,
                "pool_size": item.pool_size,
                "pooling_domain": item.pooling_domain,
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
    )
    payload["elapsed_s"] = perf_counter() - started
    _json_write(args.output_dir / "hagen_probe.json", payload)
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    args = parse_args()
    _validate_architecture(args)
    if args.phase == "train":
        train_phase(args)
    elif args.phase == "convert":
        convert_phase(args)
    elif args.phase == "probe-hagen":
        probe_phase(args)
    else:
        evaluation_phase(args)


if __name__ == "__main__":
    main()
