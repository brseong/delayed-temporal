#!/usr/bin/env python3
"""Collect and validate independent BrainScaleS-2 primitive noise artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_artifacts import (
    load_primitive_observations,
    write_primitive_noise_artifacts,
)
from utils.hardware.brainscales2.primitive_backend import (
    PrimitiveHardwareBackend,
    probe_primitive_capabilities,
)
from utils.hardware.brainscales2.primitive_noise import (
    PRIMITIVES,
    MockPrimitiveNoiseBackend,
    PrimitiveKind,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    PrimitiveValidation,
    default_primitive_coordinates,
    validate_primitive_observation,
)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(phase: str, backend: str) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "backend": backend,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
    }


def selected_primitives(value: str) -> tuple[PrimitiveKind, ...]:
    if value == "all":
        return PRIMITIVES
    return (value,)  # type: ignore[return-value]


def make_config(args: argparse.Namespace) -> PrimitiveNoiseConfig:
    repeats = 8 if args.quick else args.repeats
    calibration_repeats = 4 if args.quick else args.calibration_repeats
    return PrimitiveNoiseConfig(
        repeats=repeats,
        calibration_repeats=calibration_repeats,
        device_count=args.device_count,
        seed=args.seed,
        observation_time_s=args.observation_time,
        spiking_calibration_path=(
            args.spiking_calibration.resolve()
            if args.spiking_calibration is not None
            else None
        ),
        hagen_calibration_path=(
            args.hagen_calibration.resolve()
            if args.hagen_calibration is not None
            else None
        ),
        correlation_calibration_path=(
            args.correlation_calibration.resolve()
            if args.correlation_calibration is not None
            else None
        ),
        allow_environment_calibration=args.allow_environment_calibration,
        physical_coordinates=default_primitive_coordinates(args.device_count),
        multiple_spike_rate_limit=args.multiple_spike_rate_limit,
        deadline_miss_rate_limit=args.deadline_miss_rate_limit,
        reset_code_minimum=args.reset_code_minimum,
        reset_code_maximum=args.reset_code_maximum,
        reset_code_table=parse_reset_code_table(
            args.reset_code_table, device_count=args.device_count
        ),
        threshold_code=args.threshold_code,
        leak_bias=args.leak_bias,
        constant_current_code=args.constant_current_code,
        precharge_weight_maximum=args.precharge_weight_maximum,
        precharge_input_fan_in=args.precharge_input_fan_in,
        exponential_input_weight=args.exponential_input_weight,
        exponential_input_fan_in=args.exponential_input_fan_in,
        phi_nl_lower_bound_minimum_code=args.phi_nl_lower_bound_minimum_code,
        phi_nl_lower_bound_grid_size=args.phi_nl_lower_bound_grid_size,
        pynn_chunk_repeats=args.pynn_chunk_repeats,
        pynn_process_repeats=args.pynn_process_repeats,
        pynn_worker_timeout_s=args.pynn_worker_timeout,
        pynn_worker_cache_dir=(
            args.pynn_worker_cache_dir.resolve()
            if args.pynn_worker_cache_dir is not None
            else None
        ),
        psi_ne_input_fan_in=args.psi_ne_input_fan_in,
        psi_ne_chunk_repeats=args.psi_ne_chunk_repeats,
        psi_ed_delta_min_s=args.psi_ed_delta_min,
        psi_ed_delta_max_s=args.psi_ed_delta_max,
        psi_ed_delta_points=args.psi_ed_delta_points,
        psi_ed_separation_center_s=args.psi_ed_separation_center,
        psi_ed_post_time_s=args.psi_ed_post_time,
        psi_ed_readout_time_s=args.psi_ed_readout_time,
        psi_ed_trial_guard_s=args.psi_ed_trial_guard,
        psi_ed_trigger_fan_in=args.psi_ed_trigger_fan_in,
        psi_ed_trigger_weight=args.psi_ed_trigger_weight,
        psi_ed_plastic_weight=args.psi_ed_plastic_weight,
        psi_ed_chunk_repeats=args.psi_ed_chunk_repeats,
        tau_mem_s=args.tau_mem,
        tau_syn_s=args.tau_syn,
        hagen_wait_between_events=args.hagen_wait_between_events,
        hagen_num_sends=args.hagen_num_sends,
        hagen_candidate_count=args.hagen_candidate_count,
        hagen_chunk_repeats=args.hagen_chunk_repeats,
    )


def parse_reset_code_table(
    values: list[int] | None, *, device_count: int
) -> tuple[int, ...] | tuple[tuple[int, ...], ...] | None:
    """Parse a shared or code-major neuron-level reset lookup."""
    if values is None:
        return None
    if len(values) == 32:
        return tuple(values)
    if len(values) == 32 * device_count:
        return tuple(
            tuple(values[offset : offset + device_count])
            for offset in range(0, len(values), device_count)
        )
    raise ValueError(
        "reset code table requires 32 shared entries or "
        f"{32 * device_count} code-major neuron entries"
    )


def validate_observations(
    observations: list[PrimitiveObservation],
    config: PrimitiveNoiseConfig,
) -> list[PrimitiveValidation]:
    return [
        validate_primitive_observation(observation, config)
        for observation in observations
    ]


def collect_observations(
    args: argparse.Namespace,
    config: PrimitiveNoiseConfig,
) -> tuple[list[PrimitiveObservation], list[PrimitiveValidation]]:
    backend: Any
    if args.backend == "mock":
        backend = MockPrimitiveNoiseBackend()
    else:
        backend = PrimitiveHardwareBackend()
    requested = set(selected_primitives(args.primitive))
    observations: list[PrimitiveObservation] = []
    validations: list[PrimitiveValidation] = []

    # phi-nl is allowed to run only after the same acquisition has verified the
    # static and dynamic phi-np precharge path.
    needs_np = bool(requested & {"phi-np", "phi-nl"})
    np_dynamic_valid = False
    if needs_np:
        print("Collecting phi-np/static", flush=True)
        static = backend.collect("phi-np", config, stage="static", quick=args.quick)
        observations.append(static)
        static_validation = validate_primitive_observation(static, config)
        validations.append(static_validation)
        if static_validation.validated:
            print("Collecting phi-np/dynamic", flush=True)
            dynamic = backend.collect(
                "phi-np", config, stage="dynamic", quick=args.quick
            )
            observations.append(dynamic)
            dynamic_validation = validate_primitive_observation(dynamic, config)
            validations.append(dynamic_validation)
            np_dynamic_valid = dynamic_validation.validated
        else:
            print(
                "phi-np/static failed validation; dynamic stage is intentionally skipped",
                flush=True,
            )

    if "phi-nl" in requested:
        if np_dynamic_valid:
            print("Collecting phi-nl/transfer", flush=True)
            observation = backend.collect(
                "phi-nl", config, stage="transfer", quick=args.quick
            )
            observations.append(observation)
            validations.append(validate_primitive_observation(observation, config))
        else:
            print(
                "phi-nl skipped because the dynamic precharge prerequisite failed",
                flush=True,
            )

    for primitive in ("psi-int", "psi-ne", "psi-ed"):
        if primitive not in requested:
            continue
        print(f"Collecting {primitive}/transfer", flush=True)
        observation = backend.collect(
            primitive, config, stage="transfer", quick=args.quick
        )
        observations.append(observation)
        validations.append(validate_primitive_observation(observation, config))
    return observations, validations


def _write_probe(output_dir: Path, capabilities: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe.json").write_text(
        json.dumps(capabilities, indent=2, sort_keys=True), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    config = make_config(args)
    environment = environment_manifest(args.phase, args.backend)

    if args.phase in ("probe", "all"):
        capabilities = probe_primitive_capabilities()
        _write_probe(output_dir, capabilities)
        print(json.dumps(capabilities, indent=2, sort_keys=True), flush=True)
        if args.phase == "probe":
            return

    if args.phase == "validate":
        observations = load_primitive_observations(output_dir, config)
        validations = validate_observations(observations, config)
    else:
        observations, validations = collect_observations(args, config)

    calibration = write_primitive_noise_artifacts(
        output_dir,
        config=config,
        observations=observations,
        validations=validations,
        environment=environment,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "observations": len(observations),
                "all_five_validated": calibration["validated"],
                "primitive_status": {
                    primitive: calibration["primitives"][primitive]["validated"]
                    for primitive in PRIMITIVES
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("probe", "collect", "validate", "all"), default="all"
    )
    parser.add_argument("--primitive", choices=(*PRIMITIVES, "all"), default="all")
    parser.add_argument("--backend", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--calibration-repeats", type=int, default=128)
    parser.add_argument("--device-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-time", type=float, default=40.0e-6)
    parser.add_argument("--spiking-calibration", type=Path)
    parser.add_argument("--hagen-calibration", type=Path)
    parser.add_argument("--correlation-calibration", type=Path)
    parser.add_argument("--allow-environment-calibration", action="store_true")
    parser.add_argument("--multiple-spike-rate-limit", type=float, default=0.001)
    parser.add_argument("--deadline-miss-rate-limit", type=float, default=0.01)
    parser.add_argument("--reset-code-minimum", type=int, default=300)
    parser.add_argument("--reset-code-maximum", type=int, default=900)
    parser.add_argument("--reset-code-table", type=int, nargs="+")
    parser.add_argument("--threshold-code", type=int, default=600)
    parser.add_argument("--leak-bias", type=int, default=0)
    parser.add_argument("--constant-current-code", type=int, default=1022)
    parser.add_argument("--precharge-weight-maximum", type=int, default=63)
    parser.add_argument("--precharge-input-fan-in", type=int, default=1)
    parser.add_argument("--exponential-input-weight", type=int, default=63)
    parser.add_argument("--exponential-input-fan-in", type=int, default=8)
    parser.add_argument("--phi-nl-lower-bound-minimum-code", type=float, default=-128.0)
    parser.add_argument("--phi-nl-lower-bound-grid-size", type=int, default=2048)
    parser.add_argument("--pynn-chunk-repeats", type=int, default=8)
    parser.add_argument("--pynn-process-repeats", type=int, default=32)
    parser.add_argument("--pynn-worker-timeout", type=float, default=300.0)
    parser.add_argument("--pynn-worker-cache-dir", type=Path)
    parser.add_argument("--psi-ne-input-fan-in", type=int, default=2)
    parser.add_argument("--psi-ne-chunk-repeats", type=int, default=16)
    parser.add_argument("--psi-ed-delta-min", type=float, default=-10.0e-6)
    parser.add_argument("--psi-ed-delta-max", type=float, default=10.0e-6)
    parser.add_argument("--psi-ed-delta-points", type=int, default=11)
    parser.add_argument("--psi-ed-separation-center", type=float, default=15.0e-6)
    parser.add_argument("--psi-ed-post-time", type=float, default=30.0e-6)
    parser.add_argument("--psi-ed-readout-time", type=float, default=58.0e-6)
    parser.add_argument("--psi-ed-trial-guard", type=float, default=1.0e-3)
    parser.add_argument("--psi-ed-trigger-fan-in", type=int, default=8)
    parser.add_argument("--psi-ed-trigger-weight", type=int, default=63)
    parser.add_argument("--psi-ed-plastic-weight", type=int, default=63)
    parser.add_argument("--psi-ed-chunk-repeats", type=int, default=32)
    parser.add_argument("--tau-mem", type=float, default=100.0e-6)
    parser.add_argument("--tau-syn", type=float, default=1.0e-6)
    parser.add_argument("--hagen-wait-between-events", type=int, default=5)
    parser.add_argument("--hagen-num-sends", type=int, default=1024)
    parser.add_argument("--hagen-candidate-count", type=int, default=128)
    parser.add_argument("--hagen-chunk-repeats", type=int, default=16)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
