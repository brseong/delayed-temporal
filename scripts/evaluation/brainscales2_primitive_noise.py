#!/usr/bin/env python3
"""Collect and validate independent BrainScaleS-2 primitive noise artifacts."""

from __future__ import annotations

import argparse
import csv
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
from utils.hardware.brainscales2.primitive_optimization import (
    build_encoder_operating_point_candidates,
    calibration_transfer_gate,
    parse_current_stop_pair,
    parse_precharge_pair,
    score_encoder_operating_point,
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
    coordinates = (
        tuple(args.physical_coordinates)
        if args.physical_coordinates is not None
        else default_primitive_coordinates(args.device_count)
    )
    return PrimitiveNoiseConfig(
        repeats=repeats,
        calibration_repeats=calibration_repeats,
        device_count=args.device_count,
        seed=args.seed,
        input_early_s=args.input_early,
        input_late_s=args.input_late,
        observation_time_s=args.observation_time,
        deadline_s=args.deadline,
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
        physical_coordinates=coordinates,
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _collect_encoder_search_observations(
    args: argparse.Namespace,
    config: PrimitiveNoiseConfig,
) -> tuple[list[PrimitiveObservation], list[PrimitiveValidation], str | None]:
    backend: Any = (
        MockPrimitiveNoiseBackend()
        if args.backend == "mock"
        else PrimitiveHardwareBackend()
    )
    observations: list[PrimitiveObservation] = []
    validations: list[PrimitiveValidation] = []
    quick_codes = args.quick or args.search_quick_codes
    static = backend.collect("phi-np", config, stage="static", quick=quick_codes)
    static_validation = validate_primitive_observation(static, config)
    observations.append(static)
    validations.append(static_validation)
    if not calibration_transfer_gate(
        static,
        static_validation,
        config,
        screening=args.search_quick_codes,
    )["eligible"]:
        return observations, validations, "phi-np/static calibration gate failed"

    dynamic = backend.collect("phi-np", config, stage="dynamic", quick=quick_codes)
    dynamic_validation = validate_primitive_observation(dynamic, config)
    observations.append(dynamic)
    validations.append(dynamic_validation)
    if not calibration_transfer_gate(
        dynamic,
        dynamic_validation,
        config,
        screening=args.search_quick_codes,
    )["eligible"]:
        return observations, validations, "phi-np/dynamic calibration gate failed"

    if args.primitive == "phi-nl":
        nonlinear = backend.collect(
            "phi-nl", config, stage="transfer", quick=quick_codes
        )
        nonlinear_validation = validate_primitive_observation(nonlinear, config)
        observations.append(nonlinear)
        validations.append(nonlinear_validation)
        if not calibration_transfer_gate(
            nonlinear,
            nonlinear_validation,
            config,
            screening=args.search_quick_codes,
        )["eligible"]:
            return observations, validations, "phi-nl calibration gate failed"
    return observations, validations, None


def _search_result_row(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload["candidate"]
    score = payload.get("score") or {}
    primitive_scores = score.get("primitive_scores", {})
    row: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "status": payload["status"],
        "constant_current_code": candidate["constant_current_code"],
        "threshold_code": candidate["threshold_code"],
        "ramp_stop_s": candidate["ramp_stop_s"],
        "precharge_input_fan_in": candidate["precharge_input_fan_in"],
        "precharge_weight_maximum": candidate["precharge_weight_maximum"],
        "selection_eligible": score.get("selection_eligible"),
        "held_out_validated": score.get("held_out_validated"),
        "selection_objective_rt": score.get("selection_objective_rt"),
        "validation_objective_rt": score.get("validation_objective_rt"),
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
    }
    for primitive in ("phi-np", "phi-nl"):
        primitive_score = primitive_scores.get(primitive, {})
        row[f"{primitive}_calibration_rt"] = (
            primitive_score.get("calibration", {})
            .get("summary", {})
            .get("median")
        )
        row[f"{primitive}_held_out_rt"] = (
            primitive_score.get("held_out", {})
            .get("summary", {})
            .get("median")
        )
    for target, reached in score.get("targets", {}).items():
        row[target] = reached
    return row


def _write_search_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def optimize_encoder_operating_point(
    args: argparse.Namespace,
    base_config: PrimitiveNoiseConfig,
) -> None:
    if args.primitive not in ("phi-np", "phi-nl"):
        raise ValueError("operating-point search requires phi-np or phi-nl")
    if args.search_top_k <= 0 or args.search_max_candidates <= 0:
        raise ValueError("search limits must be positive")
    precharge_pairs = tuple(
        parse_precharge_pair(value) for value in args.search_precharge_pairs
    )
    current_stop_pairs = (
        tuple(
            parse_current_stop_pair(value)
            for value in args.search_current_stop_pairs
        )
        if args.search_current_stop_pairs is not None
        else None
    )
    candidates = build_encoder_operating_point_candidates(
        base_config,
        constant_current_codes=args.search_constant_current_codes,
        threshold_codes=args.search_threshold_codes,
        ramp_stop_times_s=(value * 1.0e-6 for value in args.search_ramp_stop_us),
        precharge_pairs=precharge_pairs,
        current_stop_pairs=current_stop_pairs,
    )
    if len(candidates) > args.search_max_candidates:
        raise ValueError(
            f"search grid has {len(candidates)} candidates; "
            f"limit is {args.search_max_candidates}"
        )
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "search_manifest.json"
    manifest = {
        "schema_version": 1,
        "primitive": args.primitive,
        "base_config": base_config.to_manifest_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selection_contract": {
            "split": "calibration repetitions only",
            "objective": "minimum requested primitive median conditional timing ratio",
            "signal_span": "frozen phi-np calibration transfer span",
            "held_out_role": "confirmation only",
            "code_grid": "representative" if args.search_quick_codes else "full",
        },
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != _json_safe(manifest):
            raise RuntimeError("existing search manifest does not match this grid")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("search output directory is nonempty without a manifest")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_dir = output_dir / "candidates" / candidate.candidate_id
        result_path = candidate_dir / "candidate_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("candidate") != candidate.to_dict():
                raise RuntimeError(
                    f"candidate identity mismatch in {result_path}"
                )
            if result.get("status") == "complete":
                print(
                    f"Reusing candidate {index}/{len(candidates)} "
                    f"{candidate.candidate_id}",
                    flush=True,
                )
                results.append(result)
                continue
            print(
                f"Retrying incomplete candidate {index}/{len(candidates)} "
                f"{candidate.candidate_id}",
                flush=True,
            )

        print(
            f"Collecting candidate {index}/{len(candidates)} "
            f"{candidate.candidate_id}",
            flush=True,
        )
        candidate_config = candidate.apply(base_config)
        result: dict[str, Any] = {
            "schema_version": 1,
            "candidate": candidate.to_dict(),
            "config": candidate_config.to_manifest_dict(),
            "status": "failed",
        }
        try:
            observations, validations, failure = (
                _collect_encoder_search_observations(args, candidate_config)
            )
            write_primitive_noise_artifacts(
                candidate_dir,
                config=candidate_config,
                observations=observations,
                validations=validations,
                environment=environment_manifest("optimize", args.backend),
            )
            if failure is not None:
                raise RuntimeError(failure)
            result["score"] = score_encoder_operating_point(
                observations,
                validations,
                candidate_config,
                primitive=args.primitive,
                screening=args.search_quick_codes,
            )
            result["status"] = "complete"
        except Exception as error:
            result["error_type"] = type(error).__name__
            result["error"] = str(error)
        _write_json(result_path, result)
        results.append(result)
        _write_search_csv(
            output_dir / "operating_point_results.csv",
            [_search_result_row(item) for item in results],
        )

    eligible = [
        result
        for result in results
        if result.get("status") == "complete"
        and result.get("score", {}).get("selection_eligible")
        and result.get("score", {}).get("selection_objective_rt") is not None
    ]
    if not eligible:
        raise RuntimeError("no operating-point candidate passed calibration gates")
    ranked = sorted(
        eligible,
        key=lambda result: (
            result["score"]["selection_objective_rt"],
            result["candidate"]["candidate_id"],
        ),
    )
    best_by_primitive: dict[str, Any] = {}
    for primitive in ("phi-np", "phi-nl"):
        primitive_ranked: list[tuple[float, dict[str, Any]]] = []
        for result in results:
            score = result.get("score", {})
            primitive_score = score.get("primitive_scores", {}).get(primitive)
            if result.get("status") != "complete" or not primitive_score:
                continue
            calibration_rt = (
                primitive_score.get("calibration", {})
                .get("summary", {})
                .get("median")
            )
            if (
                calibration_rt is None
                or not primitive_score.get("calibration_transfer", {}).get(
                    "eligible"
                )
                or not score.get("np_static", {})
                .get("calibration_transfer", {})
                .get("eligible")
            ):
                continue
            primitive_ranked.append((float(calibration_rt), result))
        if not primitive_ranked:
            continue
        primitive_ranked.sort(
            key=lambda item: (item[0], item[1]["candidate"]["candidate_id"])
        )
        calibration_rt, result = primitive_ranked[0]
        primitive_score = result["score"]["primitive_scores"][primitive]
        held_out_rt = primitive_score["held_out"]["summary"]["median"]
        held_out_validated = bool(primitive_score["held_out_validated"])
        best_by_primitive[primitive] = {
            "candidate": result["candidate"],
            "calibration_rt": calibration_rt,
            "held_out_rt": held_out_rt,
            "held_out_validated": held_out_validated,
            "targets": {
                "hardware_feasibility_1e-3": (
                    held_out_validated
                    and held_out_rt is not None
                    and held_out_rt <= 1.0e-3
                ),
                "meaningful_recovery_1e-4": (
                    held_out_validated
                    and held_out_rt is not None
                    and held_out_rt <= 1.0e-4
                ),
                "near_clean_recovery_3e-5": (
                    held_out_validated
                    and held_out_rt is not None
                    and held_out_rt <= 3.0e-5
                ),
            },
        }
    selection = {
        "schema_version": 1,
        "primitive": args.primitive,
        "selected": ranked[0],
        "top_candidates": ranked[: args.search_top_k],
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "best_by_primitive": best_by_primitive,
    }
    _write_json(output_dir / "selected_operating_point.json", selection)
    _write_search_csv(
        output_dir / "operating_point_results.csv",
        [_search_result_row(item) for item in results],
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_count": len(candidates),
                "eligible_count": len(eligible),
                "selected_candidate": ranked[0]["candidate"],
                "selection_objective_rt": ranked[0]["score"][
                    "selection_objective_rt"
                ],
                "validation_objective_rt": ranked[0]["score"][
                    "validation_objective_rt"
                ],
                "targets": ranked[0]["score"]["targets"],
                "best_by_primitive": best_by_primitive,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    config = make_config(args)
    environment = environment_manifest(args.phase, args.backend)

    if args.phase == "optimize":
        optimize_encoder_operating_point(args, config)
        return

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
        "--phase",
        choices=("probe", "collect", "validate", "optimize", "all"),
        default="all",
    )
    parser.add_argument("--primitive", choices=(*PRIMITIVES, "all"), default="all")
    parser.add_argument("--backend", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--calibration-repeats", type=int, default=128)
    parser.add_argument("--device-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-early", type=float, default=5.0e-6)
    parser.add_argument("--input-late", type=float, default=25.0e-6)
    parser.add_argument("--observation-time", type=float, default=40.0e-6)
    parser.add_argument("--deadline", type=float, default=60.0e-6)
    parser.add_argument("--physical-coordinates", type=int, nargs="+")
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
    parser.add_argument("--psi-ne-chunk-repeats", type=int, default=4)
    parser.add_argument("--psi-ed-delta-min", type=float, default=-10.0e-6)
    parser.add_argument("--psi-ed-delta-max", type=float, default=10.0e-6)
    parser.add_argument("--psi-ed-delta-points", type=int, default=11)
    parser.add_argument("--psi-ed-separation-center", type=float, default=15.0e-6)
    parser.add_argument("--psi-ed-post-time", type=float, default=30.0e-6)
    parser.add_argument("--psi-ed-readout-time", type=float, default=58.0e-6)
    parser.add_argument("--psi-ed-trial-guard", type=float, default=1.0e-3)
    parser.add_argument("--psi-ed-trigger-fan-in", type=int, default=8)
    parser.add_argument("--psi-ed-trigger-weight", type=int, default=63)
    parser.add_argument("--psi-ed-plastic-weight", type=int, default=0)
    parser.add_argument("--psi-ed-chunk-repeats", type=int, default=32)
    parser.add_argument("--tau-mem", type=float, default=100.0e-6)
    parser.add_argument("--tau-syn", type=float, default=1.0e-6)
    parser.add_argument("--hagen-wait-between-events", type=int, default=5)
    parser.add_argument("--hagen-num-sends", type=int, default=1024)
    parser.add_argument("--hagen-candidate-count", type=int, default=128)
    parser.add_argument("--hagen-chunk-repeats", type=int, default=16)
    parser.add_argument(
        "--search-constant-current-codes",
        type=int,
        nargs="+",
        default=(256, 384, 512, 768, 1022),
    )
    parser.add_argument(
        "--search-threshold-codes", type=int, nargs="+", default=(600,)
    )
    parser.add_argument(
        "--search-ramp-stop-us", type=float, nargs="+", default=(25.0, 40.0, 55.0)
    )
    parser.add_argument(
        "--search-precharge-pairs", nargs="+", default=("1:63",)
    )
    parser.add_argument("--search-current-stop-pairs", nargs="+")
    parser.add_argument("--search-quick-codes", action="store_true")
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--search-max-candidates", type=int, default=64)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
