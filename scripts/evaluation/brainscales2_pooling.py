#!/usr/bin/env python3
"""Run TTFS first-spike neuron-pooling experiments on mock or BrainScaleS-2."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
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

from utils.hardware.brainscales2.analysis import (
    analyze_cadc_diagnostic,
    bootstrap_variance_floor,
    fit_variance_floor,
    score_operating_point,
    summarize_pool_result,
)
from utils.hardware.brainscales2.artifacts import (
    write_cadc_diagnostic_artifacts,
    write_experiment_artifacts,
)
from utils.hardware.brainscales2.backend import (
    BrainScaleS2PoolBackend,
    MockPoolBackend,
    PoolBackend,
    with_operating_point,
)
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig, PoolRunResult
from utils.transforms.noise import get_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds


ESTIMATORS = ("corrected-mean", "mean", "median", "earliest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TTFS first-spike neuron pooling on BrainScaleS-2",
    )
    parser.add_argument(
        "--phase",
        choices=("diagnose-cadc", "calibrate", "run"),
        default="run",
    )
    parser.add_argument("--backend", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--encoding", choices=("identity", "log"), default="identity")
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--placements",
        nargs="+",
        choices=("same-quadrant", "cross-quadrant"),
        default=["same-quadrant", "cross-quadrant"],
    )
    parser.add_argument(
        "--routing",
        nargs="+",
        choices=("broadcast", "independent"),
        default=["broadcast", "independent"],
    )
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--positions", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--theta", type=float, default=400.0)
    parser.add_argument("--log-min", type=float, default=1.0)
    parser.add_argument("--log-max", type=float, default=100.0)
    parser.add_argument("--project-tau-s", type=float, default=1.0)
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

    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--allow-environment-calibration", action="store_true")
    parser.add_argument("--raw-time-scale-s", type=float)
    parser.add_argument("--operating-point-json", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)

    parser.add_argument(
        "--calibration-thresholds",
        type=float,
        nargs="+",
        default=[85, 90, 95, 100, 110, 125],
    )
    parser.add_argument("--calibration-weights", type=float, nargs="+", default=[31, 47, 63])
    parser.add_argument("--calibration-gains", type=float, nargs="+", default=[300, 500, 700])
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> BrainScaleS2PoolConfig:
    pool_sizes = (1, 4) if args.quick else tuple(args.pool_sizes)
    placements = (
        ("same-quadrant",)
        if args.quick
        else tuple(args.placements)
    )
    routings = ("broadcast",) if args.quick else tuple(args.routing)
    trials = 8 if args.quick else args.trials
    config = BrainScaleS2PoolConfig(
        encoding=args.encoding,
        dt_s=args.dt_s,
        input_early_s=args.input_early_s,
        input_late_s=args.input_late_s,
        observation_deadline_s=args.deadline_s,
        inter_batch_wait_s=args.inter_batch_wait_s,
        project_tau_s=args.project_tau_s,
        tau_mem_s=args.tau_m_s,
        tau_syn_s=args.tau_syn_s,
        leak=args.leak,
        reset=args.reset,
        threshold=args.threshold,
        refractory_time_s=args.refractory_time_s,
        i_synin_gm=args.i_synin_gm,
        synapse_dac_bias=args.synapse_dac_bias,
        synaptic_weight=args.synaptic_weight,
        pool_sizes=pool_sizes,
        placements=placements,
        routings=routings,
        trials=trials,
        seed=args.seed,
        calibration_path=args.calibration,
        allow_environment_calibration=args.allow_environment_calibration,
        raw_time_scale_s=args.raw_time_scale_s,
    )
    if args.operating_point_json is not None:
        payload = json.loads(args.operating_point_json.read_text(encoding="utf-8"))
        selected = payload.get("selected", payload)
        config = with_operating_point(
            config,
            threshold=float(selected["threshold"]),
            synaptic_weight=float(selected["synaptic_weight"]),
            i_synin_gm=float(selected["i_synin_gm"]),
        )
    return config


def make_potential(args: argparse.Namespace, config: BrainScaleS2PoolConfig) -> Potential:
    positions = 3 if args.quick else args.positions
    if positions < 2:
        raise ValueError("positions must be at least two")
    if config.encoding == "identity":
        domain = PotentialBounds(-args.theta, args.theta)
        values = torch.linspace(float(domain.min), float(domain.max), positions)
    else:
        if not 0.0 < args.log_min < args.log_max:
            raise ValueError("log encoding requires 0 < log_min < log_max")
        domain = PotentialBounds(args.log_min, args.log_max)
        values = torch.logspace(
            torch.log10(torch.tensor(args.log_min)),
            torch.log10(torch.tensor(args.log_max)),
            positions,
        )
    return Potential(values, domain)


def make_backend(name: str) -> PoolBackend:
    if name == "mock":
        return MockPoolBackend()
    if not BrainScaleS2PoolBackend.dependencies_available():
        raise RuntimeError(
            "hxtorch is unavailable; run hardware mode in the EBRAINS-experimental kernel"
        )
    return BrainScaleS2PoolBackend()


def run_conditions(
    backend: PoolBackend,
    potential: Potential,
    config: BrainScaleS2PoolConfig,
) -> list[PoolRunResult]:
    results: list[PoolRunResult] = []
    for placement in config.placements:
        for routing in config.routings:
            for pool_size in config.pool_sizes:
                print(
                    "Running",
                    f"M={pool_size}",
                    f"placement={placement}",
                    f"routing={routing}",
                    flush=True,
                )
                results.append(
                    backend.run(
                        potential,
                        config,
                        pool_size=pool_size,
                        placement=placement,
                        routing=routing,
                    )
                )
    return results


def analyze_results(
    results: list[PoolRunResult],
    config: BrainScaleS2PoolConfig,
    bootstrap_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = [
        summarize_pool_result(result, estimator)
        for result in results
        for estimator in ESTIMATORS
    ]
    fits: list[dict[str, Any]] = []
    for placement in config.placements:
        for routing in config.routings:
            matching_results = [
                result
                for result in results
                if result.placement == placement and result.routing == routing
            ]
            for estimator in ESTIMATORS:
                matching_summaries = [
                    summary
                    for summary in summaries
                    if summary["placement"] == placement
                    and summary["routing"] == routing
                    and summary["estimator"] == estimator
                ]
                fit = fit_variance_floor(matching_summaries)
                fit.update(
                    {
                        "placement": placement,
                        "routing": routing,
                        "estimator": estimator,
                    }
                )
                if estimator == "corrected-mean":
                    fit.update(
                        bootstrap_variance_floor(
                            matching_results,
                            estimator,
                            iterations=bootstrap_iterations,
                            seed=config.seed,
                        )
                    )
                fits.append(fit)
    return summaries, fits


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "ebrains_release": os.environ.get("EBRAINS_RELEASE"),
    }


def diagnose_cadc(
    args: argparse.Namespace,
    backend: PoolBackend,
    config: BrainScaleS2PoolConfig,
) -> None:
    """Measure a single PSP before attempting sparse first-spike sweeps."""
    if not isinstance(backend, BrainScaleS2PoolBackend):
        raise RuntimeError("diagnose-cadc requires --backend hardware")
    result = backend.diagnose_cadc(config)
    analysis = analyze_cadc_diagnostic(result, config)
    write_cadc_diagnostic_artifacts(
        args.output_dir,
        config=config,
        result=result,
        analysis=analysis,
        extra_manifest={"environment": environment_manifest()},
    )
    report = {
        "viable": analysis["viable"],
        "reason": analysis["reason"],
        "selected": analysis["selected"],
        "aggregate": analysis["aggregate"],
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Wrote CADC diagnostic artifacts to {args.output_dir}")


def calibrate_operating_point(
    args: argparse.Namespace,
    backend: PoolBackend,
    potential: Potential,
    config: BrainScaleS2PoolConfig,
) -> None:
    calibration_config = replace(
        config,
        pool_sizes=(4,),
        placements=("same-quadrant",),
        routings=("broadcast",),
        trials=min(config.trials, 32),
    )
    candidates: list[dict[str, Any]] = []
    for threshold in args.calibration_thresholds:
        for weight in args.calibration_weights:
            for gain in args.calibration_gains:
                candidate_config = with_operating_point(
                    calibration_config,
                    threshold=threshold,
                    synaptic_weight=weight,
                    i_synin_gm=gain,
                )
                result = backend.run(
                    potential,
                    candidate_config,
                    pool_size=4,
                    placement="same-quadrant",
                    routing="broadcast",
                )
                score = score_operating_point(result)
                candidates.append(
                    {
                        "threshold": threshold,
                        "synaptic_weight": weight,
                        "i_synin_gm": gain,
                        **score,
                    }
                )
                print(candidates[-1], flush=True)
    selected = min(candidates, key=lambda candidate: float(candidate["score"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "calibration_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    payload = {
        "selected": selected,
        "config": config.to_manifest_dict(),
        "environment": environment_manifest(),
    }
    (args.output_dir / "selected_operating_point.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("Selected operating point:", selected)


def main() -> None:
    args = parse_args()
    if get_gaussian_time_noise().enabled:
        raise RuntimeError(
            "disable software Gaussian timing noise before a physical pooling run"
        )
    config = make_config(args)
    potential = make_potential(args, config)
    backend = make_backend(args.backend)
    if args.phase == "diagnose-cadc":
        diagnose_cadc(args, backend, config)
        return
    if args.phase == "calibrate":
        calibrate_operating_point(args, backend, potential, config)
        return

    results = run_conditions(backend, potential, config)
    summaries, fits = analyze_results(
        results,
        config,
        bootstrap_iterations=(50 if args.quick else args.bootstrap_iterations),
    )
    write_experiment_artifacts(
        args.output_dir,
        config=config,
        potential=potential,
        results=results,
        summaries=summaries,
        fits=fits,
        extra_manifest={"environment": environment_manifest()},
    )
    print(f"Wrote BrainScaleS-2 pooling artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
