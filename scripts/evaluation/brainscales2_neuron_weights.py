#!/usr/bin/env python3
"""Calibrate digital neuron weights on the actual grouped input graph."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.neuron_weights import (
    condition_key, delivery_statistics, passing_neurons,
    select_neuron_weights, weight_context,
)
from utils.hardware.brainscales2.toy_pooling import (
    GroupedHardwarePoolBackend, ToyPoolConfig, _nominal_uint5_times,
)


def make_inputs(pool, spiking, *, seed, trials, mode):
    """Exercise every code per source, with quiet rows before and after."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.arange(32).reshape(-1, 1).repeat(trials, pool.logical_neurons)
    if mode == "mixed":
        for trial in range(trials):
            for logical in range(pool.logical_neurons):
                codes[trial * 32 : (trial + 1) * 32, logical] = torch.randperm(
                    32, generator=generator
                )
    times = _nominal_uint5_times(codes, spiking)
    nominal = torch.full((times.shape[0] + 4, pool.logical_neurons), torch.nan, dtype=torch.float64)
    nominal[2:-2] = times
    if mode == "isolated":
        # One source group at a time, keeping the physical graph identical.
        times = times.reshape(trials, 32, pool.logical_neurons)[:, [0, 15, 31]].reshape(-1, pool.logical_neurons)
        nominal = torch.full((3 * trials * pool.logical_neurons + 4, pool.logical_neurons), torch.nan, dtype=torch.float64)
        for logical in range(pool.logical_neurons):
            start = 2 + logical * 3 * trials
            nominal[start : start + 3 * trials, logical] = times[:, logical]
    inputs = torch.zeros((spiking.runtime_steps, nominal.shape[0], pool.logical_neurons * spiking.input_fan_in))
    for row, logical in torch.isfinite(nominal).nonzero().tolist():
        step = round(float(nominal[row, logical]) / spiking.dt_s)
        start = logical * spiking.input_fan_in
        inputs[step, row, start : start + spiking.input_fan_in] = 1
    return inputs, nominal.repeat_interleave(pool.pool_size, dim=1)


def worker(path):
    request = json.loads(path.read_text())
    pool = ToyPoolConfig(**request["pool"])
    values = request["spiking"]
    values["calibration_path"] = Path(values["calibration_path"])
    spiking = BrainScaleS2PoolConfig(**values)
    inputs, nominal = make_inputs(pool, spiking, seed=request["seed"], trials=request["trials"], mode=request["mode"])
    weights = torch.tensor(request["weights"], dtype=torch.float32)
    print(f"Measuring {request['mode']} inputs, {inputs.shape[1]} batches", flush=True)
    first, count, coordinates, metadata = GroupedHardwarePoolBackend()._run_inputs(
        inputs, pool, spiking, neuron_synaptic_weights=weights,
    )
    torch.save({
        "first_spike_s": first, "spike_count": count, "nominal_input_s": nominal,
        "physical_coordinates": coordinates, "metadata": metadata,
        "statistics": delivery_statistics(first, count, nominal, deadline_s=spiking.observation_deadline_s),
    }, path.with_suffix(".pt"))


def measure(directory, name, pool, spiking, weights, *, seed, trials, mode, timeout):
    request = directory / f"{name}.json"
    if request.exists():
        raise FileExistsError(f"Use a new output directory: {request}")
    request.write_text(json.dumps({
        "pool": asdict(pool), "spiking": spiking.to_manifest_dict(),
        "weights": weights.tolist(), "seed": seed, "trials": trials, "mode": mode,
    }, indent=2))
    with (directory / f"{name}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f"Measurement failed; inspect {directory / (name + '.log')}")
    result = torch.load(request.with_suffix(".pt"), map_location="cpu", weights_only=False)
    stats = result["statistics"]
    print(f"{name}: single={float(stats['single_spike_rate'].mean()):.3f}, "
          f"multi={float(stats['multi_spike_rate'].mean()):.3f}, "
          f"passing={int(passing_neurons(stats).sum())}/{weights.numel()}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--logical-neurons", type=int, default=30)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--placements", nargs="+", choices=["local-pool", "cross-quadrant"], default=["local-pool", "cross-quadrant"])
    parser.add_argument("--input-fan-in", type=int, default=4)
    parser.add_argument("--weights", type=int, nargs="+", default=[32, 36, 40, 44, 48, 52, 56, 60, 63])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--validation-trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-timeout-s", type=float, default=180)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    if args.calibration is None or args.output_dir is None:
        parser.error("--calibration and --output-dir are required")
    if args.trials < 2 or args.validation_trials < 2 or any(w < 0 or w > 63 for w in args.weights):
        parser.error("use at least two trials and digital weights in [0, 63]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spiking = BrainScaleS2PoolConfig(calibration_path=args.calibration.resolve(), input_fan_in=args.input_fan_in)
    spiking.require_reproducible_calibration()
    report = {
        "schema_version": 1, "context": weight_context(spiking),
        "validation_deadline_s": spiking.observation_deadline_s,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": args.seed, "validation_seed": args.seed + 100000,
        "calibration_trials": args.trials, "validation_trials": args.validation_trials,
        "conditions": {}, "chip_identifier": None,
    }
    for size in args.pool_sizes:
        for placement in args.placements:
            pool = ToyPoolConfig(pool_size=size, logical_neurons=args.logical_neurons, placement=placement)
            directory = args.output_dir / condition_key(pool)
            directory.mkdir(exist_ok=True)
            def acquire(name, weights, mode="mixed", validation=False):
                result = measure(directory, name, pool, spiking, weights,
                                 seed=args.seed + (100000 if validation else 0),
                                 trials=args.validation_trials if validation else args.trials,
                                 mode=mode, timeout=args.worker_timeout_s)
                chip = result["metadata"].get("chip_identifier")
                if not chip or (report["chip_identifier"] is not None and chip != report["chip_identifier"]):
                    raise RuntimeError("missing or changed chip identifier")
                report["chip_identifier"] = chip
                return result
            count = args.logical_neurons * size
            statistics = []
            candidates = sorted(set(args.weights))
            for weight in candidates:
                result = acquire(f"weight_{weight}", torch.full((count,), weight))
                statistics.append(result["statistics"])
            selected, calibration_passed = select_neuron_weights(candidates, statistics)
            validation = acquire("validation", selected, validation=True)
            uniform = acquire("validation_uniform", selected, mode="uniform", validation=True)
            validation_passed = passing_neurons(validation["statistics"]) & passing_neurons(uniform["statistics"])
            condition = {
                "physical_coordinates": validation["physical_coordinates"].tolist(),
                "neuron_synaptic_weights": selected.tolist(),
                "calibration_passed": calibration_passed.tolist(),
                "validation_passed": validation_passed.tolist(),
                "validation_statistics": {k: v.tolist() for k, v in validation["statistics"].items()},
                "uniform_statistics": {k: v.tolist() for k, v in uniform["statistics"].items()},
                "viable": bool((calibration_passed & validation_passed).all()),
            }
            report["conditions"][condition_key(pool)] = condition
            (args.output_dir / "neuron_weight_calibration.json").write_text(json.dumps(report, indent=2))
            print(f"{condition_key(pool)} viable={condition['viable']}", flush=True)
    if not all(c["viable"] for c in report["conditions"].values()):
        raise SystemExit("No validated weight calibration for every requested graph; diagnostic results were preserved")


if __name__ == "__main__":
    main()
