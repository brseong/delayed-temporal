#!/usr/bin/env python3
"""Compare calibrated gains and input counts on sixteen fixed physical neurons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.brainscales2_thresholds import measure, run_job, write_json
from utils.hardware.brainscales2.backend import (
    BrainScaleS2PoolBackend, calibration_sha256, resolve_physical_neuron_indices,
)
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.neuron_weights import delivery_statistics
from utils.hardware.brainscales2.toy_pooling import ToyPoolConfig


def cadc_worker(request: dict, output: Path) -> None:
    values = dict(request["spiking"])
    values["calibration_path"] = Path(values["calibration_path"])
    config = BrainScaleS2PoolConfig(**values)
    if calibration_sha256(config.calibration_path) != request["calibration_sha256"]:
        raise ValueError("calibration checksum changed")
    result = BrainScaleS2PoolBackend().diagnose_cadc(
        config, pool_size=16, placement="cross-quadrant", capture_raw=True,
    )
    if result.metadata["chip_identifier"] != request["chip_identifier"]:
        raise RuntimeError("diagnostic chip differs from calibration chip")
    torch.save({"baseline_cadc": result.baseline_cadc,
                "stimulated_cadc": result.stimulated_cadc,
                "time_s": result.time_s, "stimulus_time_s": result.stimulus_time_s,
                "physical_coordinates": result.physical_coordinates,
                "metadata": result.metadata}, output.with_suffix(".pt"))


def summarize_cadc(data: dict, deadline_s: float) -> list[dict]:
    first = torch.tensor(data["metadata"]["first_spike_s"], dtype=torch.float64)
    count = torch.tensor(data["metadata"]["spike_count"], dtype=torch.long)
    nominal = torch.full_like(first, torch.nan)
    nominal[1::2] = data["stimulus_time_s"]
    stats = delivery_statistics(first, count, nominal, deadline_s=deadline_s)
    stimulus_step = int((data["time_s"] - data["stimulus_time_s"]).abs().argmin())
    pre = torch.arange(data["time_s"].numel()) < stimulus_step
    post = ~pre
    baseline = data["baseline_cadc"]
    reference = baseline[:, pre].median(dim=1).values
    noise = (baseline - reference[:, None]).abs()[:, post].amax(dim=1)
    peak = (data["stimulated_cadc"] - baseline)[:, post].amax(dim=1)
    rows = []
    for neuron, coordinate in enumerate(data["physical_coordinates"]):
        rows.append({
            "physical_coordinate": coordinate,
            "baseline_excursion_q99": float(torch.quantile(noise[:, neuron], .99)),
            "psp_peak_q10": float(torch.quantile(peak[:, neuron], .1)),
            "psp_peak_median": float(torch.quantile(peak[:, neuron], .5)),
            # Reset after a spike changes the trace: retain it, but flag it.
            "trace_contains_spikes": bool((count[:, neuron] > 0).any()),
            **{key: float(value[neuron]) for key, value in stats.items()},
        })
    return rows


# @lat: [[hardware#Toy ANN2SNN Hardware-in-the-Loop#Gain diagnostic]]
def run_diagnostic(base: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    coordinates = resolve_physical_neuron_indices(16, "cross-quadrant")
    manifest = {
        "schema_version": 1, "status": "running", "python": platform.python_version(),
        "torch": torch.__version__, "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_calibration": str(base), "base_sha256": calibration_sha256(base),
        "physical_coordinates": coordinates, "calibrations": {},
        "gains": [500, 700], "thresholds": [125, 100], "input_fan_in": [1, 4],
        "cadc_trials": 32, "delivery_trials_per_code": 8, "quiet_windows": 64,
        "full_experiment_enabled": False,
    }
    started = time.monotonic()
    rows = []
    try:
        write_json(output / "manifest.json", manifest)
        reference_chip = json.loads(base.with_suffix(".json").read_text())["chip_identifier"]
        manifest["chip_identifier"] = reference_chip
        for gain in (500, 700):
            # Only the gain-500 base can reuse the existing native calibration.
            request = {"action": "calibrate", "threshold": 125, "i_synin_gm": gain,
                       "base": str(base) if gain == 500 else None}
            if gain == 500:
                request["base_sha256"] = calibration_sha256(base)
            calibrated = run_job(output / "calibrations", f"gain_{gain}_125", request, 2400)
            for threshold in (125, 100):
                candidate = calibrated
                if threshold != 125:
                    candidate = run_job(output / "calibrations", f"gain_{gain}_{threshold}", {
                        "action": "calibrate", "threshold": threshold, "i_synin_gm": gain,
                        "base": str(calibrated.with_suffix(".pkl")),
                        "base_sha256": calibration_sha256(calibrated.with_suffix(".pkl")),
                    }, 600)
                metadata = json.loads(candidate.with_suffix(".json").read_text())
                if metadata["chip_identifier"] != reference_chip:
                    raise RuntimeError("chip changed between gain conditions")
                manifest["calibrations"][candidate.name] = metadata
                write_json(output / "manifest.json", manifest)
                for fan_in in (1, 4):
                    name = f"gain_{gain}_threshold_{threshold}_fanin_{fan_in}"
                    config = BrainScaleS2PoolConfig(
                        calibration_path=candidate.with_suffix(".pbin"), threshold=threshold,
                        i_synin_gm=gain, input_fan_in=fan_in, trials=32, input_early_s=15e-6,
                    )
                    cadc = run_job(output / "cadc", name, {
                        "action": "cadc", "spiking": config.to_manifest_dict(),
                        "chip_identifier": reference_chip,
                        "calibration_sha256": calibration_sha256(config.calibration_path),
                    }, 240)
                    data = torch.load(cadc.with_suffix(".pt"), weights_only=False, map_location="cpu")
                    condition = summarize_cadc(data, config.observation_deadline_s)
                    for row in condition:
                        rows.append({"i_synin_gm": gain, "threshold": threshold,
                                     "input_fan_in": fan_in, **row})
                    with (output / "cadc_summary.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    print(name, "CADC complete; mean single-spike rate:",
                          sum(row["single_spike_rate"] for row in condition) / 16, flush=True)
                    # Repeat all input codes with CADC disabled to check delivery
                    # in the recording mode used for inference, not only CADC mode.
                    measure(output / "delivery", name,
                            ToyPoolConfig(logical_neurons=1, pool_size=16, placement="cross-quadrant"),
                            replace(config, input_early_s=5e-6),
                            torch.tensor(coordinates).reshape(1, 16), reference_chip,
                            trials=8, mode="uniform", seed=42, quiet_windows=64)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
        jobs = [json.loads(path.read_text()) for path in output.rglob("*.done.json")]
        write_json(output / "runtime.json", {
            "elapsed_s": manifest["elapsed_s"], "status": manifest["status"],
            "completed_jobs": len(jobs), "maximum_worker_rss_bytes": max(
                (job.get("maximum_worker_rss_bytes", 0) for job in jobs), default=0),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_diagnostic(args.base_calibration.resolve(), args.output_dir.resolve())
