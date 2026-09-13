#!/usr/bin/env python3
"""Calibrate physical thresholds, validate delivery, and freeze placement."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.backend import calibration_sha256
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.neuron_weights import delivery_statistics, weight_context
from utils.hardware.brainscales2.thresholds import (
    allocate_validated_coordinates, passing_threshold_neurons, refine_result,
)
from utils.hardware.brainscales2.toy_pooling import (
    GroupedHardwarePoolBackend, ToyPoolConfig, _nominal_uint5_times,
)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def calibrate_worker(request, output):
    """Use a managed connection so Calix and hxtorch never own it together."""
    import calix
    from calix.common import base, cadc
    from calix.spiking import SpikingCalibTarget
    from calix.spiking.neuron import NeuronCalibTarget, refine_potentials
    from dlens_vx_v3 import hxcomm, sta, halco
    import quantities as pq

    gain = float(request.get("i_synin_gm", 500))
    if not 0 < gain <= 1022:
        raise ValueError("invalid calibration gain")
    target = NeuronCalibTarget(leak=80, reset=80, threshold=request["threshold"],
                              tau_mem=20 * pq.us, tau_syn=1 * pq.us,
                              i_synin_gm=gain, membrane_capacitance=63,
                              refractory_time=1 * pq.us, synapse_dac_bias=600)
    print(f"Calix physical threshold target {request['threshold']}", flush=True)
    with hxcomm.ManagedConnection() as connection:
        chip = [str(value) for value in connection.get_unique_identifier()]
        state = base.StatefulConnection(connection)
        if request.get("base") is None:
            result = calix.calibrate(SpikingCalibTarget(neuron_target=target),
                                     cache_paths=[], connection=state)
        else:
            parent = Path(request["base"])
            metadata = json.loads(parent.with_suffix(".json").read_text())
            if metadata["chip_identifier"] != chip:
                raise RuntimeError("chip changed since base calibration")
            if calibration_sha256(parent) != request["base_sha256"]:
                raise ValueError("base calibration checksum changed")
            with parent.open("rb") as handle:
                result = pickle.load(handle)
            validate_refinement_gain(result.target.neuron_target.i_synin_gm, gain)
            builder = base.WriteRecordingPlaybackProgramBuilder()
            result.apply(builder)
            base.run(state, builder)
        # Neuron calibration can affect the CADC; calibrate it in that state,
        # then refine potentials last, as prescribed by Calix.
        result.cadc_result = cadc.calibrate(state)
        result = refine_result(result, target, state, refine_potentials)
        dumper = sta.PlaybackProgramBuilderDumper()
        result.apply(dumper)
        binary = sta.to_portablebinary(dumper.done())
        output.with_suffix(".pbin").write_bytes(binary)
        with output.with_suffix(".pkl").open("wb") as handle:
            pickle.dump(result, handle)
        physical = []
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            neuron = result.neuron_result.neurons[coord]
            physical.append({"physical_coordinate": int(coord.toEnum()),
                             "threshold": int(neuron.threshold.v_threshold),
                             "leak": int(neuron.leak.v_leak), "reset": int(neuron.reset.v_reset)})
        write_json(output.with_suffix(".json"), {
            "chip_identifier": chip, "threshold": request["threshold"],
            "i_synin_gm": gain,
            "calibration_sha256": calibration_sha256(output.with_suffix(".pbin")),
            "native_sha256": calibration_sha256(output.with_suffix(".pkl")),
            "targets": str(result.target), "physical_parameters": physical,
            "initial_success": str(getattr(result.neuron_result, "success", None)),
            "refinement_requires_delivery_validation": True,
        })
        del state


def validate_refinement_gain(previous_gain, requested_gain):
    if not bool((torch.as_tensor(previous_gain) == requested_gain).all()):
        raise ValueError("gain change requires full calibration, not potential refinement")


def code_schedule(logical, trials, mode, seed, quiet_windows):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.arange(32).reshape(-1, 1).repeat(trials, logical)
    if mode == "mixed":
        for trial in range(trials):
            for neuron in range(logical):
                codes[32 * trial:32 * (trial + 1), neuron] = torch.randperm(32, generator=generator)
    elif mode == "isolated":
        individual = torch.full((codes.shape[0] * logical, logical), -1, dtype=torch.long)
        for neuron in range(logical):
            individual[neuron * codes.shape[0]:(neuron + 1) * codes.shape[0], neuron] = codes[:, neuron]
        codes = individual
    elif mode != "uniform":
        raise ValueError("unknown input mode")
    return torch.cat((torch.full((quiet_windows, logical), -1, dtype=torch.long), codes))


def measure_worker(request, output):
    values = dict(request["spiking"])
    values["calibration_path"] = Path(values["calibration_path"])
    spiking = BrainScaleS2PoolConfig(**values)
    if calibration_sha256(spiking.calibration_path) != request["calibration_sha256"]:
        raise ValueError("analog calibration changed")
    pool = ToyPoolConfig(**request["pool"])
    codes = torch.tensor(request["codes"], dtype=torch.long)
    nominal = _nominal_uint5_times(codes.clamp_min(0), spiking)
    nominal[codes < 0] = torch.nan
    inputs = torch.zeros((spiking.runtime_steps, codes.shape[0], pool.logical_neurons * spiking.input_fan_in))
    for batch, logical in torch.isfinite(nominal).nonzero().tolist():
        index = round(float(nominal[batch, logical]) / spiking.dt_s)
        inputs[index, batch, logical * spiking.input_fan_in:(logical + 1) * spiking.input_fan_in] = 1
    first, count, coordinates, metadata = GroupedHardwarePoolBackend()._run_inputs(
        inputs, pool, spiking, physical_coordinates=torch.tensor(request["coordinates"], dtype=torch.long))
    if metadata.get("chip_identifier") != request["chip_identifier"]:
        raise RuntimeError("measurement chip does not match calibration")
    torch.save({"first_spike_s": first, "spike_count": count,
                "nominal_input_s": nominal.repeat_interleave(pool.pool_size, dim=1),
                "codes": codes, "physical_coordinates": coordinates,
                "metadata": metadata}, output.with_suffix(".pt"))


def run_job(directory, name, request, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".request.json")
    output = directory / name
    request = {**request, "code_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    done = output.with_suffix(".done.json")
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError(f"request changed; choose a new run directory: {path}")
        if done.exists():
            info = json.loads(done.read_text())
            if all(calibration_sha256(directory / filename) == checksum
                   for filename, checksum in info["files"].items()):
                return output
            raise ValueError(f"completed artifact changed: {output}")
    write_json(path, request)
    started = time.monotonic()
    print(f"Starting {name}", flush=True)
    with output.with_suffix(".log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: running {time.monotonic() - started:.0f}s; {output.with_suffix('.log')}", flush=True)
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{name} exceeded {timeout}s")
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f"{name} failed; inspect {output.with_suffix('.log')}")
    suffixes = (".pkl", ".pbin", ".json") if request["action"] == "calibrate" else (".pt",)
    write_json(done, {"elapsed_s": time.monotonic() - started,
                      "maximum_worker_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
                      "files": {output.with_suffix(s).name: calibration_sha256(output.with_suffix(s)) for s in suffixes}})
    return output


def measure(directory, name, pool, spiking, coordinates, chip, *, trials, mode, seed, quiet_windows):
    codes = code_schedule(pool.logical_neurons, trials, mode, seed, quiet_windows)
    chunks = []
    # Keep the largest native recording bounded, including simultaneous inputs.
    batch_size = max(8, min(128, 4096 // coordinates.numel()))
    for start in range(0, len(codes), batch_size):
        output = run_job(directory / name, f"batch_{start:06d}", {
            "action": "measure", "pool": asdict(pool), "spiking": spiking.to_manifest_dict(),
            "calibration_sha256": calibration_sha256(spiking.calibration_path),
            "chip_identifier": chip, "coordinates": coordinates.tolist(),
            "codes": codes[start:start + batch_size].tolist(),
        }, timeout=180)
        chunks.append(torch.load(output.with_suffix(".pt"), weights_only=False, map_location="cpu"))
    first = torch.cat([chunk["first_spike_s"] for chunk in chunks])
    count = torch.cat([chunk["spike_count"] for chunk in chunks])
    nominal = torch.cat([chunk["nominal_input_s"] for chunk in chunks])
    stats = delivery_statistics(first, count, nominal, deadline_s=spiking.observation_deadline_s)
    passed = passing_threshold_neurons(stats)
    # Wilson intervals describe sampling uncertainty, not the acceptance cutoff.
    intervals = {}
    for key, mask in (("single_spike_rate", torch.isfinite(nominal)),
                      ("quiet_fired_rate", ~torch.isfinite(nominal))):
        n = mask.sum(0).double().clamp_min(1)
        p = stats[key].double()
        z = 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * torch.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        intervals[key] = torch.stack((center-half, center+half), dim=-1).tolist()
    write_json(directory / name / "summary.json", {
        "statistics": {key: value.tolist() for key, value in stats.items()},
        "confidence_intervals": intervals, "validation_passed": passed.tolist(),
        "coordinates": coordinates.tolist(), "trials": trials,
        "quiet_windows": quiet_windows, "seed": seed, "mode": mode,
    })
    print(f"{name}: {int(passed.sum())}/{passed.numel()} passed", flush=True)
    return passed


def _run_threshold_experiment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = run_job(output_dir / "calibrations", "base_125",
                   {"action": "calibrate", "threshold": 125, "base": None}, 1800)
    chip = json.loads(base.with_suffix(".json").read_text())["chip_identifier"]
    cache = {}

    def candidate(threshold):
        if threshold in cache:
            return cache[threshold]
        calibration = base if threshold == 125 else run_job(output_dir / "calibrations", f"threshold_{threshold}", {
            "action": "calibrate", "threshold": threshold, "base": str(base.with_suffix('.pkl')),
            "base_sha256": calibration_sha256(base.with_suffix('.pkl')),
        }, 900)
        spiking = BrainScaleS2PoolConfig(calibration_path=calibration.with_suffix(".pbin"), threshold=threshold, input_fan_in=1)
        directory = output_dir / f"threshold_{threshold}"
        good = torch.zeros(512, dtype=torch.bool)
        # One logical source drives 32 measured circuits. This screens individual
        # responses without 30 simultaneous independent logical sources.
        for start in range(0, 512, 32):
            pool = ToyPoolConfig(logical_neurons=1, pool_size=32)
            coordinates = torch.arange(start, start+32).reshape(1, 32)
            good[start:start+32] = measure(directory, f"screen_{start}", pool, spiking, coordinates, chip,
                                            trials=8, mode="uniform", seed=42, quiet_windows=64)
        write_json(directory / "screen.json", {"good": good.tolist(), "excluded": torch.where(~good)[0].tolist()})
        try:
            mapping = allocate_validated_coordinates(good)
        except ValueError as error:
            write_json(directory / "development.json", {"viable": False, "reason": str(error)})
            cache[threshold] = None
            return None
        for placement, values in mapping.items():
            # Test the most heavily populated graph first, then every smaller M.
            for size in (16, 1, 2, 4, 8):
                pool = ToyPoolConfig(logical_neurons=30, pool_size=size, placement=placement)
                coordinates = torch.tensor(values)[:, :size].contiguous()
                for mode in ("mixed", "uniform"):
                    passed = measure(directory, f"development_{placement}_{size}_{mode}", pool, spiking,
                                     coordinates, chip, trials=8, mode=mode, seed=43, quiet_windows=64)
                    if not bool(passed.all()):
                        write_json(directory / "development.json", {"viable": False, "reason": "grouped input validation failed"})
                        cache[threshold] = None
                        return None
        result = {"spiking": spiking, "coordinates": mapping, "excluded": torch.where(~good)[0].tolist()}
        write_json(directory / "development.json", {"viable": True, "coordinates": mapping})
        cache[threshold] = result
        return result

    chosen = None
    threshold = None
    previous = None
    for coarse in range(125, 84, -5):
        found = candidate(coarse)
        if found is not None:
            threshold, chosen = coarse, found
            if previous is not None:
                for fine in range(previous - 1, coarse, -1):
                    found = candidate(fine)
                    if found is not None:
                        threshold, chosen = fine, found
                        break
            break
        previous = coarse
    selection_path = output_dir / "threshold_selection.json"
    if chosen is None:
        write_json(selection_path, {"schema_version": 1, "viable": False, "reason": "no development candidate passed"})
        raise RuntimeError("No single-input threshold passed; full evaluation is blocked")
    spiking = chosen["spiking"]
    report = {"schema_version": 1, "viable": False, "threshold": threshold,
              "context": weight_context(spiking), "calibration_path": str(spiking.calibration_path.resolve()),
              "chip_identifier": chip, "coordinates": chosen["coordinates"],
              "excluded_coordinates": chosen["excluded"], "development_seed": 42,
              "validation_seed": 100042, "validation": {}}
    write_json(selection_path, report)
    # Selection is frozen before held-out observations. Failure stops rather
    # than selecting another candidate against this same validation split.
    for placement, values in chosen["coordinates"].items():
        for size in (1, 2, 4, 8, 16):
            pool = ToyPoolConfig(logical_neurons=30, pool_size=size, placement=placement)
            for mode in ("mixed", "uniform"):
                name = f"{placement}_{size}_{mode}"
                passed = measure(output_dir / "validation", name, pool, spiking,
                                 torch.tensor(values)[:, :size].contiguous(), chip,
                                 trials=64, mode=mode, seed=100042, quiet_windows=512)
                report["validation"][name] = bool(passed.all())
                write_json(selection_path, report)
                if not bool(passed.all()):
                    raise RuntimeError("Independent threshold validation failed; full evaluation is blocked")
    for placement, values in chosen["coordinates"].items():
        pool = ToyPoolConfig(logical_neurons=30, pool_size=1, placement=placement)
        coordinates = torch.tensor(values)[:, :1].contiguous()
        for analog, target in ((base.with_suffix(".pbin"), 125), (spiking.calibration_path, threshold)):
            for fan_in in (1, 4):
                control = replace(spiking, calibration_path=analog, threshold=target, input_fan_in=fan_in)
                for mode in ("isolated", "mixed", "uniform"):
                    measure(output_dir / "controls", f"{placement}_{target}_{fan_in}_{mode}", pool,
                            control, coordinates, chip, trials=8, mode=mode, seed=200042, quiet_windows=64)
    report["viable"] = True
    write_json(selection_path, report)
    print(f"Validated physical threshold selection: {selection_path}", flush=True)


def run_threshold_experiment(output_dir: Path) -> None:
    started = time.monotonic()
    status = "failed"
    try:
        _run_threshold_experiment(output_dir)
        status = "passed"
    finally:
        jobs = [json.loads(path.read_text()) for path in output_dir.rglob("*.done.json")]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "runtime.json", {
            "status": status, "elapsed_s": time.monotonic() - started,
            "completed_jobs": len(jobs), "completed_job_elapsed_s": sum(job["elapsed_s"] for job in jobs),
            "maximum_worker_rss_bytes": max((job.get("maximum_worker_rss_bytes", 0) for job in jobs), default=0),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        request = json.loads(args.worker.read_text())
        output = args.worker.with_name(args.worker.name.removesuffix(".request.json"))
        if request["action"] == "calibrate":
            calibrate_worker(request, output)
        elif request["action"] == "cadc":
            from scripts.evaluation.brainscales2_gain_diagnostic import cadc_worker
            cadc_worker(request, output)
        else:
            measure_worker(request, output)
    elif args.output_dir is not None:
        run_threshold_experiment(args.output_dir.resolve())
    else:
        parser.error("--output-dir is required")


if __name__ == "__main__":
    main()
