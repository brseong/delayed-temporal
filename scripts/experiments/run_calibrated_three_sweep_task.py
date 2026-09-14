"""Execute one immutable task on a single allocated RTX A6000 device."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.calibrated_three_sweeps import (
    TAG, parse_result_log, safe_output, sha256_file, task_sha256, validate_experiment,
    validate_result, validate_table, validate_task, write_immutable_json,
)


def check_source(experiment: dict[str, Any]) -> None:
    source = Path(experiment["source_root"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != experiment["source_commit"] or dirty.strip():
        raise ValueError("Source root must have its frozen HEAD and no tracked modifications")
    for prefix in ("evaluator", "calibration_evaluator", "gelu_evaluator"):
        path = source / experiment[f"{prefix}_path"]
        if sha256_file(path) != experiment[f"{prefix}_sha256"]:
            raise ValueError(f"Evaluator content mismatch: {prefix}")
    for relative, expected in experiment.get("runtime_sha256", {}).items():
        if sha256_file(safe_output(source, relative)) != expected:
            raise ValueError(f"Runtime content mismatch: {relative}")
    from scripts.experiments.ubai.prepare_calibrated_three_sweeps_ubai import package_source_identity
    dependencies = experiment.get("dependency_sha256", {})
    if set(dependencies) != {"transformers", "spikingjelly"}:
        raise ValueError("Editable dependency source identities are required")
    for package, subtree in (("transformers", "src"), ("spikingjelly", "spikingjelly")):
        if package_source_identity(source / "src" / package / subtree)[0] != dependencies[package]:
            raise ValueError(f"Editable dependency source differs: {package}")


def require_gpu(experiment: dict[str, Any], host_label: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible in {"-1", "all", "none"}:
        raise ValueError("Exactly one allocated GPU must be visible")
    if host_label == "local" and visible not in {"4", "5", "6", "7"}:
        raise ValueError("Local evaluation is restricted to physical GPU devices 4 through 7")
    if host_label == "ubai" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("UBAI evaluation requires a Slurm allocation")
    probe = subprocess.check_output([
        experiment["python_bin"], "-c", "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),"
        "'model':torch.cuda.get_device_name(0) if torch.cuda.device_count() else ''}))"], text=True)
    data = json.loads(probe)
    if data["count"] != 1 or "RTX A6000" not in data["model"]:
        raise ValueError("Exactly one RTX A6000 GPU is required")
    return data["model"]


def evaluator_command(experiment: dict[str, Any], task: dict[str, Any], output_root: Path) -> list[str]:
    source = Path(experiment["source_root"])
    dense = task["kind"] == "dense"
    collect = task["kind"] == "collect"
    command = [experiment["python_bin"], "-u", str(source / experiment["evaluator_path" if dense else "calibration_evaluator_path"])]
    if not dense:
        command += ["--source-root", str(source), "--calibration-dataset-path", experiment["calibration_dataset_path"],
                    "--calibration-dataset-fingerprint", experiment["calibration_dataset_fingerprint"],
                    "--gelu-cubic-implementation", "phi_nl_psi_ed", "--gelu-cubic-floor", "1e-5"]
    command += ["--experiment_name", task["run_id"], "--device", "cuda", "--model_backend", task["backend"],
                "--model_id", experiment["checkpoint_path"], "--dataset_id", "imagenet-1k",
                "--evaluation-dataset-path", experiment["calibration_dataset_path" if task["split"] == "train" else "dataset_path"],
                "--evaluation-split", task["split"], "--batch_size", "32", "--theta", str(task["theta"]),
                "--precision", "float64", "--source-commit", experiment["source_commit"],
                "--checkpoint-sha256", experiment["checkpoint_sha256"], "--no-tensorboard", "--report-clamp-stats",
                "--spiking-layernorm", "--spiking-ln-mul", "--spiking-ln-log", "--spiking-ln-expdiff",
                "--spiking-mlp", "--spiking-attention", "--no-spiking-mlp-exact-gelu",
                "--calibration-mode", "none" if dense else "collect" if collect else "validate"]
    if not dense:
        command += ["--calibration-path", str(safe_output(output_root, task["calibration_file"])),
                    "--calibration-samples", "5000", "--calibration-seed", "0", "--calibration-bins", "2048",
                    "--calibration-lower-quantile", "0", "--calibration-upper-quantile", "1", "--calibration-margin-fraction", "0.05"]
    if task["split"] == "validation":
        command.append("--quick-test")
    if task["kind"].startswith("smoke"):
        command += ["--max_eval_batches", "5"]
    noisy = task["kind"] in {"noise", "smoke_noise"}
    command += ["--gaussian-time-noise" if noisy else "--no-gaussian-time-noise",
                "--time-noise-seed", str(task["seed"] if noisy else 0),
                "--time-noise-std-frac", str(task["time_noise_std_frac"]), "--time-noise-mean", "0",
                "--time-noise-deadline-margin-std", str(task["deadline_margin_std"]),
                "--no-mismatch-enabled", "--mismatch-theta-std", "0", "--mismatch-seed", "0",
                "--weight-noise-std", "0", "--bias-noise-std", "0"]
    return command


def execute(experiment_path: Path, task_path: Path, output_root: Path, host_label: str) -> dict[str, Any]:
    experiment = json.loads(experiment_path.read_text())
    task = json.loads(task_path.read_text())
    validate_experiment(experiment)
    validate_task(task, experiment)
    if task.get("host_label", host_label) != host_label:
        raise ValueError("Task was assigned to a different execution environment")
    if platform.python_version() != "3.12.13":
        raise ValueError("Python 3.12.13 is required")
    check_source(experiment)
    output_root = output_root.resolve()
    result_path = safe_output(output_root, task["result_file"])
    log_path = safe_output(output_root, task["log_file"])
    locks = output_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{task['run_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            validate_result(task, result, experiment, output_root)
            return result
        require_gpu(experiment, host_label)
        if task["kind"] not in {"dense", "collect"}:
            table_path = safe_output(output_root, task["calibration_file"])
            if sha256_file(table_path) != task["calibration_sha256"]:
                raise ValueError("Calibration table differs from its assigned hash")
            validate_table(table_path, task, experiment)
        # Only this exclusively locked task's unfinished outputs are moved.
        rejected = output_root / "rejected" / f"{task['run_id']}-{time.time_ns()}"
        leftovers = [log_path]
        if task["kind"] == "collect":
            leftovers.append(safe_output(output_root, task["calibration_file"]))
        for path in leftovers:
            if path.exists():
                rejected.mkdir(parents=True, exist_ok=True)
                path.rename(rejected / path.name)
        runtime_base = Path(os.environ["TMPDIR"]) if host_label == "ubai" else Path(experiment.get(
            "runtime_root", f"/data/delayed-temporal/artifacts/runtime/{TAG}"))
        if runtime_base == Path("/tmp") or runtime_base.is_relative_to("/tmp"):
            raise ValueError("Runtime storage cannot use /tmp")
        runtime_base.mkdir(parents=True, exist_ok=True)
        runtime = Path(tempfile.mkdtemp(prefix=f"{task['run_id']}-", dir=runtime_base))
        environment = dict(os.environ)
        environment.update(WANDB_MODE="disabled", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
                           HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           TMPDIR=str(runtime), XDG_CACHE_HOME=str(runtime / "cache"),
                           PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        source = Path(experiment["source_root"])
        environment["PYTHONPATH"] = os.pathsep.join(str(p) for p in (
            source, source / "src/transformers/src", source / "src/spikingjelly"))
        for key in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            environment.pop(key, None)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if task["calibration_file"]:
            safe_output(output_root, task["calibration_file"]).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        previous_handlers = {}
        child: subprocess.Popen | None = None

        def stop(signum: int, _frame: Any) -> None:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise SystemExit(128 + signum)

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, stop)
            with log_path.open("x") as log:
                log.write(f"Slurm identity — job: {os.environ.get('SLURM_JOB_ID', 'local')}, gpu_family: rtxa6000\n")
                log.flush()
                child = subprocess.Popen(evaluator_command(experiment, task, output_root), cwd=source,
                                         env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
            if code != 0:
                raise RuntimeError(f"Evaluator failed with exit code {code}; preserved {log_path}")
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        check_source(experiment)
        result = {**task, "success": True, "task_sha256": task_sha256(task),
                  "experiment_sha256": task_sha256(experiment), "host_label": host_label,
                  "elapsed_seconds": time.monotonic() - started, "log_sha256": sha256_file(log_path)}
        if task["kind"] == "collect":
            table = safe_output(output_root, task["calibration_file"])
            validate_table(table, task, experiment)
            result.update(calibration_sha256=sha256_file(table), sites=48)
        else:
            result.update(parse_result_log(task, output_root))
            result["log_file"] = task["log_file"]
        validate_result(task, result, experiment, output_root)
        write_immutable_json(result_path, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host-label", choices=("local", "ubai"), required=True)
    args = parser.parse_args()
    result = execute(args.experiment, args.task, args.output_root, args.host_label)
    print(json.dumps({"run_id": result["run_id"], "success": result["success"]}, sort_keys=True))


if __name__ == "__main__":
    main()
