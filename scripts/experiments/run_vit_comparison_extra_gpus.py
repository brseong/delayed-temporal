"""Use explicitly authorized extra GPUs for this comparison's pending dense evaluations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

TAG = "vit_conversion_comparison_theta40_calibrated_float64_bounds3_v1"
DEFAULT_ROOT = Path("/data/delayed-temporal/artifacts/logs/conversion_comparison") / TAG
SOURCE = Path("/data/delayed-temporal-worktrees/vit-conversion-comparison")
SOURCE_COMMIT = "2c0fbd3f6fd1d043ffc34295f891d4bccaa5113a"
EXPERIMENT_FILE_SHA256 = "176dded80d1bfd804b831e6e584fa638b97a0f42193dcaf97b7fa8535d3fd14f"
MODEL_KEYS = ("cifar10_vit_small", "imagenet_vit_small", "imagenet_vit_base", "imagenet_vit_large")
EXTRA_GPUS = (1, 2, 3)
GPU_LOCK_ROOT = Path("/data/delayed-temporal/artifacts/runtime/gpu-locks")
MINIMUM_REMAINING = 40
STOP_REMAINING = 20


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scope(root: Path, gpu: int, model: str, temporary_local_gpus: bool) -> dict:
    if not temporary_local_gpus or type(gpu) is not int or gpu not in EXTRA_GPUS:
        raise ValueError("An explicit current-comparison authorization for GPU1,2,3 is required")
    if root.resolve() != DEFAULT_ROOT or model not in MODEL_KEYS:
        raise ValueError("Temporary authorization does not cover this root or model")
    path = root / "experiment.json"
    if digest(path) != EXPERIMENT_FILE_SHA256:
        raise ValueError("The frozen experiment file differs")
    experiment = read_json(path)
    if (experiment.get("tag") != TAG or experiment.get("source_commit") != SOURCE_COMMIT
            or experiment.get("source_root") != str(SOURCE)
            or experiment.get("local_gpu_ids") != [4, 5, 6, 7]):
        raise ValueError("The experiment is outside this temporary authorization")
    return experiment


def load_frozen_runner(experiment: dict):
    """Import no mutable-checkout project modules into the temporary worker."""
    for name, module in tuple(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(SOURCE):
                raise ValueError("A project module was imported outside the frozen checkout")
            paths = getattr(module, "__path__", ())
            if any(not Path(path).resolve().is_relative_to(SOURCE) for path in paths):
                raise ValueError("A project package was imported outside the frozen checkout")
    sys.path.insert(0, str(SOURCE))
    path = SOURCE / "scripts/experiments/run_vit_comparison.py"
    spec = importlib.util.spec_from_file_location("_frozen_vit_comparison_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the frozen comparison runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.validate_experiment(experiment)
    runner.check_source(experiment)
    if platform.python_version() != "3.12.13" or runner.package_versions() != experiment["package_versions"]:
        raise ValueError("The frozen Python or dependency versions differ")
    return runner


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def lock_held(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with lock(path):
            return False
    except BlockingIOError:
        return True


def process_identity(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return {"pid": pid, "start_ticks": int(fields[19]), "command": command, "state": fields[0]}
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def calibration_progress(text: str) -> dict:
    records = []
    for line in text.splitlines():
        if line.startswith("Calibration progress — "):
            try:
                records.append(json.loads(line.removeprefix("Calibration progress — ")))
            except json.JSONDecodeError:
                continue
    if not records:
        raise ValueError("Known calibration progress is required")
    row = records[-1]
    completed, total = row.get("completed_batches"), row.get("total_batches")
    if (type(completed) is not int or type(total) is not int or not 0 <= completed <= total
            or total <= 0 or row.get("pass") not in (1, 2)):
        raise ValueError("Invalid calibration progress")
    return {**row, "remaining_batches": total - completed}


def prepare_dense(root: Path, experiment: dict, model: str, runner) -> dict:
    if not (root / "admissions" / (model + ".json")).is_file():
        raise ValueError("An existing verified admission is required; this helper cannot collect")
    admission = runner.admission(root, experiment, model, "local")
    return runner.make_task(experiment, model, "dense", admission["batch_size"])


def require_window(root: Path, experiment: dict, model: str, runner,
                   minimum_remaining: int = MINIMUM_REMAINING) -> dict:
    assignments = read_json(root / "assignments.json")
    if assignments.get("experiment_sha256") != runner.task_sha256(experiment):
        raise ValueError("Central assignment identity differs")
    assignment = assignments.get("models", {}).get(model, {})
    worker = assignments.get("local", {}).get(model, {})
    if (assignment.get("owner") != "local" or assignment.get("status") != "running"
            or worker.get("status") != "running" or worker.get("mode") != "pipeline"):
        raise ValueError("The main model pipeline is not running locally")
    identity = process_identity(int(worker.get("pid", -1)))
    if not identity or identity["state"] == "Z" or any(
            identity[field] != worker.get(field) for field in ("pid", "start_ticks", "command")):
        raise ValueError("The main pipeline process identity differs")
    task = runner.make_task(experiment, model, "collect", prepare_dense(root, experiment, model, runner)["batch_size"])
    if (root / task["result_file"]).exists() or not lock_held(root / "locks" / (task["run_id"] + ".lock")):
        raise ValueError("The main collection is no longer running")
    actual = read_json(root / "tasks" / (task["run_id"] + ".json"))
    if actual != task:
        raise ValueError("The running collection task differs")
    log = (root / task["log_file"]).read_text()
    headers = [json.loads(line.removeprefix("Comparison task — ")) for line in log.splitlines()
               if line.startswith("Comparison task — ")]
    if headers != [{"task_sha256": runner.task_sha256(task), "batch_size": task["batch_size"]}]:
        raise ValueError("The running collection log identity differs")
    progress = calibration_progress(log)
    if progress["total_batches"] != 2 * ((5000 + task["batch_size"] - 1) // task["batch_size"]):
        raise ValueError("The collection progress total differs")
    if progress["remaining_batches"] < minimum_remaining:
        raise ValueError("Insufficient collection time remains for an extra evaluation")
    return {**progress, "main_pid": identity["pid"], "main_start_ticks": identity["start_ticks"],
            "collect_task_sha256": runner.task_sha256(task)}


def gpu_probe(experiment: dict, gpu: int, runner) -> dict:
    if gpu not in EXTRA_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Exactly the explicitly authorized GPU must be visible")
    sample = runner.gpu_activity(gpu_ids=(gpu,))[gpu]
    if not runner.gpu_available(sample):
        raise RuntimeError("The extra GPU is occupied")
    output = subprocess.check_output([
        experiment["python_bin"], "-c",
        "import json,torch; n=torch.cuda.device_count(); print(json.dumps({'count':n,'model':torch.cuda.get_device_name(0) if n else ''}))"
    ], text=True, timeout=45)
    probe = json.loads(output)
    if probe.get("count") != 1 or "RTX A6000" not in probe.get("model", ""):
        raise ValueError("Exactly one RTX A6000 is required")
    return probe


def watchdog(root: Path, experiment: dict, model: str, runner, stop_event: threading.Event,
             signal_self=None, interval: float = 2.0, reason: list | None = None,
             expected_main: tuple[int, int] | None = None) -> None:
    signal_self = os.kill if signal_self is None else signal_self
    while not stop_event.wait(interval):
        try:
            progress = require_window(root, experiment, model, runner, minimum_remaining=0)
            if expected_main is not None and (progress["main_pid"], progress["main_start_ticks"]) != expected_main:
                raise ValueError("The collection process differs from the temporary grant")
            if progress["remaining_batches"] > STOP_REMAINING:
                continue
            message = "Collection is within 20 remaining batches"
        except Exception as error:
            message = str(error)
        if reason is not None:
            reason.append(message)
        if not stop_event.is_set():
            signal_self(os.getpid(), signal.SIGTERM)
        return


# @lat: [[conversion-comparison#Scheduling]]
def execute(root: Path, gpu: int, model: str, temporary_local_gpus: bool) -> dict:
    experiment = validate_scope(root, gpu, model, temporary_local_gpus)
    if socket.gethostname() != "baekryun-cuda129":
        raise ValueError("This temporary authorization is local to baekryun")
    runner = load_frozen_runner(experiment)
    task = prepare_dense(root, experiment, model, runner)
    result = runner.completed(root, experiment, task)
    if result is not None:
        return {"status": "reused", "run_id": task["run_id"]}
    control = root / "temporary_local_gpus"
    try:
        with lock(control / (model + ".lock")), lock(GPU_LOCK_ROOT / f"gpu-{gpu}.lock"):
            if lock_held(root / "locks" / (task["run_id"] + ".lock")):
                return {"status": "deferred", "reason": "The dense evaluation is already active"}
            progress = require_window(root, experiment, model, runner)
            cpus = set(range(16 + (gpu - 1) * 4, 20 + (gpu - 1) * 4))
            if not cpus.issubset(os.sched_getaffinity(0)):
                raise ValueError("Four dedicated extra CPU cores are unavailable")
            assigned = read_json(root / "assignments.json").get("local", {}).values()
            if any(cpus.intersection(worker.get("cpus", [])) for worker in assigned):
                raise ValueError("Extra CPU cores overlap a main worker")
            os.sched_setaffinity(0, cpus)
            runtime = Path(experiment["runtime_root"])
            if runtime != Path("/data/delayed-temporal/artifacts/runtime") / TAG:
                raise ValueError("The disk runtime path differs")
            runtime.mkdir(parents=True, exist_ok=True)
            if subprocess.check_output(["stat", "-f", "-c", "%T", str(runtime)], text=True).strip() in {"tmpfs", "ramfs"}:
                raise ValueError("The extra worker cannot use a RAM filesystem")
            environment = runner.worker_environment(dict(os.environ), runtime / f"extra-gpu-{gpu}", str(gpu))
            for secret in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                os.environ.pop(secret, None)
            os.environ.update(environment)
            Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            probe = gpu_probe(experiment, gpu, runner)
            progress = require_window(root, experiment, model, runner)
            own = process_identity(os.getpid())
            grant = {"tag": TAG, "root": str(root.resolve()), "source_commit": SOURCE_COMMIT,
                     "experiment_sha256": EXPERIMENT_FILE_SHA256, "control_script_sha256": digest(Path(__file__)),
                     "gpu": gpu, "cpus": sorted(cpus), "run_id": task["run_id"], "model_key": model,
                     "pid": os.getpid(), "start_ticks": own["start_ticks"], "progress": progress,
                     "gpu_probe": probe, "temporary_local_gpus": True, "started_at": time.time()}
            grant_id = uuid.uuid4().hex
            runner.write_immutable_json(control / "grants" / (grant_id + ".json"), grant)
            runner.event(root, "temporary_gpu_started", grant_id=grant_id, **grant)
            stop_event, reason = threading.Event(), []
            def interrupted(signum, _frame):
                raise InterruptedError(f"Temporary evaluation interrupted: {signum}")
            previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
            watcher = threading.Thread(target=watchdog, args=(root, experiment, model, runner, stop_event),
                                       kwargs={"reason": reason, "expected_main":
                                               (progress["main_pid"], progress["main_start_ticks"])}, daemon=True)
            outcome = {"status": "failed", "run_id": task["run_id"]}
            try:
                watcher.start()
                runner.run_task(root, experiment, task, "local")
                outcome["status"] = "complete"
            except (InterruptedError, BlockingIOError) as error:
                outcome.update(status="deferred", reason=reason[-1] if reason else str(error))
            finally:
                stop_event.set()
                watcher.join(timeout=5)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                runner.write_immutable_json(control / "end" / (grant_id + ".json"),
                                            {**outcome, "grant_id": grant_id, "finished_at": time.time()})
                runner.event(root, "temporary_gpu_finished", grant_id=grant_id, gpu=gpu, **outcome)
            return outcome
    except BlockingIOError:
        return {"status": "deferred", "reason": "The model or extra GPU is already locked"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=EXTRA_GPUS, required=True)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--temporary-local-gpus", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.root, args.gpu, args.model, args.temporary_local_gpus), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
