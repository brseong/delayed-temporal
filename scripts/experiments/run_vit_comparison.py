"""Run and resume independent deterministic ViT comparison pipelines."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.experiments.vit_comparison import (
    DEFAULT_ROOT, MODEL_KEYS, PYTHON, SOURCE, TAG, check_source, evaluator_command,
    make_task, model_by_key, parsed_result, read_json, require_gpu, safe_output,
    sha256_file, task_sha256, validate_experiment, validate_result, validate_table,
    validate_task, write_immutable_json, require_current_experiment,
)
from scripts.experiments.run_calibrated_three_sweeps import atomic_json, gpu_activity, gpu_available
from scripts.experiments.ubai.run_calibrated_three_sweep_pair import worker_environment

SCRIPT = "scripts/experiments/run_vit_comparison.py"
REMOTE_BASE = "/home1/sizz1997/myubai"
REMOTE_ROOT = f"{REMOTE_BASE}/delayed-temporal-experiments/{TAG}"


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "transformers", "spikingjelly", "numpy", "datasets",
        "safetensors", "timm", "tokenizers", "scipy")}


def event(root: Path, name: str, **fields: Any) -> None:
    payload = {"event": name, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    line = json.dumps(payload, sort_keys=True, allow_nan=False)
    with (root / "events.jsonl").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def discover_calibration_sites(config_fields: dict) -> list[dict]:
    """Record actual module identities without allocating checkpoint weights."""
    import torch
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification
    from utils.transformers.models.spiking_vit.calibration import vit_calibration_specs
    config = ViTConfig(**config_fields)
    config.theta, config.tau_s, config.clip_margin = 40.0, 1.0, 1e-5
    config._attn_implementation = "eager"
    config.use_spiking_layernorm = config.use_spiking_mlp = True
    config.spiking_ln_mul = config.spiking_ln_log = config.spiking_ln_expdiff = True
    config.spiking_mlp_exact_gelu = False
    with torch.device("meta"):
        model = ViTForImageClassification(config).double().eval()
    config._attn_implementation = "spiking_sdpa"
    return [{"module_name": spec.module_name, "tensor_name": spec.tensor_name,
             "range_policy": spec.range_policy.value, "fixed_min": spec.fixed_min,
             "fixed_max": spec.fixed_max}
            for spec in vit_calibration_specs(model, lower_quantile=0.0,
                                             upper_quantile=1.0, margin_fraction=0.05)]


def initialize(root: Path, source: Path, assets_manifest: Path) -> dict:
    from scripts.experiments.ubai.prepare_calibrated_three_sweeps_ubai import artifact_records, package_source_identity
    if socket.gethostname() != "baekryun-cuda129" or root.name != TAG:
        raise ValueError("Initialization requires the local host and fixed comparison tag")
    if (root / "experiment.json").exists():
        experiment = read_json(root / "experiment.json")
        validate_experiment(experiment)
        require_current_experiment(experiment)
        check_source(experiment)
        return experiment
    assets = read_json(assets_manifest)
    experiment = {"tag": TAG, "source_root": str(source.resolve()), "python_bin": PYTHON,
                  "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                  "theta": 40.0, "precision": "float64", "output_bounds_version": 3,
                  "vit_calibration_policy_version": 2,
                  "tau_s": 1.0, "tracking": "disabled", "local_gpu_ids": [4, 5, 6, 7],
                  "evaluation_count": 8, "calibration_count": 4,
                  "models": assets["models"], "assets_manifest_sha256": sha256_file(assets_manifest),
                  "runtime_root": f"/data/delayed-temporal/artifacts/runtime/{TAG}",
                  "package_versions": package_versions(),
                  "asset_checks": {}}
    for prefix, relative in (("evaluator", "scripts/evaluation/error_analysis_vit.py"),
                             ("calibration_evaluator", "scripts/analysis/evaluate_calibrated_vit.py"),
                             ("gelu_evaluator", "scripts/analysis/gelu_cubic_phi_nl_vit.py")):
        experiment[prefix + "_path"] = relative
        experiment[prefix + "_sha256"] = sha256_file(source / relative)
    for row in experiment["models"]:
        row["calibration_sites"] = discover_calibration_sites(row["checkpoint_config"])
        row["preprocessing_sha256"] = sha256_file(Path(row["checkpoint_path"]) / "preprocessor_config.json")
        for field in ("checkpoint", "dataset", "calibration_dataset"):
            path = row[field + "_path"]
            if path not in experiment["asset_checks"]:
                digest, records = artifact_records(Path(path))
                experiment["asset_checks"][path] = {"sha256": digest, "files": records}
            if experiment["asset_checks"][path]["sha256"] != row[field + "_sha256"]:
                raise ValueError("Prepared asset hash differs")
    experiment["dependency_sha256"] = {
        name: package_source_identity(source / "src" / name / subtree)[0]
        for name, subtree in (("transformers", "src"), ("spikingjelly", "spikingjelly"))}
    runtime_files = [SCRIPT, "scripts/experiments/vit_comparison.py",
                     "scripts/experiments/vit_comparison_controller.py",
                     "scripts/experiments/ubai/run_vit_comparison_ubai.py",
                     "scripts/experiments/ubai/vit_comparison_prep.sbatch",
                     "scripts/experiments/ubai/vit_comparison_task.sbatch",
                     "scripts/analysis/summarize_vit_comparison.py", "scripts/analysis/vit_comparison_costs.py",
                     "scripts/analysis/publish_vit_comparison.py"]
    experiment["runtime_sha256"] = {name: sha256_file(source / name) for name in runtime_files}
    validate_experiment(experiment)
    check_source(experiment)
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / "experiment.json", experiment)
    return experiment


def check_assets(experiment: dict, model: dict, host_label: str) -> None:
    # Remote mounts have different inode timestamps; the Slurm preparation checks
    # their hashes and immutable file membership before starting the container.
    if host_label == "ubai":
        return
    for field in ("checkpoint", "dataset", "calibration_dataset"):
        root = Path(model[field + "_path"])
        records = experiment["asset_checks"][str(root)]["files"]
        expected = {item["path"] for item in records}
        actual = {str(item) for item in root.rglob("*") if item.is_file()}
        if actual != expected:
            raise ValueError("Asset file membership changed after hashing")
        for record in records:
            stat = Path(record["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError("Asset changed after hashing")


def completed(root: Path, experiment: dict, task: dict) -> dict | None:
    path = safe_output(root, task["result_file"])
    if not path.exists():
        return None
    result = read_json(path)
    validate_result(task, result, experiment, root)
    return result


def run_task(root: Path, experiment: dict, task: dict, host_label: str) -> dict:
    require_current_experiment(experiment)
    validate_task(task, experiment)
    task_path = root / "tasks" / (task["run_id"] + ".json")
    write_immutable_json(task_path, task)
    locks = root / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / (task["run_id"] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = completed(root, experiment, task)
        if result is not None:
            return result
        if host_label == "local" and os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", "5", "6", "7"}:
            raise ValueError("Local comparison requires exactly one physical GPU from 4 through 7")
        check_source(experiment)
        model = model_by_key(experiment, task["model_key"])
        check_assets(experiment, model, host_label)
        collect = task["kind"] in {"collect", "smoke_collect"}
        table_path = safe_output(root, task["calibration_file"]) if task["calibration_file"] else None
        if table_path is not None and not collect:
            if sha256_file(table_path) != task["calibration_sha256"]:
                raise ValueError("Assigned calibration hash differs")
            validate_table(table_path, task, experiment)
        log_path = safe_output(root, task["log_file"])
        rejected = root / "rejected" / (task["run_id"] + "-" + str(time.time_ns()))
        for path in [log_path] + ([table_path] if collect and table_path is not None else []):
            if path.exists():
                rejected.mkdir(parents=True, exist_ok=True)
                path.rename(rejected / path.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if table_path is not None:
            table_path.parent.mkdir(parents=True, exist_ok=True)
        base = Path(os.environ["TMPDIR"]) if host_label == "ubai" else Path(experiment["runtime_root"])
        if base == Path("/tmp") or base.is_relative_to("/tmp"):
            raise ValueError("Task runtime cannot reside in /tmp")
        base.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=task["run_id"] + "-", dir=base))
        environment = worker_environment(dict(os.environ), scratch, os.environ["CUDA_VISIBLE_DEVICES"])
        environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        source = Path(experiment["source_root"])
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in (
            source, source / "src/transformers/src", source / "src/spikingjelly"))
        previous = {}
        child = None
        started = time.monotonic()
        event(root, "task_started", run_id=task["run_id"], model_key=task["model_key"], host_label=host_label,
              batch_size=task["batch_size"], gpu=os.environ["CUDA_VISIBLE_DEVICES"])
        def interrupted(signum: int, _frame: Any) -> None:
            raise InterruptedError(f"Evaluator interrupted: {signum}")
        try:
            previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
            with log_path.open("x") as log:
                log.write(f"Slurm identity — job: {os.environ.get('SLURM_JOB_ID', 'local')}, gpu_family: rtxa6000\n")
                log.write("Comparison task — " + json.dumps({"task_sha256": task_sha256(task), "batch_size": task["batch_size"]}) + "\n")
                log.flush()
                child = subprocess.Popen(evaluator_command(experiment, task, root), cwd=source,
                                         env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                code = child.wait()
            if code != 0:
                raise RuntimeError(f"Evaluator exit {code}; preserved {log_path}")
        finally:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        result = {**task, "success": True, "task_sha256": task_sha256(task),
                  "experiment_sha256": task_sha256(experiment), "host_label": host_label,
                  "elapsed_seconds": time.monotonic() - started, "log_sha256": sha256_file(log_path)}
        if collect:
            assert table_path is not None
            table = validate_table(table_path, task, experiment)
            result.update(calibration_sha256=sha256_file(table_path), sites=len(table["layers"]),
                          samples=task["calibration_samples"], total=task["calibration_samples"])
        else:
            result.update(parsed_result(task, root))
        check_source(experiment)
        validate_result(task, result, experiment, root)
        write_immutable_json(safe_output(root, task["result_file"]), result)
        # The exact task-owned scratch directory is recoverable data only until
        # successful completion; failed runtime directories remain for inspection.
        if scratch.parent == base and not scratch.is_symlink() and scratch.stat().st_uid == os.getuid():
            shutil.rmtree(scratch)
        event(root, "task_completed", run_id=task["run_id"], model_key=task["model_key"],
              host_label=host_label, elapsed_seconds=result["elapsed_seconds"], correct=result.get("correct"))
        return result


def admission(root: Path, experiment: dict, key: str, host_label: str) -> dict:
    path = root / "admissions" / (key + ".json")
    if path.exists():
        selected = read_json(path)
        if (selected.get("experiment_sha256") != task_sha256(experiment)
                or selected.get("model_key") != key or type(selected.get("batch_size")) is not int
                or selected["batch_size"] not in (32, 16, 8)):
            raise ValueError("Admission source differs")
        batch = selected["batch_size"]
        expected_ids = {f"{key}_smoke_collect_bs{batch}", f"{key}_smoke_spiking_bs{batch}"}
        if set(selected.get("result_sha256", {})) != expected_ids:
            raise ValueError("Admission requires both smoke results")
        evidence = {}
        for run_id, digest in selected["result_sha256"].items():
            result_path = root / "results" / (run_id + ".json")
            if sha256_file(result_path) != digest:
                raise ValueError("Admission evidence changed")
            validate_result(read_json(root / "tasks" / (run_id + ".json")), read_json(result_path), experiment, root)
            evidence[run_id] = read_json(result_path)
        evaluated = evidence[f"{key}_smoke_spiking_bs{batch}"]
        collected = evidence[f"{key}_smoke_collect_bs{batch}"]
        if any(selected.get(field) != evaluated.get(field) for field in
               ("model_key", "batch_size", "correct", "samples", "prediction_sha256", "calibration_sha256")):
            raise ValueError("Admission differs from its evaluated evidence")
        if collected["calibration_sha256"] != selected["calibration_sha256"]:
            raise ValueError("Admission collection and evaluation differ")
        return selected
    for batch in (32, 16, 8):
        task = make_task(experiment, key, "smoke_collect", batch)
        try:
            collection = run_task(root, experiment, task, host_label)
            task = make_task(experiment, key, "smoke_spiking", batch,
                             calibration_sha256=collection["calibration_sha256"])
            evaluated = run_task(root, experiment, task, host_label)
        except RuntimeError:
            log = safe_output(root, task["log_file"])
            content = log.read_text() if log.exists() else ""
            if not re.search(r"(?:CUDA out of memory|torch\.OutOfMemoryError)", content):
                raise
            event(root, "memory_candidate_rejected", model_key=key, batch_size=batch, log_file=task["log_file"])
            continue
        selected = {"model_key": key, "batch_size": batch, "experiment_sha256": task_sha256(experiment),
                    "calibration_sha256": collection["calibration_sha256"],
                    "correct": evaluated["correct"], "samples": evaluated["samples"],
                    "prediction_sha256": evaluated["prediction_sha256"],
                    "result_sha256": {r["run_id"]: sha256_file(safe_output(root, r["result_file"])) for r in (collection, evaluated)}}
        write_immutable_json(path, selected)
        event(root, "model_admitted", model_key=key, batch_size=batch)
        return selected
    raise RuntimeError(f"All batch sizes failed memory admission: {key}")


def pipeline(root: Path, experiment: dict, key: str, host_label: str) -> None:
    selected = admission(root, experiment, key, host_label)
    batch = selected["batch_size"]
    collection = run_task(root, experiment, make_task(experiment, key, "collect", batch), host_label)
    run_task(root, experiment, make_task(experiment, key, "dense", batch), host_label)
    run_task(root, experiment, make_task(experiment, key, "spiking", batch,
                                       calibration_sha256=collection["calibration_sha256"]), host_label)
    event(root, "model_completed", model_key=key)


def environment_replay(root: Path, experiment: dict, host_label: str) -> None:
    selected = admission(root, experiment, "imagenet_vit_base", host_label)
    task = make_task(experiment, "imagenet_vit_base", "environment_spiking", selected["batch_size"],
                     calibration_sha256=selected["calibration_sha256"], host_label="ubai")
    result = run_task(root, experiment, task, host_label)
    if any(result[field] != selected[field] for field in ("correct", "samples", "prediction_sha256")):
        raise ValueError("Local and UBAI environment predictions differ")
    event(root, "environment_confirmed", model_key="imagenet_vit_base")


def worker(root: Path, mode: str, key: str, host_label: str) -> None:
    experiment = read_json(root / "experiment.json")
    validate_experiment(experiment)
    require_current_experiment(experiment)
    if platform.python_version() != "3.12.13":
        raise ValueError("The evaluator requires Python3.12.13")
    if package_versions() != experiment["package_versions"]:
        raise ValueError("Dependency package versions differ")
    check_source(experiment)
    gpu_lock = None
    if host_label == "local":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible not in {"4", "5", "6", "7"}:
            raise ValueError("Local execution is restricted to GPU4 through7")
        lock_dir = Path("/data/delayed-temporal/artifacts/runtime/gpu-locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        gpu_lock = (lock_dir / f"gpu-{visible}.lock").open("a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not gpu_available(gpu_activity()[int(visible)]):
            raise RuntimeError("Assigned local GPU is occupied")
    require_gpu(experiment, host_label)
    try:
        if mode == "admit":
            admission(root, experiment, key, host_label)
        elif mode == "environment":
            environment_replay(root, experiment, host_label)
        else:
            pipeline(root, experiment, key, host_label)
    finally:
        if gpu_lock is not None:
            gpu_lock.close()


def summarize(root: Path, experiment: dict, *, require_complete: bool = False) -> dict:
    from scripts.analysis.summarize_vit_comparison import build_outputs
    results = []
    for path in sorted((root / "results").glob("*.json")):
        result = read_json(path)
        if result["kind"] not in {"collect", "dense", "spiking"}:
            continue
        task = read_json(root / "tasks" / (result["run_id"] + ".json"))
        validate_result(task, result, experiment, root)
        results.append(result)
    return build_outputs(experiment, results, root / "outputs", require_complete=require_complete)


def prepare_execution(root: Path, experiment: dict) -> dict:
    """Save validated next commands without launching a full model evaluation."""
    validate_experiment(experiment)
    require_current_experiment(experiment)
    check_source(experiment)
    prepared = []
    for key in MODEL_KEYS:
        selected_path = root / "admissions" / (key + ".json")
        selected = admission(root, experiment, key, "local") if selected_path.exists() else None
        command = [experiment["python_bin"], str(Path(experiment["source_root"]) / SCRIPT),
                   "pipeline", "--root", str(root.resolve()), "--model", key, "--host-label", "local"]
        prepared.append({"model_key": key, "batch_size": selected["batch_size"] if selected else None,
                         "admission_verified": selected is not None,
                         "requires_full_calibration": True, "command": command,
                         "commands_by_gpu": {str(gpu): ["env", f"CUDA_VISIBLE_DEVICES={gpu}", *command]
                                             for gpu in (4, 5, 6, 7)}})
    value = {"experiment_sha256": task_sha256(experiment), "source_commit": experiment["source_commit"],
             "vit_calibration_policy_version": 2, "allowed_gpu_ids": [4, 5, 6, 7],
             "launch_performed": False, "models": prepared}
    destination = root / "prepared" / (task_sha256(value) + ".json")
    write_immutable_json(destination, value)
    event(root, "execution_prepared", manifest=str(destination), launch_performed=False)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initialize", "admit", "prepare", "pipeline", "environment", "summarize", "run"))
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--source", type=Path, default=Path(SOURCE))
    parser.add_argument("--assets-manifest", type=Path,
                        default=Path("/data/delayed-temporal/artifacts/assets/vit-conversion-comparison-v1/manifest.json"))
    parser.add_argument("--model", choices=MODEL_KEYS, default="imagenet_vit_base")
    parser.add_argument("--host-label", choices=("local", "ubai"), default="local")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args.root, args.source, args.assets_manifest)
    elif args.mode == "prepare":
        prepare_execution(args.root, read_json(args.root / "experiment.json"))
    elif args.mode == "summarize":
        summarize(args.root, read_json(args.root / "experiment.json"), require_complete=args.require_complete)
    elif args.mode == "run":
        require_current_experiment(read_json(args.root / "experiment.json"))
        from scripts.experiments.vit_comparison_controller import run
        run(args.root)
    else:
        worker(args.root, args.mode, args.model, args.host_label)


if __name__ == "__main__":
    main()
