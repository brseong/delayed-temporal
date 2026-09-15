"""Run one small calibrated ViT ImageNet comparison without admitting full results."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


def set_option(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def require_frozen_source(source: Path, commit: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("A full source commit is required")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True,
    )
    if head != commit or dirty.strip():
        raise ValueError("Source root must have its frozen HEAD and no tracked modifications")


def derive_experiment(original: dict, source_root: Path | None,
                      source_commit: str | None) -> dict:
    """Change only source identity when both explicit overrides are supplied."""
    if (source_root is None) != (source_commit is None):
        raise ValueError("--source-root and --source-commit must be supplied together")
    experiment = deepcopy(original)
    source = Path(experiment["source_root"]) if source_root is None else source_root.resolve()
    commit = experiment["source_commit"] if source_commit is None else source_commit
    require_frozen_source(source, commit)
    if source_root is None:
        return experiment

    def code_hash(relative: str) -> str:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (source / path).resolve().is_relative_to(source):
            raise ValueError("Code identity paths must remain inside the frozen source")
        return hashlib.sha256((source / path).read_bytes()).hexdigest()

    experiment["source_root"] = str(source)
    experiment["source_commit"] = commit
    for prefix in ("evaluator", "calibration_evaluator", "gelu_evaluator"):
        experiment[prefix + "_sha256"] = code_hash(experiment[prefix + "_path"])
    experiment["runtime_sha256"] = {
        relative: code_hash(relative) for relative in experiment["runtime_sha256"]
    }
    return experiment


# @lat: [[quick-family-checks#Quick Model Family Checks#Small ViT Comparison]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if socket.gethostname() != "baekryun-cuda129":
        raise ValueError("Quick local evaluation must run on baekryun-cuda129")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in {"4", "5", "6", "7"}:
        raise ValueError("Exactly one allowed local GPU is required")
    root = args.output_root.resolve()
    if not root.is_relative_to("/data/delayed-temporal/artifacts/logs/quick_family_checks"):
        raise ValueError("Quick output must remain in the dedicated artifact directory")
    original_bytes = args.experiment.read_bytes()
    original = json.loads(original_bytes)
    experiment = derive_experiment(original, args.source_root, args.source_commit)
    root.mkdir(parents=True, exist_ok=False)
    source = Path(experiment["source_root"])
    sys.path.insert(0, str(source))
    from scripts.experiments.run_vit_comparison import check_assets
    from scripts.experiments.run_calibrated_three_sweeps import gpu_activity, gpu_available
    from scripts.experiments.vit_comparison import (
        check_source, evaluator_command, make_task, model_by_key, require_gpu,
    )
    locks = Path("/data/delayed-temporal/artifacts/runtime/gpu-locks")
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"gpu-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not gpu_available(gpu_activity()[int(gpu)]):
            raise RuntimeError("Assigned GPU is occupied")
        check_source(experiment)
        require_gpu(experiment, "local")
        model = model_by_key(experiment, "imagenet_vit_small")
        check_assets(experiment, model, "local")
        runtime = Path("/data/delayed-temporal/artifacts/runtime/quick_family_checks") / root.name
        runtime.mkdir(parents=True, exist_ok=False)
        fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
        if not fs or fs in {"tmpfs", "ramfs"}:
            raise ValueError("Quick runtime must use real disk")
        environment = dict(os.environ)
        environment.update(
            TMPDIR=str(runtime), TMP=str(runtime), TEMP=str(runtime),
            WANDB_MODE="disabled", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            PYTHONPATH=os.pathsep.join(map(str, (source, source / "src/transformers/src", source / "src/spikingjelly"))),
        )
        calibration = root / "calibration.json"
        commands = []
        for kind in ("smoke_collect", "dense", "smoke_spiking"):
            task = make_task(experiment, model["model_key"], kind, 32)
            command = evaluator_command(experiment, task, root)
            set_option(command, "--experiment_name", "quick_imagenet_vit_small_" + kind)
            if kind != "dense":
                set_option(command, "--calibration-smoke-samples", "256")
                set_option(command, "--calibration-samples", "256")
                set_option(command, "--calibration-path", str(calibration))
            if "--max_eval_batches" in command:
                set_option(command, "--max_eval_batches", "8")
            elif kind == "dense":
                command += ["--max_eval_batches", "8"]
            commands.append({"kind": kind, "command": command, "log": str(root / (kind + ".log"))})
        derived_path = root / "experiment.json"
        derived_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
        manifest = {
            "purpose": "quick_check_only", "source_commit": experiment["source_commit"],
            "source_root": str(source), "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "experiment_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "original_experiment_path": str(args.experiment.resolve()),
            "original_source_root": original["source_root"],
            "original_source_commit": original["source_commit"],
            "source_override": args.source_root is not None,
            "derived_experiment_path": str(derived_path),
            "derived_experiment_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
            "model": model, "calibration_samples": 256, "evaluation_samples": 256,
            "precision": "float64", "theta": 40, "gpu": gpu, "runtime": str(runtime),
            "commands": commands,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("Quick comparison — " + json.dumps({"phase": "started", "root": str(root), "gpu": gpu}), flush=True)
        for row in commands:
            started = time.monotonic()
            print("Quick comparison — " + json.dumps({"phase": row["kind"], "log": row["log"]}), flush=True)
            with Path(row["log"]).open("x") as log:
                completed = subprocess.run(row["command"], cwd=source, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),))
            result = {"kind": row["kind"], "returncode": completed.returncode,
                      "elapsed_seconds": time.monotonic() - started,
                      "log_sha256": hashlib.sha256(Path(row["log"]).read_bytes()).hexdigest()}
            (root / (row["kind"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            print("Quick comparison — " + json.dumps(result), flush=True)
            if completed.returncode:
                raise RuntimeError("Quick comparison failed; logs preserved")
        print("Quick comparison — completed", flush=True)


if __name__ == "__main__":
    main()
