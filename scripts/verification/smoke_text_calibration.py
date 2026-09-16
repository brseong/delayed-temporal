#!/usr/bin/env python3
"""Run small local text calibration checks using cached assets and a locked GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(family: str) -> dict[str, str]:
    paths = [
        Path(__file__), ROOT / "scripts/evaluation" / f"error_analysis_{family}.py",
        ROOT / "scripts/evaluation/text_calibration_runtime.py",
        ROOT / "utils/transformers/calibration.py",
        ROOT / "utils/transformers/tokenizer_identity.py",
        ROOT / "utils/transformers/models/text_calibration.py",
        ROOT / "utils/transformers/models/spiking_ops.py",
        ROOT / "utils/transformers/optional_tensorboard.py",
    ]
    for directory in (
        ROOT / "utils/transforms", ROOT / "utils/transformers/integrations",
        ROOT / "utils/transformers/models" / f"spiking_{family}",
    ):
        paths.extend(directory.rglob("*.py"))
    # Generic text site discovery imports both encoder adapter families.
    if family in ("bert", "roberta"):
        other = "roberta" if family == "bert" else "bert"
        paths.extend((ROOT / "utils/transformers/models" / f"spiking_{other}").rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(paths))}


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_commands(args, output: Path) -> dict[str, list[str]]:
    task = "wikitext2" if args.family == "gpt2" else "sst2"
    common = [
        args.python_bin, "-u",
        str(ROOT / "scripts/evaluation" / f"error_analysis_{args.family}.py"),
        "--experiment_name", output.name, "--model_backend", "spiking",
        "--model_id", args.model_id, "--task", task,
        "--cache-dir", args.cache_dir, "--device", "cuda",
        "--dtype", "float64", "--theta", "40", "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length), "--no-tensorboard",
        "--calibration-samples", str(args.calibration_samples),
        "--calibration-seed", "0", "--calibration-bins", "2048",
        "--calibration-lower-quantile", "0", "--calibration-upper-quantile", "1",
        "--calibration-margin-fraction", "0.05",
        "--calibration-path", str(output / "calibration.json"),
        "--no-gaussian-time-noise", "--report-clamp-stats",
    ]
    return {
        "collect": [*common, "--calibration-mode", "collect"],
        "validate": [
            *common, "--calibration-mode", "validate",
            "--max_eval_batches", str(args.eval_batches),
        ],
    }


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("bert", "roberta", "gpt2"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default="/root/.cache/huggingface/datasets")
    parser.add_argument("--gpu", required=True, type=int, choices=(4, 5, 6, 7))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--calibration-samples", default=16, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--max-length", default=32, type=int)
    parser.add_argument("--eval-batches", default=2, type=int)
    args = parser.parse_args()
    if socket.gethostname() != "baekryun-cuda129":
        raise RuntimeError("this helper only runs on baekryun-cuda129")
    if not (1 <= args.calibration_samples <= 64 and 1 <= args.eval_batches <= 8):
        raise ValueError("this diagnostic is limited to 64 training samples and 8 evaluation batches")
    if not (1 <= args.batch_size <= 8 and 2 <= args.max_length <= 128):
        raise ValueError("this diagnostic requires batch size at most 8 and sequence length at most 128")
    output = args.output_dir.resolve()
    artifact_root = Path("/data/delayed-temporal/artifacts").resolve()
    if output == artifact_root or not output.is_relative_to(artifact_root) or output.exists():
        raise ValueError("output-dir must be a new dedicated path below artifacts")
    runtime_id = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    runtime = artifact_root / "runtime" / "text-calibration-smoke" / runtime_id
    if runtime.exists():
        raise FileExistsError(runtime)

    # Admission happens before importing torch, transformers or any evaluator.
    from scripts.runtime.local_gpu import gpu_activity, gpu_available
    if "torch" in sys.modules:
        raise RuntimeError("GPU admission must precede importing torch")
    lock_root = artifact_root / "runtime" / "gpu-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"gpu-{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        activity = gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
        if not gpu_available(activity):
            raise RuntimeError(f"GPU {args.gpu} is occupied")
        output.mkdir(parents=True, exist_ok=False)
        runtime.mkdir(parents=True, exist_ok=False)
        filesystem = subprocess.check_output(
            ["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True,
        ).strip()
        if not filesystem or any(kind in ("tmpfs", "ramfs") for kind in filesystem.splitlines()):
            raise RuntimeError("temporary files must use a disk filesystem")
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="disabled",
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false", TMPDIR=str(runtime), TMP=str(runtime),
            TEMP=str(runtime), WANDB_DIR=str(runtime),
        )
        commands = build_commands(args, output)
        hashes = source_hashes(args.family)
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"], text=True,
        )
        manifest = {
            "family": args.family, "model_id": args.model_id, "gpu": args.gpu,
            "source_commit": commit, "source_hashes": hashes,
            "tracked_checkout_clean": not bool(subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True,
            ).strip()),
            "worktree_status": dirty, "diagnostic_only": True, "paper_reuse_allowed": False,
            "commands": commands, "runtime_dir": str(runtime), "runtime_filesystem": filesystem,
            "gpu_admission": activity,
            "environment": {name: environment[name] for name in (
                "CUDA_VISIBLE_DEVICES", "WANDB_MODE", "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE", "TMPDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )},
        }
        write_new_json(output / "manifest.json", manifest)
        started = time.monotonic()
        result = {"state": "failed", "completed_phases": [], "diagnostic_only": True}
        child = None

        def interrupted(signum, _frame):
            raise InterruptedError(f"received signal {signum}")

        previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for phase, command in commands.items():
                if source_hashes(args.family) != hashes:
                    raise RuntimeError("source changed after the diagnostic manifest was written")
                phase_started = time.monotonic()
                print(f"Starting {args.family} {phase}; log={output / (phase + '.log')}", flush=True)
                with (output / (phase + ".log")).open("x") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lock.fileno(),),
                    )
                    write_new_json(output / (phase + "-started.json"), {"pid": child.pid, "phase": phase})
                    exit_code = child.wait()
                write_new_json(output / (phase + "-exit.json"), {
                    "phase": phase, "exit_code": exit_code,
                    "elapsed_seconds": time.monotonic() - phase_started,
                })
                if exit_code != 0:
                    raise RuntimeError(f"{phase} exited with status {exit_code}")
                log_text = (output / (phase + ".log")).read_text()
                if phase == "collect":
                    if not (output / "calibration.json").is_file() or "Saved calibration artifact" not in log_text:
                        raise RuntimeError("collection lacks its complete calibration output")
                elif ("Perplexity:" if args.family == "gpt2" else "Prediction SHA256:") not in log_text:
                    raise RuntimeError("evaluation lacks its final metric record")
                result["completed_phases"].append(phase)
            if source_hashes(args.family) != hashes:
                raise RuntimeError("source changed during the diagnostic")
            result["state"] = "complete"
            result["calibration_sha256"] = sha256_file(output / "calibration.json")
        except BaseException as error:
            if child is not None:
                stop_child(child)
            result["error"] = str(error)
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            result["elapsed_seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", result)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
