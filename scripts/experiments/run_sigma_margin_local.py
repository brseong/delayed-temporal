#!/usr/bin/env python3
"""Run a sigma/deadline-margin manifest on local physical GPUs 4--7.

The scheduler keeps one evaluator process on each GPU and lets a worker claim
the next manifest row as soon as its previous row finishes.  Completed logs are
validated against the immutable full manifest before they are reused.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import files as runtime_files


ALLOWED_GPUS = ("4", "5", "6", "7")
CORRECT_RE = re.compile(r"^Correct: (?P<value>\d+)$", re.MULTILINE)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/opt/conda/envs/dt/bin/python"))
    parser.add_argument("--gpus", nargs="+", default=list(ALLOWED_GPUS))
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--reference-log-dir", type=Path)
    parser.add_argument("--max-accuracy-delta", type=float, default=0.01)
    parser.add_argument("--allow-noncanonical-manifest", action="store_true")
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=Path("scripts/evaluation/error_analysis_vit.py"),
    )
    parser.add_argument("--evaluator-arg", action="append", default=[])
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    if not rows:
        raise ValueError(f"empty task manifest: {path}")
    return rows


def live_compute_pids(gpu: str) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value.isdigit() and Path(f"/proc/{value}").exists():
            pids.append(int(value))
    return pids


class LocalScheduler:
    def __init__(self, args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
        self.args = args
        self.rows: queue.Queue[dict[str, str]] = queue.Queue()
        for row in rows:
            self.rows.put(row)
        self.total = len(rows)
        self.started_at = time.time()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.running: dict[str, str] = {}
        self.completed: list[str] = []
        self.skipped: list[str] = []
        self.failed: dict[str, str] = {}
        self.elapsed: list[float] = []
        self.children: dict[str, subprocess.Popen[str]] = {}

    def write_status(self) -> None:
        if self.args.status_json is None:
            return
        with self.lock:
            elapsed_seconds = time.time() - self.started_at
            mean_seconds = sum(self.elapsed) / len(self.elapsed) if self.elapsed else None
            remaining = self.rows.qsize() + len(self.running)
            eta_seconds = (
                mean_seconds * remaining / len(self.args.gpus)
                if mean_seconds is not None else None
            )
            payload = {
                "format_version": 1,
                "status": "failed" if self.failed else ("running" if remaining else "complete"),
                "gpus": self.args.gpus,
                "total": self.total,
                "completed": len(self.completed),
                "skipped": len(self.skipped),
                "failed": self.failed,
                "running": self.running,
                "queued": self.rows.qsize(),
                "elapsed_seconds": elapsed_seconds,
                "mean_run_seconds": mean_seconds,
                "estimated_remaining_seconds": eta_seconds,
            }
            runtime_files.atomic_json(self.args.status_json, payload)

    def validate(self, run_id: str) -> bool:
        command = [
                str(self.args.python_bin),
                str(self.args.source_root / "scripts/analysis/summarize_sigma_margin_sweep.py"),
                "--manifest", str(self.args.manifest),
                "--log-dir", str(self.args.log_dir),
                "--check-run-id", run_id,
        ]
        if self.args.allow_noncanonical_manifest:
            command.append("--allow-noncanonical-manifest")
        result = subprocess.run(
            command,
            cwd=self.args.source_root,
            env=self.base_environment(),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def base_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": ":".join((
                str(self.args.source_root),
                str(self.args.source_root / "src/transformers/src"),
                str(self.args.source_root / "src/spikingjelly"),
            )),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "WANDB_CONSOLE": "off",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        })
        environment.pop("WANDB_API_KEY", None)
        return environment

    def command(self, row: dict[str, str]) -> list[str]:
        gaussian = ["--no-gaussian-time-noise", "--time-noise-seed", "0"]
        if row["stage"] == "sigma_margin":
            gaussian = ["--gaussian-time-noise", "--time-noise-seed", row["seed"]]
        evaluator_script = self.args.evaluator_script
        if not evaluator_script.is_absolute():
            evaluator_script = self.args.source_root / evaluator_script
        return [
            str(self.args.python_bin),
            str(evaluator_script),
            *self.args.evaluator_arg,
            "--experiment_name", row["run_id"],
            "--device", "cuda",
            "--model_backend", row["backend"],
            "--model_id", row["checkpoint_path"],
            "--dataset_id", "imagenet-1k",
            "--evaluation-dataset-path", row["dataset_path"],
            "--evaluation-split", row["split"],
            "--batch_size", "32",
            "--quick-test",
            "--theta", row["theta"],
            "--precision", row["precision"],
            "--calibration-mode", "none",
            *gaussian,
            "--time-noise-std-frac", row["time_noise_std_frac"],
            "--time-noise-mean", "0",
            "--time-noise-deadline-margin-std", row["deadline_margin_std"],
            "--no-mismatch-enabled",
            "--mismatch-theta-std", "0",
            "--weight-noise-std", "0",
            "--bias-noise-std", "0",
            "--source-commit", row["source_commit"],
            "--checkpoint-sha256", row["checkpoint_sha256"],
            "--no-tensorboard",
            "--report-clamp-stats",
            "--spiking-layernorm",
            "--spiking-mlp",
            "--spiking-attention",
        ]

    def run_one(self, gpu: str, row: dict[str, str]) -> None:
        run_id = row["run_id"]
        log_path = self.args.log_dir / row["log_file"]
        if self.validate(run_id):
            with self.lock:
                self.skipped.append(run_id)
            return
        if log_path.exists():
            rejected = log_path.with_name(f"{log_path.name}.rejected.{int(time.time())}")
            log_path.replace(rejected)
        pids = live_compute_pids(gpu)
        if pids:
            raise RuntimeError(f"GPU {gpu} became occupied by live PIDs {pids}")

        runtime_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=self.args.runtime_root))
        temporary_log = log_path.with_name(f"{log_path.name}.partial.{os.getpid()}.{gpu}")
        environment = self.base_environment()
        environment.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "TMPDIR": str(runtime_dir),
            "XDG_CACHE_HOME": str(runtime_dir / "cache"),
            "HF_HOME": str(runtime_dir / "huggingface"),
        })
        start = time.time()
        try:
            with temporary_log.open("w", encoding="utf-8") as handle:
                handle.write(
                    f"Slurm identity — job_id: local-{os.getpid()}, task_id: {run_id}, "
                    f"node: {os.uname().nodename}, gpu_family: {row['gpu_family']}\n"
                )
                handle.flush()
                child = subprocess.Popen(
                    self.command(row),
                    cwd=self.args.source_root,
                    env=environment,
                    text=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                with self.lock:
                    self.children[gpu] = child
                return_code = child.wait()
            with self.lock:
                self.children.pop(gpu, None)
            if return_code != 0:
                raise RuntimeError(f"evaluator exited with status {return_code}")
            temporary_log.replace(log_path)
            if not self.validate(run_id):
                failed_path = log_path.with_name(f"{log_path.name}.failed.{int(time.time())}")
                log_path.replace(failed_path)
                raise RuntimeError("completed output failed manifest validation")
            with self.lock:
                self.completed.append(run_id)
                self.elapsed.append(time.time() - start)
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    def worker(self, gpu: str) -> None:
        while not self.stop.is_set():
            try:
                row = self.rows.get_nowait()
            except queue.Empty:
                return
            run_id = row["run_id"]
            with self.lock:
                self.running[gpu] = run_id
            self.write_status()
            try:
                self.run_one(gpu, row)
            except Exception as error:  # preserve failure details and stop the campaign
                with self.lock:
                    self.failed[run_id] = f"{type(error).__name__}: {error}"
                self.stop.set()
            finally:
                with self.lock:
                    self.running.pop(gpu, None)
                self.rows.task_done()
                self.write_status()

    def terminate(self) -> None:
        self.stop.set()
        with self.lock:
            children = list(self.children.values())
        for child in children:
            child.terminate()

    def execute(self) -> None:
        self.args.log_dir.mkdir(parents=True, exist_ok=True)
        self.args.runtime_root.mkdir(parents=True, exist_ok=True)
        self.write_status()
        threads = [threading.Thread(target=self.worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in self.args.gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.write_status()
        if self.failed:
            raise RuntimeError(f"local campaign stopped after failure: {self.failed}")


def compare_reference(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    if args.reference_log_dir is None:
        return
    failures = []
    for row in rows:
        current = (args.log_dir / row["log_file"]).read_text(encoding="utf-8")
        reference = (args.reference_log_dir / row["log_file"]).read_text(encoding="utf-8")
        current_correct = int(CORRECT_RE.search(current).group("value"))  # type: ignore[union-attr]
        reference_correct = int(CORRECT_RE.search(reference).group("value"))  # type: ignore[union-attr]
        delta = abs(current_correct - reference_correct) / int(row["expected_samples"])
        print(f"comparison\t{row['run_id']}\t{reference_correct}\t{current_correct}\t{delta:.6f}")
        if delta > args.max_accuracy_delta:
            failures.append((row["run_id"], delta))
    if failures:
        raise RuntimeError(f"cross-code accuracy delta exceeds gate: {failures}")


def main() -> None:
    args = parse_arguments()
    if tuple(args.gpus) != ALLOWED_GPUS:
        raise ValueError(f"local sigma-margin runner requires GPUs {ALLOWED_GPUS}, got {tuple(args.gpus)}")
    if subprocess.run(["git", "-C", str(args.source_root), "diff", "--quiet"]).returncode != 0:
        raise ValueError("source worktree has tracked modifications")
    full_rows = read_tsv(args.manifest)
    by_id = {row["run_id"]: row for row in full_rows}
    task_rows = read_tsv(args.task_manifest)
    if any(by_id.get(row["run_id"]) != row for row in task_rows):
        raise ValueError("task manifest is not an exact subset of the full manifest")
    for gpu in args.gpus:
        pids = live_compute_pids(gpu)
        if pids:
            raise RuntimeError(f"GPU {gpu} is occupied by live PIDs {pids}")

    scheduler = LocalScheduler(args, task_rows)
    signal.signal(signal.SIGTERM, lambda *_: scheduler.terminate())
    signal.signal(signal.SIGINT, lambda *_: scheduler.terminate())
    scheduler.execute()
    compare_reference(args, task_rows)


if __name__ == "__main__":
    main()
