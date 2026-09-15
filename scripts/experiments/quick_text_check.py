"""Run a frozen text evaluator with local progress output and no tracking files."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


class NoOpWriter:
    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False), flush=True)


class Progress:
    """Observe existing metrics without replacing the evaluator's final metric."""

    def __init__(self):
        self.started = time.monotonic()
        self.batches = 0
        self.loss_sum = 0.0
        self.final_seen = False

    def log(self, metrics: dict, *args, **kwargs):
        values = {}
        for key, value in metrics.items():
            if key.startswith(("Batch ", "Final ")):
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Nonfinite evaluator metric: {key}")
                values[key] = number
        if not values:
            return
        if "Batch Accuracy" in values or "Batch Loss" in values:
            self.batches += 1
        if "Batch Loss" in values:
            self.loss_sum += values["Batch Loss"]
            mean = self.loss_sum / self.batches
            values["Cumulative Average Loss"] = mean
            values["Cumulative Perplexity"] = math.exp(mean)
        if any(key.startswith("Final ") for key in values):
            self.final_seen = True
        emit("quick_progress", completed_batches=self.batches,
             elapsed_seconds=time.monotonic() - self.started, **values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("bert", "gpt2"))
    parser.add_argument("--backend", required=True, choices=("hf", "spiking"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.evaluator_args[1:] if args.evaluator_args[:1] == ["--"] else args.evaluator_args
    source, runtime = args.source_root.resolve(strict=True), args.runtime_dir.resolve()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if socket.gethostname() != "baekryun-cuda129" or visible not in {"4", "5", "6", "7"}:
        raise ValueError("Use exactly one idle local GPU from 4 through 7")
    if runtime == Path("/data") or not runtime.is_relative_to("/data"):
        raise ValueError("Runtime must be a dedicated directory on /data")
    runtime.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
    if not filesystem or any(kind in {"tmpfs", "ramfs"} for kind in filesystem.splitlines()):
        raise ValueError("Runtime must use a disk filesystem")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError("Evaluator source must be a clean tracked checkout")
    for name, value in {"WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                        "TMPDIR": str(runtime), "TMP": str(runtime), "TEMP": str(runtime),
                        "WANDB_DIR": str(runtime), "PYTHONUNBUFFERED": "1"}.items():
        os.environ[name] = value
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    sys.path.insert(0, str(source))
    from scripts.experiments.run_calibrated_three_sweeps import gpu_activity, gpu_available
    if "torch" in sys.modules:
        raise RuntimeError("GPU admission must precede importing torch")
    lock_dir = Path("/data/delayed-temporal/artifacts/runtime/gpu-locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"gpu-{visible}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not gpu_available(gpu_activity()[int(visible)]):
            raise RuntimeError("Assigned local GPU is occupied")
        evaluator_path = source / "scripts" / "evaluation" / f"error_analysis_{args.family}.py"
        evaluator_args = [*forwarded, "--model_backend", args.backend,
                          "--experiment_name", args.experiment_name, "--device", "cuda"]
        emit("quick_start", source_root=str(source), source_commit=head,
             helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             evaluator_sha256=hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
             evaluator_args=evaluator_args, gpu=int(visible), runtime_dir=str(runtime),
             family=args.family, backend=args.backend)
        os.chdir(runtime)
        module = importlib.import_module(f"scripts.evaluation.error_analysis_{args.family}")
        if Path(module.__file__).resolve() != evaluator_path.resolve():
            raise RuntimeError("Imported evaluator differs from selected source")
        progress = Progress()
        module.create_summary_writer = lambda **kwargs: NoOpWriter()
        module.wandb = SimpleNamespace(init=lambda **kwargs: None, log=progress.log, finish=lambda: None)
        import torch

        def check_output(_module, _inputs, output):
            # GPT-2's historical evaluator skips NaN losses; reject them before it can.
            loss = getattr(output, "loss", None)
            logits = getattr(output, "logits", None)
            for name, value in (("loss", loss), ("logits", logits)):
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Nonfinite model output: {name}")

        hook = torch.nn.modules.module.register_module_forward_hook(check_output)
        try:
            sys.argv = [str(evaluator_path), *evaluator_args]
            parsed = module.parse_arguments()
            getattr(module, f"evaluate_{args.family}_model")(parsed)
            if not progress.final_seen or not progress.batches:
                raise RuntimeError("Evaluator did not produce complete batch and final metrics")
            emit("quick_done", completed_batches=progress.batches,
                 elapsed_seconds=time.monotonic() - progress.started)
        finally:
            hook.remove()


if __name__ == "__main__":
    main()
