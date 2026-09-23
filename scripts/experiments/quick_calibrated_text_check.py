"""Run a short calibrated text comparison from an explicitly frozen checkout."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity

ARTIFACTS = Path("/data/delayed-temporal/artifacts")
SAMPLES = 256
BATCH_SIZE = 8
BATCHES = SAMPLES // BATCH_SIZE


def validate_gpu_request(gpu: int, hostname: str) -> None:
    if hostname != "baekryun-cuda129" or type(gpu) is not int or gpu not in (4, 5, 6, 7):
        raise ValueError("use one local GPU from 4 through 7")


def validate_disk_filesystem(filesystem: str) -> None:
    if not filesystem or any(kind in {"tmpfs", "ramfs"} for kind in filesystem.splitlines()):
        raise RuntimeError("temporary files must use a disk filesystem")


def check_source(source: Path, expected_commit: str) -> str:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("source HEAD differs from the full expected commit")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if dirty:
        raise ValueError("source must be a clean tracked checkout")
    return head


def source_hashes(source: Path, family: str) -> dict[str, str]:
    paths = list((source / "utils").rglob("*.py"))
    paths += [source / "scripts/evaluation" / f"error_analysis_{family}.py",
              source / "scripts/evaluation/text_calibration_runtime.py",
              source / "scripts/experiments/quick_calibrated_text_check.py",
              source / "scripts/runtime/files.py",
              source / "scripts/runtime/identity.py"]
    return {
        str(path.relative_to(source)): identity.sha256_file(path)
        for path in sorted(set(paths))
    }


def build_commands(args: argparse.Namespace, output: Path) -> dict[str, list[str]]:
    common = [
        args.python_bin, "-u", str(args.source_root / "scripts/evaluation" / f"error_analysis_{args.family}.py"),
        "--model_id", args.model_id, "--task", "wikitext2" if args.family == "gpt2" else "sst2",
        "--cache-dir", args.cache_dir, "--device", "cuda", "--dtype", "float64",
        "--batch_size", str(BATCH_SIZE), "--max_length", "128",
        "--no-tensorboard", "--no-gaussian-time-noise",
        "--spiking-layernorm", "--spiking-attention", "--spiking-mlp",
        "--spiking-ln-mul", "--spiking-ln-log", "--spiking-ln-expdiff",
        "--activation", "gelu_new" if args.family == "gpt2" else "gelu",
    ]
    if args.family == "gpt2":
        common += ["--tau-s", "1"]
    calibration = [
        "--calibration-path", str(output / "calibration.json"),
        "--calibration-samples", str(SAMPLES), "--calibration-seed", "0",
        "--calibration-bins", "2048", "--calibration-lower-quantile", "0",
        "--calibration-upper-quantile", "1", "--calibration-margin-fraction", "0.05",
    ]
    return {
        "collect": [*common, "--experiment_name", output.name + "_collect", "--model_backend", "spiking",
                    *calibration, "--calibration-mode", "collect"],
        "ann": [*common, "--experiment_name", output.name + "_ann", "--model_backend", "hf",
                "--calibration-mode", "none", "--max_eval_batches", str(BATCHES)],
        "snn": [*common, "--experiment_name", output.name + "_snn", "--model_backend", "spiking",
                *calibration, "--calibration-mode", "validate", "--max_eval_batches", str(BATCHES)],
    }


def calibration_sites(path: Path) -> tuple[dict[str, Any], set[str]]:
    table = json.loads(path.read_text())
    layers = table["layers"]
    sites = {row["module_name"] + "/" + row["tensor_name"] for row in layers}
    if not sites or len(sites) != len(layers):
        raise ValueError("calibration sites are empty or duplicated")
    metadata = table["metadata"]
    preprocessing = json.loads(metadata["preprocessing"])
    if preprocessing["subset_samples"] != SAMPLES or metadata["dtype"] != "float64":
        raise ValueError("calibration sample count or dtype differs from the pilot")
    if dict(metadata["model_options"])["text_calibration_policy_version"] != 1:
        raise ValueError("calibration must use complete text policy 1")
    return metadata, sites


def parse_evaluation(log: str, family: str, sites: set[str] | None = None) -> dict[str, Any]:
    fingerprints = re.findall(r"^Evaluation dataset fingerprint: (\S+)$", log, re.MULTILINE)
    if len(fingerprints) != 1:
        raise ValueError("evaluation dataset fingerprint is missing or duplicated")
    if family == "gpt2":
        records = []
        for match in re.finditer(r'\{"average_loss":.*?\}', log):
            row = json.loads(match.group())
            if row.get("event") == "evaluation_progress":
                records.append(row)
        if len(records) != BATCHES or [row["batch"] for row in records] != list(range(1, BATCHES + 1)):
            raise ValueError("GPT-2 progress is missing, duplicated or out of order")
        for index, row in enumerate(records, 1):
            if (row["total_batches"] != BATCHES or row["valid_loss_batches"] != index
                    or row["evaluated_samples"] != index * BATCH_SIZE):
                raise ValueError("GPT-2 loss or sample counts are incomplete")
            if any(row[key] is None or not math.isfinite(row[key]) for key in ("average_loss", "perplexity")):
                raise ValueError("GPT-2 metrics are not finite")
        final = records[-1]
        if len(re.findall(r"^Perplexity: ", log, re.MULTILINE)) != 1:
            raise ValueError("GPT-2 final metric is missing or duplicated")
        result = {key: final[key] for key in ("average_loss", "perplexity", "loss_aggregation")}
        result.update(total=SAMPLES, completed_batches=BATCHES)
    else:
        finals = re.findall(r"^Correct/total: (\d+)/(\d+)\s*$", log, re.MULTILINE)
        digests = re.findall(r"^Prediction SHA256: ([0-9a-f]{64})\s*$", log, re.MULTILINE)
        batches = re.findall(r"Evaluation progress: batch=(\d+)/(\d+) correct=(\d+) total=(\d+)/(\d+)", log)
        if len(finals) != 1 or len(digests) != 1 or len(fingerprints) != 1 or len(batches) != BATCHES:
            raise ValueError("classification output is incomplete or duplicated")
        correct, total = map(int, finals[0])
        if total != SAMPLES or not 0 <= correct <= total:
            raise ValueError("classification correct/total is invalid")
        for index, values in enumerate(batches, 1):
            batch, batch_count, batch_correct, count, expected = map(int, values)
            if (batch, batch_count, count, expected) != (index, BATCHES, index * BATCH_SIZE, SAMPLES):
                raise ValueError("classification sample or batch count differs")
            if not 0 <= batch_correct <= count:
                raise ValueError("classification progress count is invalid")
        if int(batches[-1][2]) != correct:
            raise ValueError("classification final count disagrees with progress")
        result = {"correct": correct, "total": total, "accuracy": correct / total,
                  "prediction_sha256": digests[0], "evaluation_dataset_fingerprint": fingerprints[0],
                  "completed_batches": BATCHES}
    result["evaluation_dataset_fingerprint"] = fingerprints[0]
    if sites is not None:
        reports = re.findall(r"^Calibration\[([^\]]+)\] values=(\d+),", log, re.MULTILINE)
        if len(reports) != len(sites) or {site for site, _ in reports} != sites:
            raise ValueError("calibration execution reports are missing or duplicated")
        if any(int(count) <= 0 for _, count in reports):
            raise ValueError("calibration site was not executed")
        result["calibration_site_count"] = len(sites)
    return result


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


# @lat: [[quick-family-checks#Quick Model Family Checks#Calibrated Text Comparison]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("bert", "roberta", "gpt2"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default="/root/.cache/huggingface/datasets")
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--gpu", required=True, type=int, choices=(4, 5, 6, 7))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    validate_gpu_request(args.gpu, socket.gethostname())
    args.source_root = args.source_root.resolve(strict=True)
    check_source(args.source_root, args.expected_commit)
    if Path(__file__).resolve() != args.source_root / "scripts/experiments/quick_calibrated_text_check.py":
        raise ValueError("run the committed helper from the selected source checkout")
    model = Path(args.model_id).resolve(strict=True)
    if not model.is_dir():
        raise ValueError("model-id must identify a cached local checkpoint directory")
    args.model_id = str(model)
    output = args.output_root.resolve()
    allowed_output = ARTIFACTS / "logs/quick_family_checks"
    if output == allowed_output or not output.is_relative_to(allowed_output) or output.exists():
        raise ValueError("output must be a new dedicated quick comparison artifact directory")
    runtime = ARTIFACTS / "runtime/quick_family_checks" / hashlib.sha256(str(output).encode()).hexdigest()[:16]
    if runtime.exists():
        raise FileExistsError(runtime)
    sys.path.insert(0, str(args.source_root))
    from scripts.runtime.local_gpu import gpu_activity, gpu_available
    if "torch" in sys.modules:
        raise RuntimeError("GPU admission must precede importing torch")
    locks = ARTIFACTS / "runtime/gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"gpu-{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        activity = gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
        if not gpu_available(activity):
            raise RuntimeError(f"GPU {args.gpu} is occupied")
        output.mkdir(parents=True, exist_ok=False)
        runtime.mkdir(parents=True, exist_ok=False)
        filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
        validate_disk_filesystem(filesystem)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="disabled",
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            TMPDIR=str(runtime), TMP=str(runtime), TEMP=str(runtime), WANDB_DIR=str(runtime),
            PYTHONPATH=os.pathsep.join(map(str, (args.source_root, args.source_root / "src/transformers/src",
                                                args.source_root / "src/spikingjelly"))))
        commands = build_commands(args, output)
        hashes = source_hashes(args.source_root, args.family)
        checkpoint_hashes = {
            str(path.relative_to(model)): identity.sha256_file(path)
            for path in sorted(model.rglob("*"))
                             if path.is_file()}
        runtime_files.new_json(output / "manifest.json", {
            "family": args.family, "model_id": args.model_id, "gpu": args.gpu, "gpu_admission": activity,
            "source_root": str(args.source_root), "source_commit": args.expected_commit, "source_hashes": hashes,
            "checkpoint_files_sha256": checkpoint_hashes, "commands": commands,
            "calibration_samples": SAMPLES, "evaluation_samples": SAMPLES, "batch_size": BATCH_SIZE,
            "evaluation_order": "first 256 in the existing evaluation split after evaluator filtering",
            "evaluation_split": "test" if args.family == "gpt2" else "validation",
            "calibration_split": "train", "calibration_order": "existing seeded training subset",
            "text_filter": "nonempty stripped text" if args.family == "gpt2" else "none",
            "tokenizer_model_id": args.model_id, "padding": "max_length", "truncation": True,
            "max_length": 128, "range_contract": "operator_local_v1",
            "dtype": "float64", "calibration_seed": 0,
            "diagnostic_only": True, "paper_reuse_allowed": False, "runtime_dir": str(runtime),
            "runtime_filesystem": filesystem, "wandb_mode": "disabled", "tensorboard": False,
        })
        started = time.monotonic()
        result: dict[str, Any] = {"state": "failed", "completed_phases": [], "metrics": {}, "diagnostic_only": True}
        child = None

        def interrupted(signum, _frame):
            raise InterruptedError(f"received signal {signum}")

        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            sites: set[str] | None = None
            for phase, command in commands.items():
                check_source(args.source_root, args.expected_commit)
                if source_hashes(args.source_root, args.family) != hashes:
                    raise RuntimeError("source changed after manifest creation")
                phase_started = time.monotonic()
                log_path = output / f"{phase}.log"
                print(json.dumps({"family": args.family, "phase": phase, "state": "started", "log": str(log_path)}), flush=True)
                with log_path.open("x") as log:
                    child = subprocess.Popen(command, cwd=args.source_root, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                    runtime_files.new_json(
                        output / f"{phase}-started.json", {"phase": phase, "pid": child.pid}
                    )
                    returncode = child.wait()
                phase_result = {"phase": phase, "returncode": returncode,
                                "elapsed_seconds": time.monotonic() - phase_started,
                                "log_sha256": identity.sha256_file(log_path)}
                runtime_files.new_json(output / f"{phase}-exit.json", phase_result)
                if returncode:
                    raise RuntimeError(f"{phase} exited with status {returncode}; logs preserved")
                log_text = log_path.read_text()
                if phase == "collect":
                    if len(re.findall(r"Saved calibration artifact", log_text)) != 1:
                        raise ValueError("collection completion is missing or duplicated")
                    metadata, sites = calibration_sites(output / "calibration.json")
                    result["calibration_metadata"] = metadata
                    result["calibration_site_count"] = len(sites)
                    result["calibration_sha256"] = identity.sha256_file(
                        output / "calibration.json"
                    )
                else:
                    metrics = parse_evaluation(log_text, args.family, sites if phase == "snn" else None)
                    result["metrics"][phase] = metrics
                    runtime_files.new_json(output / f"{phase}-metrics.json", metrics)
                result["completed_phases"].append(phase)
                print(json.dumps(phase_result, sort_keys=True), flush=True)
            ann, snn = result["metrics"]["ann"], result["metrics"]["snn"]
            if ann["evaluation_dataset_fingerprint"] != snn["evaluation_dataset_fingerprint"]:
                raise ValueError("ANN and SNN evaluation dataset identities differ")
            if args.family != "gpt2":
                result["accuracy_difference"] = snn["accuracy"] - ann["accuracy"]
            check_source(args.source_root, args.expected_commit)
            if source_hashes(args.source_root, args.family) != hashes:
                raise RuntimeError("source changed during evaluation")
            result["state"] = "complete"
        except BaseException as error:
            if child is not None:
                stop_child(child)
            result["error"] = str(error)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            result["elapsed_seconds"] = time.monotonic() - started
            runtime_files.new_json(output / "result.json", result)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
