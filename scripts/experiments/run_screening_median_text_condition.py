#!/usr/bin/env python3
"""Run one authenticated RoBERTa or GPT-2 screening-median replica."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_full_calibrated_text_comparison import (
    parse_evaluation,
    parse_sites,
    source_identity,
)
from scripts.experiments.run_vit_local_range_noise_condition import parse_physical_counts
from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    canonical_alpha,
    load_screening_median,
    scaled_fractions,
)
from scripts.experiments.screening_median_text_sweep import (
    TEXT_EVALUATION_SAMPLES,
    TEXT_MODELS,
    TEXT_SMOKE_SAMPLES,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


ARTIFACTS = Path(
    os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts")
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    """Load and authenticate the immutable text sweep protocol."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or not isinstance(payload.get("identity"), dict)
        or payload.get("protocol_id") != identity.json_sha256(payload["identity"])
    ):
        raise ValueError("text screening-median protocol identity differs")
    return resolved, payload


def completed_preparation(
    resource: dict[str, Any], expected_commit: str
) -> tuple[Path, set[str]]:
    """Return the frozen calibration and its executed site population."""

    root = Path(resource["preparation_root"]).resolve(strict=True)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    calibration = (root / "calibration.json").resolve(strict=True)
    calibration_sha256 = identity.sha256_file(calibration)
    family = resource["evaluator_family"]
    if (
        result.get("state") != "complete"
        or manifest.get("source_commit") != expected_commit
        or manifest.get("evaluator_family") != family
        or result.get("calibration_sha256") != calibration_sha256
        or resource.get("calibration_sha256") != calibration_sha256
    ):
        raise ValueError("text calibration source identity differs")
    _, sites = parse_sites(calibration, int(resource["calibration_site_count"]))
    return calibration, sites


def evaluator_command(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    resource: dict[str, Any],
    calibration: Path,
    *,
    linear_fraction: float,
    log_fraction: float,
    run_id: str,
) -> list[str]:
    """Build the complete frozen text-model evaluation command."""

    source_root = Path(protocol["resources"]["source_root"])
    family = resource["evaluator_family"]
    command = [
        args.python_bin,
        "-u",
        str(source_root / "scripts/evaluation" / f"error_analysis_{family}.py"),
        "--experiment_name",
        run_id,
        "--model_backend",
        "spiking",
        "--model_id",
        resource["model_id"],
        "--task",
        resource["task"],
        "--cache-dir",
        resource["cache_dir"],
        "--device",
        "cuda",
        "--dtype",
        "float64",
        "--batch_size",
        "8",
        "--max_length",
        "128",
        "--max_eval_batches",
        "8" if args.phase == "smoke" else "0",
        "--no-tensorboard",
        "--report-clamp-stats",
        "--spiking-layernorm",
        "--spiking-attention",
        "--spiking-mlp",
        "--spiking-ln-mul",
        "--spiking-ln-log",
        "--spiking-ln-expdiff",
        "--activation",
        resource["activation"],
        "--calibration-mode",
        "validate",
        "--calibration-path",
        str(calibration),
        "--calibration-samples",
        "5000",
        "--calibration-seed",
        "0",
        "--calibration-bins",
        "2048",
        "--calibration-lower-quantile",
        "0",
        "--calibration-upper-quantile",
        "1",
        "--calibration-margin-fraction",
        "0.05",
        "--calibration-dataset-path",
        resource["calibration_dataset_path"],
        "--calibration-dataset-fingerprint",
        resource["calibration_dataset_fingerprint"],
        "--evaluation-dataset-path",
        resource["evaluation_dataset_path"],
        "--evaluation-dataset-fingerprint",
        resource["evaluation_dataset_fingerprint"],
        "--gaussian-time-noise",
        "--time-noise-std-frac",
        "0",
        "--linear-time-noise-std-frac",
        repr(linear_fraction),
        "--log-time-noise-std-frac",
        repr(log_fraction),
        "--time-noise-deadline-margin-std",
        repr(DEADLINE_MARGIN_SIGMA_RATIO),
        "--time-noise-mean",
        "0",
        "--time-noise-seed",
        str(args.seed),
    ]
    if family == "gpt2":
        command += ["--tau-s", "1"]
    return command


def environment(source_root: Path, runtime: Path, gpu: int) -> dict[str, str]:
    """Build the single-GPU offline evaluator environment."""

    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TMPDIR=str(runtime),
        TMP=str(runtime),
        TEMP=str(runtime),
        WANDB_DIR=str(runtime),
        SOURCE_ROOT=str(source_root),
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                (
                    source_root,
                    source_root / "src/transformers/src",
                    source_root / "src/spikingjelly",
                ),
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model", choices=TEXT_MODELS, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--evaluation-samples", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    expected_samples = (
        TEXT_SMOKE_SAMPLES if args.phase == "smoke" else TEXT_EVALUATION_SAMPLES
    )[args.model]
    if args.evaluation_samples != expected_samples:
        raise ValueError("text evaluation sample count differs from the fixed protocol")

    protocol_path, protocol = load_protocol(args.protocol)
    alpha = canonical_alpha(args.alpha)
    pair = load_screening_median(Path(protocol["resources"]["hardware_summary"]))
    if pair.summary_sha256 != protocol["identity"]["hardware"]["summary_sha256"]:
        raise ValueError("protocol and hardware summary identities differ")
    linear_fraction, log_fraction = scaled_fractions(pair, alpha)
    resource = protocol["resources"]["models"][args.model]
    expected_commit = protocol["identity"]["source_commit"]
    calibration, sites = completed_preparation(resource, expected_commit)
    source_root = Path(protocol["resources"]["source_root"]).resolve(strict=True)
    source_hashes = source_identity(
        source_root, expected_commit, resource["evaluator_family"]
    )
    for path in (
        Path(__file__).resolve(),
        source_root / "scripts/experiments/screening_median_text_sweep.py",
    ):
        source_hashes[str(path.relative_to(source_root))] = identity.sha256_file(path)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True
    ).strip() in {"tmpfs", "ramfs"}:
        raise RuntimeError("text condition runtime must use a disk filesystem")

    run_id = f"{args.model}_alpha_{alpha.replace('.', 'p')}_seed_{args.seed}"
    command = evaluator_command(
        args,
        protocol,
        resource,
        calibration,
        linear_fraction=linear_fraction,
        log_fraction=log_fraction,
        run_id=run_id,
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": identity.sha256_file(protocol_path),
        "phase": args.phase,
        "model": args.model,
        "alpha": alpha,
        "seed": args.seed,
        "evaluation_samples": args.evaluation_samples,
        "linear_time_noise_std_fraction": linear_fraction,
        "log_time_noise_std_fraction": log_fraction,
        "deadline_margin_sigma_ratio": DEADLINE_MARGIN_SIGMA_RATIO,
        "time_noise_scope": "all_encoder_outputs",
        "hardware_summary_sha256": pair.summary_sha256,
        "source_commit": expected_commit,
        "source_hashes": source_hashes,
        "calibration_sha256": resource["calibration_sha256"],
        "checkpoint_sha256": resource["checkpoint_sha256"],
        "physical_gpu": args.gpu,
        "command": command,
    }
    runtime_files.immutable_json(args.output_dir / "manifest.json", manifest)
    result_path = args.output_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("state") == "complete":
            print(canonical(existing), flush=True)
            return

    lock_path = ARTIFACTS / "runtime/gpu-locks" / f"gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        samples = []
        for check in range(2):
            sample = local_gpu.gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
            samples.append(sample)
            if not local_gpu.gpu_available(sample):
                raise RuntimeError(f"GPU {args.gpu} is occupied")
            if check == 0:
                time.sleep(10)
        attempt = len(list(args.output_dir.joinpath("logs").glob("evaluation.attempt-*.log"))) + 1
        log_path = args.output_dir / "logs" / f"evaluation.attempt-{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_files.atomic_json(
            result_path,
            {"state": "running", "run_id": run_id, "admission": samples},
        )
        started = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=source_root,
                env=environment(source_root, runtime, args.gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            runtime_files.atomic_json(
                result_path,
                {
                    "state": "failed",
                    "run_id": run_id,
                    "exit_status": completed.returncode,
                    "log_file": str(log_path.relative_to(args.output_dir)),
                    "admission": samples,
                },
            )
            raise RuntimeError(f"text evaluator exited with status {completed.returncode}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_evaluation(
        log_text,
        resource["evaluator_family"],
        sites,
        expected_samples=args.evaluation_samples,
    )
    if int(metrics["total"]) != args.evaluation_samples:
        raise ValueError("text evaluator sample count differs")
    metrics["physical_counts"] = parse_physical_counts(log_text)
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key in {"accuracy", "token_weighted_loss", "token_weighted_perplexity"}
    ):
        raise ValueError("text task metric is non-finite")
    result = {
        "state": "complete",
        "run_id": run_id,
        "elapsed_seconds": time.monotonic() - started,
        "log_file": str(log_path.relative_to(args.output_dir)),
        "log_sha256": identity.sha256_file(log_path),
        "metrics": metrics,
        "admission": samples,
    }
    runtime_files.atomic_json(result_path, result)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
