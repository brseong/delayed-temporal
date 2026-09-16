#!/usr/bin/env python3
"""Run the three ImageNet ViTs with their frozen timm evaluation transform."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


TAG = "conversion_comparison_imagenet_timm_theta40_float64_v4"
MODEL_KEYS = ("imagenet_vit_small", "imagenet_vit_base", "imagenet_vit_large")
SOURCE = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/conda/envs/dt/bin/python")
ROOT = Path("/data/delayed-temporal/artifacts/logs/conversion_comparison") / TAG
V3_ROOT = Path("/data/delayed-temporal/artifacts/logs/conversion_comparison/conversion_comparison_theta40_calibrated_float64_bounds3_v3")
PREPROCESSING = SOURCE / "scripts/configs/vit_timm_preprocessing.json"
PHASES = ("collect", "dense", "spiking")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_immutable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"refusing to replace immutable artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


def source_commit() -> str:
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"], text=True,
    )
    if dirty.strip():
        raise ValueError("timm recheck requires a clean frozen source checkout")
    return commit


def build_experiment() -> dict[str, Any]:
    previous_path = V3_ROOT / "experiment.json"
    previous = read_json(previous_path)
    models = []
    for key in MODEL_KEYS:
        matches = [row for row in previous["models"] if row["model_key"] == key]
        if len(matches) != 1:
            raise ValueError(f"v3 evidence does not contain one {key} row")
        row = matches[0]
        models.append({name: row[name] for name in (
            "model_key", "task", "checkpoint_path", "checkpoint_sha256",
            "calibration_dataset_path", "calibration_dataset_fingerprint",
            "calibration_dataset_sha256", "dataset_path", "dataset_fingerprint",
            "dataset_sha256", "evaluation_split", "expected_samples", "checkpoint_config",
        )})
    experiment = {
        "tag": TAG,
        "source_root": str(SOURCE),
        "source_commit": source_commit(),
        "python_bin": str(PYTHON),
        "theta": 40.0,
        "precision": "float64",
        "tau_s": 1.0,
        "batch_size": 32,
        "tracking": "disabled",
        "evaluation_population": "fixed_validation_5000",
        "calibration_population": "training_seed0_5000",
        "preprocessing_backend": "timm",
        "preprocessing_config": str(PREPROCESSING),
        "preprocessing_config_sha256": sha256_file(PREPROCESSING),
        "v3_experiment": str(previous_path),
        "v3_experiment_sha256": sha256_file(previous_path),
        "evaluator_sha256": sha256_file(SOURCE / "scripts/evaluation/error_analysis_vit.py"),
        "calibration_evaluator_sha256": sha256_file(SOURCE / "scripts/analysis/evaluate_calibrated_vit.py"),
        "gelu_evaluator_sha256": sha256_file(SOURCE / "scripts/analysis/gelu_cubic_phi_nl_vit.py"),
        "models": models,
    }
    return experiment


def validate_experiment(experiment: dict[str, Any]) -> None:
    expected = build_experiment()
    if experiment != expected:
        raise ValueError("stored v4 experiment differs from the current frozen identities")
    if {row["model_key"] for row in experiment["models"]} != set(MODEL_KEYS):
        raise ValueError("v4 requires exactly the three ImageNet ViTs")


def initialize() -> dict[str, Any]:
    experiment = build_experiment()
    ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("calibration", "logs", "results", "locks", "status", "rejected", "outputs"):
        (ROOT / directory).mkdir(exist_ok=True)
    write_immutable(ROOT / "experiment.json", experiment)
    validate_experiment(experiment)
    return experiment


def model_row(experiment: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [row for row in experiment["models"] if row["model_key"] == key]
    if len(matches) != 1:
        raise ValueError(f"unknown model key: {key}")
    return matches[0]


def common_arguments(experiment: dict[str, Any], model: dict[str, Any], phase: str) -> list[str]:
    backend = "hf" if phase == "dense" else "spiking"
    arguments = [
        "--experiment_name", f"{model['model_key']}_{phase}",
        "--device", "cuda", "--model_backend", backend,
        "--model_id", model["checkpoint_path"], "--dataset_id", "imagenet-1k",
        "--evaluation-dataset-path", model["dataset_path"],
        "--evaluation-split", model["evaluation_split"],
        "--image-preprocessing-config", experiment["preprocessing_config"],
        "--batch_size", "32", "--theta", "40", "--precision", "float64",
        "--source-commit", experiment["source_commit"],
        "--checkpoint-sha256", model["checkpoint_sha256"],
        "--no-tensorboard", "--report-clamp-stats", "--quick-test",
        "--spiking-layernorm", "--spiking-ln-mul", "--spiking-ln-log",
        "--spiking-ln-expdiff", "--spiking-attention", "--spiking-mlp",
        "--no-spiking-mlp-exact-gelu", "--no-gaussian-time-noise",
        "--time-noise-seed", "0", "--time-noise-std-frac", "0",
        "--time-noise-mean", "0", "--time-noise-deadline-margin-std", "0",
        "--no-mismatch-enabled", "--mismatch-theta-std", "0", "--mismatch-seed", "0",
        "--weight-noise-std", "0", "--bias-noise-std", "0",
    ]
    if phase == "dense":
        arguments += ["--calibration-mode", "none"]
    else:
        calibration = ROOT / "calibration" / f"{model['model_key']}.json"
        arguments += [
            "--calibration-mode", "collect" if phase == "collect" else "validate",
            "--calibration-path", str(calibration), "--calibration-samples", "5000",
            "--calibration-seed", "0", "--calibration-bins", "2048",
            "--calibration-lower-quantile", "0", "--calibration-upper-quantile", "1",
            "--calibration-margin-fraction", "0.05",
        ]
    return arguments


def command(experiment: dict[str, Any], model: dict[str, Any], phase: str) -> list[str]:
    arguments = common_arguments(experiment, model, phase)
    if phase == "dense":
        return [str(PYTHON), "-u", str(SOURCE / "scripts/evaluation/error_analysis_vit.py"), *arguments]
    return [
        str(PYTHON), "-u", str(SOURCE / "scripts/analysis/evaluate_calibrated_vit.py"),
        "--source-root", str(SOURCE),
        "--calibration-dataset-path", model["calibration_dataset_path"],
        "--calibration-dataset-fingerprint", model["calibration_dataset_fingerprint"],
        "--gelu-cubic-implementation", "phi_nl_psi_ed", "--gelu-cubic-floor", "1e-5",
        *arguments,
    ]


def single(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def validate_calibration(experiment: dict[str, Any], model: dict[str, Any], path: Path) -> dict[str, Any]:
    table = read_json(path)
    metadata = table["metadata"]
    preprocessing = json.loads(metadata["preprocessing"])
    if preprocessing.get("preprocessing_backend") != "timm":
        raise ValueError("calibration did not use timm preprocessing")
    if preprocessing.get("preprocessing_config_sha256") != experiment["preprocessing_config_sha256"]:
        raise ValueError("calibration preprocessing hash differs")
    if preprocessing.get("subset_fingerprint") != model["calibration_dataset_fingerprint"]:
        raise ValueError("calibration training population differs")
    if preprocessing.get("subset_samples") != 5000 or preprocessing.get("subset_seed") != 0:
        raise ValueError("calibration sampling contract differs")
    if metadata.get("model_id") != model["checkpoint_path"] or metadata.get("theta") != 40.0:
        raise ValueError("calibration model or threshold differs")
    expected_sites = 217 if model["model_key"] == "imagenet_vit_large" else 109
    if len(table["layers"]) != expected_sites:
        raise ValueError("calibration site count differs")
    return table


def parse_phase(experiment: dict[str, Any], model: dict[str, Any], phase: str, log: Path) -> dict[str, Any]:
    text = log.read_text(errors="strict")
    if "Traceback (most recent call last)" in text:
        raise ValueError("evaluator log contains a traceback")
    pre = json.loads(single(r"^Image preprocessing — (\{.*\})$", text, "preprocessing record"))
    if pre.get("backend") != "timm" or pre.get("config_sha256") != experiment["preprocessing_config_sha256"]:
        raise ValueError("evaluator preprocessing identity differs")
    result: dict[str, Any] = {
        "tag": TAG, "model_key": model["model_key"], "phase": phase,
        "source_commit": experiment["source_commit"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "preprocessing_config_sha256": experiment["preprocessing_config_sha256"],
        "log_file": str(log.relative_to(ROOT)), "log_sha256": sha256_file(log),
        "gpu_model": single(r"^GPU model: (.+)$", text, "GPU model"),
        "success": True,
    }
    calibration = ROOT / "calibration" / f"{model['model_key']}.json"
    if phase == "collect":
        table = validate_calibration(experiment, model, calibration)
        digest = sha256_file(calibration)
        marker = single(r"^Calibration identity — mode: collect, sha256: ([0-9a-f]{64})$", text, "calibration identity")
        if marker != digest:
            raise ValueError("saved calibration hash differs from its log")
        result.update(calibration_sha256=digest, samples=5000, sites=len(table["layers"]))
        return result

    correct = int(single(r"^Correct: (\d+)$", text, "correct count"))
    samples = int(single(r"^Evaluated samples: (\d+)$", text, "sample count"))
    digest = single(r"^Prediction SHA256: ([0-9a-f]{64})$", text, "prediction digest")
    accuracy = float(single(r"^Accuracy: ([0-9.]+)$", text, "accuracy"))
    if samples != 5000 or not 0 <= correct <= samples or abs(accuracy - correct / samples) > 5e-9:
        raise ValueError("evaluation metric is incomplete or inconsistent")
    result.update(correct=correct, samples=samples, accuracy=correct / samples, prediction_sha256=digest)
    if phase == "spiking":
        calibration_sha256 = sha256_file(calibration)
        marker = single(r"^Calibration identity — mode: validate, sha256: ([0-9a-f]{64})$", text, "calibration replay")
        if marker != calibration_sha256:
            raise ValueError("spiking evaluation used another calibration table")
        result["calibration_sha256"] = calibration_sha256
    return result


def run_phase(experiment: dict[str, Any], model: dict[str, Any], phase: str) -> dict[str, Any]:
    result_path = ROOT / "results" / f"{model['model_key']}_{phase}.json"
    if result_path.exists():
        result = read_json(result_path)
        log = ROOT / result["log_file"]
        parsed = parse_phase(experiment, model, phase, log)
        if result != parsed:
            raise ValueError("stored result differs from its immutable evidence")
        return result
    log = ROOT / "logs" / f"{model['model_key']}_{phase}.log"
    if log.exists():
        rejected = ROOT / "rejected" / f"{log.name}.{time.time_ns()}"
        shutil.move(log, rejected)
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    with log.open("w") as handle:
        process = subprocess.Popen(
            command(experiment, model, phase), cwd=SOURCE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(f"[{model['model_key']}:{phase}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{model['model_key']} {phase} failed with exit code {return_code}")
    result = parse_phase(experiment, model, phase, log)
    result["elapsed_seconds"] = time.monotonic() - started
    # Elapsed time is controller metadata and is removed for deterministic replay validation.
    write_immutable(result_path, result)
    return result


def completed_result(experiment: dict[str, Any], model: dict[str, Any], phase: str) -> dict[str, Any] | None:
    path = ROOT / "results" / f"{model['model_key']}_{phase}.json"
    if not path.exists():
        return None
    result = read_json(path)
    elapsed = result.pop("elapsed_seconds", None)
    parsed = parse_phase(experiment, model, phase, ROOT / result["log_file"])
    if result != parsed or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError("stored result differs from its immutable evidence")
    return {**result, "elapsed_seconds": elapsed}


def run_pipeline(experiment: dict[str, Any], model_key: str, physical_gpu: int) -> None:
    if physical_gpu not in (4, 5, 6, 7) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("v4 local execution requires exactly one physical GPU from 4 through 7")
    validate_experiment(experiment)
    model = model_row(experiment, model_key)
    lock_path = ROOT / "locks" / f"{model_key}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in PHASES:
            if completed_result(experiment, model, phase) is None:
                run_phase(experiment, model, phase)
            write_immutable(ROOT / "status" / f"{model_key}_{phase}.json", {
                "model_key": model_key, "phase": phase, "state": "complete",
                "result": f"results/{model_key}_{phase}.json",
            })


def summarize(experiment: dict[str, Any]) -> None:
    rows = []
    for model_key in MODEL_KEYS:
        model = model_row(experiment, model_key)
        dense = completed_result(experiment, model, "dense")
        spiking = completed_result(experiment, model, "spiking")
        collect = completed_result(experiment, model, "collect")
        if dense is None or spiking is None or collect is None:
            raise ValueError("all three model pipelines must finish before summarization")
        if spiking["calibration_sha256"] != collect["calibration_sha256"]:
            raise ValueError("summary calibration linkage differs")
        rows.append({
            "model_key": model_key,
            "evaluation_population": "fixed validation 5k",
            "ann_correct": dense["correct"], "snn_correct": spiking["correct"], "total": 5000,
            "ann_accuracy_percent": 100.0 * dense["accuracy"],
            "snn_accuracy_percent": 100.0 * spiking["accuracy"],
            "delta_percentage_points": 100.0 * (spiking["accuracy"] - dense["accuracy"]),
            "ann_prediction_sha256": dense["prediction_sha256"],
            "snn_prediction_sha256": spiking["prediction_sha256"],
            "calibration_sha256": collect["calibration_sha256"],
        })
    write_immutable(ROOT / "outputs" / "summary.json", {
        "tag": TAG, "experiment_sha256": canonical_sha256(experiment), "rows": rows,
    })
    csv = ["model_key,evaluation_population,ann_correct,snn_correct,total,ann_accuracy_percent,snn_accuracy_percent,delta_percentage_points"]
    for row in rows:
        csv.append(
            f"{row['model_key']},fixed validation 5k,{row['ann_correct']},{row['snn_correct']},5000,"
            f"{row['ann_accuracy_percent']:.2f},{row['snn_accuracy_percent']:.2f},{row['delta_percentage_points']:.2f}"
        )
    csv_path = ROOT / "outputs" / "summary.csv"
    rendered = "\n".join(csv) + "\n"
    if csv_path.exists() and csv_path.read_text() != rendered:
        raise ValueError("refusing to replace a different summary CSV")
    csv_path.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    experiment = initialize()
    if args.initialize and (args.model_key is not None or args.summarize):
        raise ValueError("--initialize must be used alone")
    if args.model_key is not None:
        if args.physical_gpu is None:
            raise ValueError("--physical-gpu is required with --model-key")
        run_pipeline(experiment, args.model_key, args.physical_gpu)
    elif args.summarize:
        summarize(experiment)
    elif not args.initialize:
        parser.error("select --initialize, --model-key, or --summarize")


if __name__ == "__main__":
    main()
