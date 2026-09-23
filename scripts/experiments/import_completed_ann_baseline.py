#!/usr/bin/env python3
"""Authenticate a completed ANN phase and materialize it into another pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_full_calibrated_text_comparison as text_runner
from scripts.experiments import run_full_calibrated_vit_comparison as vit_runner
from scripts.runtime import ann_baseline
from scripts.runtime import files as runtime_files
from scripts.runtime import identity


def text_identity(manifest: dict) -> dict:
    family = manifest["family"]
    args = argparse.Namespace(family=family)
    return text_runner.dense_baseline_identity(
        args,
        checkpoint_files_sha256=manifest["checkpoint_files_sha256"],
        evaluation_dataset=manifest["evaluation_dataset"],
    )


def vit_identity(manifest: dict) -> dict:
    model_key = manifest["model_key"]
    config = vit_runner.MODEL_CONFIG[model_key]
    command = manifest["commands"]["ann"]
    preprocessing_sha256 = None
    if "--image-preprocessing-config" in command:
        index = command.index("--image-preprocessing-config")
        preprocessing_sha256 = identity.sha256_file(Path(command[index + 1]))
    return ann_baseline.build_identity(
        model_key=model_key,
        model_family="vit",
        checkpoint_identity=manifest["checkpoint_sha256"],
        evaluation_dataset=manifest["evaluation_dataset"],
        evaluation_settings={
            "backend": "hf",
            "metric": "top1_accuracy",
            "split": config["split"],
            "evaluation_samples": config["samples"],
            "batch_size": manifest["batch_size"],
            "precision": manifest["dtype"],
            "image_preprocessing_sha256": preprocessing_sha256,
            "quick_test": config["dataset_id"] == "imagenet-1k",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--kind", choices=("vit", "text"), required=True)
    parser.add_argument("--source-pipeline", type=Path, required=True)
    parser.add_argument("--target-pipeline", type=Path)
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("/data/delayed-temporal/artifacts/logs/ann_baselines/v1"),
    )
    args = parser.parse_args()
    source = args.source_pipeline.resolve(strict=True)
    source_manifest = json.loads((source / "manifest.json").read_text())
    build_identity = text_identity if args.kind == "text" else vit_identity
    source_identity = build_identity(source_manifest)
    target = args.target_pipeline.resolve(strict=True) if args.target_pipeline else None
    if target is not None:
        if source == target:
            raise ValueError("ANN baseline source and target must differ")
        target_manifest = json.loads((target / "manifest.json").read_text())
        target_identity = build_identity(target_manifest)
        if source_identity != target_identity:
            raise ValueError("source and target dense evaluation identities differ")
    else:
        target_manifest = source_manifest
        target_identity = source_identity

    source_result = json.loads((source / "result.json").read_text())
    phase_path = source / "phases/ann.json"
    if not phase_path.is_file():
        raise ValueError("source pipeline has no completed ANN phase")
    source_phase = json.loads(phase_path.read_text())
    if source_result.get("phases", {}).get("ann") not in (None, source_phase):
        raise ValueError("source ANN result and phase evidence differ")
    source_log = source / source_phase["log_file"]
    if identity.sha256_file(source_log) != source_phase.get("log_sha256"):
        raise ValueError("source ANN log hash differs")
    if args.kind == "text":
        family = target_manifest["family"]
        metrics = text_runner.parse_evaluation(source_log.read_text(), family)
    else:
        expected = vit_runner.MODEL_CONFIG[target_manifest["model_key"]]["samples"]
        metrics = vit_runner.parse_metric(source_log.read_text(), expected)
    if metrics != source_phase.get("metrics"):
        raise ValueError("source ANN metrics differ from its log")

    record = ann_baseline.publish(
        cache_root=args.cache_root,
        baseline_identity=target_identity,
        source_log=source_log,
        metrics=metrics,
        elapsed_seconds=source_phase["elapsed_seconds"],
    )
    cached = ann_baseline.load(
        cache_root=args.cache_root,
        baseline_identity=target_identity,
        parse_metrics=(
            (lambda text: text_runner.parse_evaluation(text, target_manifest["family"]))
            if args.kind == "text" else
            (lambda text: vit_runner.parse_metric(
                text, vit_runner.MODEL_CONFIG[target_manifest["model_key"]]["samples"],
            ))
        ),
    )
    if cached is None:
        raise RuntimeError("published ANN baseline is unavailable")
    if target is not None:
        phase = ann_baseline.materialize_phase(record=cached[0], source_log=cached[1], output=target)
        phase_path = target / "phases/ann.json"
        if phase_path.exists():
            if json.loads(phase_path.read_text()) != phase:
                raise ValueError("target ANN phase already differs")
        else:
            runtime_files.new_json(phase_path, phase)
    print(json.dumps({
        "state": "complete",
        "source_pipeline": str(source),
        "target_pipeline": str(target) if target is not None else None,
        "ann_baseline_identity_sha256": record["identity_sha256"],
        "metrics": metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
