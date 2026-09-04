#!/usr/bin/env python3
"""Focused offline checks for the deterministic theta-selection workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from datasets import Dataset

_ROOT = Path(__file__).resolve().parents[2]
for path in (_ROOT, _ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.analysis.select_ubai_gpu_family import (
    choose_family,
    failed_run_reason,
    gpu_model_matches_family,
)
from scripts.analysis.summarize_theta_selection import (
    BASE_THETAS,
    Run,
    choose_theta,
    parse_log,
    validate_replay_and_validation,
    validation_neighbors,
)
from scripts.evaluation.error_analysis_vit import load_evaluation_dataset
from scripts.experiments.ubai.build_theta_selection_manifest import BASE_THETAS as MANIFEST_THETAS
from scripts.experiments.ubai.build_theta_selection_manifest import row as manifest_row
from scripts.setup.prepare_imagenet_theta_selection import label_sha256
from utils.transformers.optional_tensorboard import create_summary_writer


def make_run(
    theta: float | None,
    accuracy: float,
    *,
    stage: str = "selection",
    backend: str = "spiking",
    digest: str = "a" * 64,
) -> Run:
    samples = 5000
    correct = round(accuracy * samples)
    return Run(
        run_id=f"{stage}_{backend}_{theta}",
        stage=stage,
        backend=backend,
        theta=theta,
        split="train-selection-seed0-5000",
        samples=samples,
        correct=correct,
        accuracy=correct / samples,
        prediction_sha256=digest,
        dataset_fingerprint="dataset-fingerprint",
        source_commit="commit",
        checkpoint_sha256="checkpoint",
        gpu_model="NVIDIA RTX A6000",
        semantic_max_rate=0.0,
        semantic_max_site="",
        structural_events=0,
        benchmark_seconds_per_image=0.1,
        benchmark_peak_memory_bytes=1024,
        log_file="fixture.log",
    )


def verify_selection_rule() -> None:
    runs = [make_run(theta, 0.80) for theta in BASE_THETAS]
    runs[BASE_THETAS.index(40.0)] = make_run(40.0, 0.7948)
    runs[BASE_THETAS.index(80.0)] = make_run(80.0, 0.7950)
    selected, status, best = choose_theta(runs)
    assert status == "selected" and selected == 80.0 and best == 0.80
    assert validation_neighbors(80.0, BASE_THETAS) == [40.0, 80.0, 160.0]

    rising = [make_run(theta, 0.80) for theta in BASE_THETAS]
    rising[-2] = make_run(2800.0, 0.80)
    rising[-1] = make_run(4000.0, 0.802)
    selected, status, _ = choose_theta(rising)
    assert selected is None and status == "needs_extension"
    rising.extend((make_run(5600.0, 0.803), make_run(8000.0, 0.805)))
    selected, status, _ = choose_theta(rising)
    assert selected is None and status == "range_insufficient"


def verify_confirmation_rule() -> None:
    selection = make_run(640.0, 0.80, digest="b" * 64)
    replay = make_run(640.0, 0.80, stage="replay", digest="b" * 64)
    validation = [
        make_run(320.0, 0.795, stage="validation"),
        make_run(640.0, 0.800, stage="validation"),
        make_run(1000.0, 0.802, stage="validation"),
    ]
    validate_replay_and_validation([selection, replay, *validation], 640.0)
    failed = replace(validation[1], accuracy=0.79, correct=3950)
    try:
        validate_replay_and_validation([selection, replay, validation[0], failed, validation[2]], 640.0)
    except ValueError as error:
        assert "stability" in str(error)
    else:
        raise AssertionError("unstable validation selection was accepted")


def write_fixture_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "Artifact identity — source_commit: commit, checkpoint_sha256: checkpoint",
                "GPU model: NVIDIA RTX A6000",
                "Evaluation metadata — model: checkpoint, dataset: imagenet-1k, split: validation, samples: 5000, theta: 640.0, precision: float64, source: disk:/dataset, fingerprint: dataset-fingerprint",
                "Correct: 4000",
                "Evaluated samples: 5000",
                f"Prediction SHA256: {'c' * 64}",
                "Accuracy: 0.80000000",
                "Benchmark — warmup_batches: 5, measure_batches: 20, images: 640, seconds: 64, seconds_per_image: 0.1, peak_memory_bytes: 1024",
                "Clamp[layer/x_err_pos] values=100, underflows=50 (rate=0.5), overflows=0 (rate=0)",
                "Clamp[layer/attn_score] values=100, underflows=0 (rate=0), overflows=1 (rate=0.01)",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def verify_log_parser() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_fixture_log(root / "run.log")
        spec = {
            "run_id": "run",
            "stage": "validation",
            "backend": "spiking",
            "theta": "640",
            "split": "validation",
            "expected_samples": "5000",
            "dataset_fingerprint": "dataset-fingerprint",
            "precision": "float64",
            "source_commit": "commit",
            "checkpoint_sha256": "checkpoint",
            "gpu_family": "rtxa6000",
            "log_file": "run.log",
        }
        run = parse_log(spec, root)
        assert run.correct == 4000 and run.semantic_max_site == "layer/attn_score"
        assert run.structural_events == 50


def verify_offline_dataset_and_tensorboard() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_path = root / "dataset"
        source = Dataset.from_dict({"image": [0, 1], "label": [3, 4]})
        source.save_to_disk(str(dataset_path))
        args = SimpleNamespace(
            dataset_id="fixture",
            evaluation_dataset_path=str(dataset_path),
            evaluation_split="train-selection",
        )
        loaded, split, origin = load_evaluation_dataset(
            args,
            configured_split="validation",
        )
        assert loaded["label"] == [3, 4]
        assert loaded._fingerprint == Dataset.load_from_disk(str(dataset_path))._fingerprint
        assert label_sha256(loaded) == label_sha256(source)
        assert split == "train-selection" and origin.startswith("disk:")

        log_dir = root / "runs"
        writer = create_summary_writer(log_dir=str(log_dir), enabled=False)
        writer.add_scalar("x", 1, 0)
        writer.close()
        assert not log_dir.exists()


def verify_manifest_contract() -> None:
    rows = [
        manifest_row(
            run_id=f"selection_theta_{theta}",
            stage="selection",
            backend="spiking",
            theta=theta,
            split="train-selection-seed0-5000",
            expected_samples=5000,
            dataset_path="/data/dataset",
            dataset_fingerprint="dataset",
            source_commit="commit",
            checkpoint_path="/data/checkpoint",
            checkpoint_sha256="checkpoint",
            gpu_family="a10",
        )
        for theta in MANIFEST_THETAS
    ]
    assert len(rows) == 10
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert [int(row["theta"]) for row in rows] == list(MANIFEST_THETAS)
    task_script = (
        _ROOT / "scripts" / "experiments" / "ubai" / "theta_selection_task.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in task_script
    assert "DataParallel" not in task_script

    controller_script = (
        _ROOT
        / "scripts"
        / "experiments"
        / "ubai"
        / "continue_theta_selection_workflow.sh"
    ).read_text(encoding="utf-8")
    assert "start-selection|post-selection|post-extension|post-confirmation|finalize" in controller_script
    assert "selection-expanded.tsv" in controller_script
    assert "03:00:00" in controller_script
    assert "08:00:00" in controller_script
    assert "--dependency=\"afterok:${dependency}\"" in controller_script

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.tsv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            import csv

            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(rows[0]),
                dialect="excel-tab",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        assert b"\r" not in manifest_path.read_bytes()


def verify_gpu_selection() -> None:
    reference = [
        replace(
            make_run(2000.0, 0.8, stage="benchmark"),
            samples=640,
            correct=512,
            benchmark_seconds_per_image=value,
        )
        for value in (0.10, 0.11)
    ]
    ada = [
        replace(
            make_run(2000.0, 0.8, stage="benchmark"),
            samples=640,
            correct=512,
            benchmark_seconds_per_image=value,
        )
        for value in (0.102, 0.103)
    ]
    selected, payload = choose_family(
        {"rtxa6000": reference, "rtx6000ada": ada},
        availability={"rtxa6000": 20, "rtx6000ada": 4},
    )
    assert selected == "rtxa6000"
    assert payload["selected_partition"] == "gpu4,gpu5"
    assert gpu_model_matches_family("NVIDIA RTX 6000 Ada Generation", "rtx6000ada")
    assert not gpu_model_matches_family("NVIDIA RTX A6000", "rtx6000ada")

    wrong_model = [replace(run, gpu_model="NVIDIA A10") for run in ada]
    selected, payload = choose_family(
        {"rtxa6000": reference, "rtx6000ada": wrong_model},
        availability={},
    )
    assert selected == "rtxa6000"
    assert "GPU model" in payload["rejected"]["rtx6000ada"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = {
            "log_file": "a10.log",
            "source_commit": "commit",
            "gpu_family": "a10",
        }
        (root / "a10.log.partial.1").write_text(
            "gpu_family: a10\nsource commit\ntorch.OutOfMemoryError: CUDA out of memory\n",
            encoding="utf-8",
        )
        assert failed_run_reason(spec, root) == "OOM at batch size 32"


def main() -> None:
    # @lat: [[lat.md/evaluation#ViT-B/16 Global Theta Selection]]
    verify_selection_rule()
    verify_confirmation_rule()
    verify_log_parser()
    verify_offline_dataset_and_tensorboard()
    verify_manifest_contract()
    verify_gpu_selection()
    print("theta selection verification passed")


if __name__ == "__main__":
    main()
