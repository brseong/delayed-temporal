#!/usr/bin/env python3
"""Dataset-free verification for the UBAI sigma/deadline-margin sweep."""

from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.analysis.summarize_sigma_margin_sweep import (
    aggregate_runs,
    aggregate_sites,
    build_frontier,
    parse_run_log,
    plot_summary,
    read_manifest,
    write_pending_manifest,
    write_provenance,
)
from scripts.experiments.ubai.build_sigma_margin_manifest import (
    PILOT_RUN_IDS,
    build_rows,
    resolve_confirmed_theta,
    serialized_tsv,
)


def common_identity() -> dict[str, str]:
    return {
        "split": "validation",
        "expected_samples": "5000",
        "dataset_path": "/data/ubai-assets/datasets/validation_50000",
        "dataset_fingerprint": "quick-fingerprint",
        "precision": "float64",
        "source_commit": "experiment-commit",
        "checkpoint_path": "/data/ubai-assets/checkpoint",
        "checkpoint_sha256": "checkpoint-sha",
        "gpu_family": "rtxa6000",
        "theta_selection_sha256": "a" * 64,
        "theta_selection_raw_sha256": "b" * 64,
        "theta_confirmation_manifest_sha256": "c" * 64,
        "gpu_selection_sha256": "d" * 64,
    }


def write_theta_evidence(root: Path) -> None:
    (root / "selection.json").write_text(
        json.dumps({
            "status": "confirmed",
            "selected_theta": 40.0,
            "evaluated_thetas": [10.0, 20.0, 40.0, 80.0],
            "validation_neighbors": [20.0, 40.0, 80.0],
        }),
        encoding="utf-8",
    )
    manifest_fields = (
        "run_id", "stage", "backend", "theta", "expected_samples",
        "checkpoint_sha256", "dataset_fingerprint", "source_commit",
    )
    confirmation_rows = [
        ("replay_theta_40", "replay", "spiking", "40", "train-fingerprint"),
        ("validation_theta_20", "validation", "spiking", "20", "quick-fingerprint"),
        ("validation_theta_40", "validation", "spiking", "40", "quick-fingerprint"),
        ("validation_theta_80", "validation", "spiking", "80", "quick-fingerprint"),
        ("validation_dense_reference", "validation", "hf", "2000", "quick-fingerprint"),
    ]
    with (root / "confirmation.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        for run_id, stage, backend, theta, fingerprint in confirmation_rows:
            writer.writerow({
                "run_id": run_id,
                "stage": stage,
                "backend": backend,
                "theta": theta,
                "expected_samples": "5000",
                "checkpoint_sha256": "checkpoint-sha",
                "dataset_fingerprint": fingerprint,
                "source_commit": "theta-commit",
            })
    raw_fields = (
        "run_id", "stage", "backend", "theta", "samples", "correct",
        "accuracy", "prediction_sha256", "checkpoint_sha256",
        "dataset_fingerprint", "source_commit",
    )
    selection_values = [
        (10, 3900, "1" * 64),
        (20, 4400, "2" * 64),
        (40, 4554, "4" * 64),
        (80, 4566, "8" * 64),
    ]
    raw_rows: list[dict[str, str]] = []
    for theta, correct, digest in selection_values:
        raw_rows.append({
            "run_id": f"selection_theta_{theta}", "stage": "selection",
            "backend": "spiking", "theta": str(theta), "samples": "5000",
            "correct": str(correct), "accuracy": str(correct / 5000),
            "prediction_sha256": digest, "checkpoint_sha256": "checkpoint-sha",
            "dataset_fingerprint": "train-fingerprint", "source_commit": "theta-commit",
        })
    confirmation_values = {
        "replay_theta_40": (40, 4554, "4" * 64, "train-fingerprint", "spiking"),
        "validation_theta_20": (20, 4067, "a" * 64, "quick-fingerprint", "spiking"),
        "validation_theta_40": (40, 4257, "b" * 64, "quick-fingerprint", "spiking"),
        "validation_theta_80": (80, 4273, "c" * 64, "quick-fingerprint", "spiking"),
        "validation_dense_reference": (2000, 4275, "d" * 64, "quick-fingerprint", "hf"),
    }
    for run_id, (theta, correct, digest, fingerprint, backend) in confirmation_values.items():
        raw_rows.append({
            "run_id": run_id,
            "stage": "replay" if run_id.startswith("replay") else "validation",
            "backend": backend, "theta": str(theta) if backend == "spiking" else "",
            "samples": "5000", "correct": str(correct),
            "accuracy": str(correct / 5000), "prediction_sha256": digest,
            "checkpoint_sha256": "checkpoint-sha", "dataset_fingerprint": fingerprint,
            "source_commit": "theta-commit",
        })
    with (root / "theta-raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)


def verify_confirmation_gate(root: Path) -> None:
    write_theta_evidence(root)
    theta, source = resolve_confirmed_theta(
        selection_json=root / "selection.json",
        theta_raw_csv=root / "theta-raw.csv",
        theta_confirmation_manifest=root / "confirmation.tsv",
        checkpoint_sha256="checkpoint-sha",
        validation_prefix_fingerprint="quick-fingerprint",
    )
    assert theta == Decimal("40") and source == "theta-commit"
    selection_path = root / "selection.json"
    original = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_path.write_text(json.dumps({**original, "status": "selected"}), encoding="utf-8")
    try:
        resolve_confirmed_theta(
            selection_json=selection_path,
            theta_raw_csv=root / "theta-raw.csv",
            theta_confirmation_manifest=root / "confirmation.tsv",
            checkpoint_sha256="checkpoint-sha",
            validation_prefix_fingerprint="quick-fingerprint",
        )
    except ValueError as error:
        assert "confirmed" in str(error)
    else:
        raise AssertionError("unconfirmed theta evidence was accepted")
    selection_path.write_text(json.dumps(original), encoding="utf-8")
    raw_path = root / "theta-raw.csv"
    original_raw = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(original_raw.replace("4" * 64, "9" * 64, 1), encoding="utf-8")
    try:
        resolve_confirmed_theta(
            selection_json=selection_path,
            theta_raw_csv=raw_path,
            theta_confirmation_manifest=root / "confirmation.tsv",
            checkpoint_sha256="checkpoint-sha",
            validation_prefix_fingerprint="quick-fingerprint",
        )
    except ValueError as error:
        assert "replay" in str(error)
    else:
        raise AssertionError("replay digest mismatch was accepted")
    raw_path.write_text(original_raw, encoding="utf-8")


def verify_builder_cli(root: Path) -> None:
    dataset_manifest = root / "dataset.json"
    dataset_manifest.write_text(json.dumps({
        "validation": {
            "quick_prefix_samples": 5000,
            "quick_prefix_fingerprint": "quick-fingerprint",
            "fingerprint": "full-fingerprint",
        }
    }), encoding="utf-8")
    gpu_selection = root / "gpu-selection.json"
    gpu_selection.write_text(json.dumps({
        "selected_family": "rtxa6000", "selected_partition": "gpu4,gpu5",
    }), encoding="utf-8")
    output = root / "canonical.tsv"
    pilot = root / "pilot.tsv"
    experiment = root / "experiment.json"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/experiments/ubai/build_sigma_margin_manifest.py"),
        "--output", str(output), "--pilot-output", str(pilot),
        "--experiment-json", str(experiment),
        "--selection-json", str(root / "selection.json"),
        "--theta-raw-csv", str(root / "theta-raw.csv"),
        "--theta-confirmation-manifest", str(root / "confirmation.tsv"),
        "--dataset-manifest", str(dataset_manifest),
        "--gpu-selection", str(gpu_selection),
        "--source-commit", "experiment-commit",
        "--checkpoint-path", "/data/ubai-assets/checkpoint",
        "--checkpoint-sha256", "checkpoint-sha",
    ], check=True, text=True, capture_output=True)
    specs = read_manifest(output)
    assert len(specs) == 470 and all(spec.theta == 40 for spec in specs)
    with pilot.open(newline="", encoding="utf-8") as handle:
        pilot_rows = list(csv.DictReader(handle, dialect="excel-tab"))
    assert {row["run_id"] for row in pilot_rows} == set(PILOT_RUN_IDS)
    contract = json.loads(experiment.read_text(encoding="utf-8"))
    assert contract["runs"] == 470 and contract["stochastic_runs"] == 468
    assert contract["selected_theta"] == 40.0


def verify_submit_dry_run(root: Path) -> None:
    assets = root / "assets"
    theta_root = root / "theta-result"
    sigma_root = root / "sigma-result"
    dataset_root = assets / "datasets/imagenet_theta_selection_v1"
    runtime = assets / "runtime"
    (theta_root / "outputs").mkdir(parents=True)
    (theta_root / "manifests").mkdir(parents=True)
    dataset_root.mkdir(parents=True)
    runtime.mkdir(parents=True)
    shutil.copy2(root / "selection.json", theta_root / "outputs/selection.json")
    shutil.copy2(root / "theta-raw.csv", theta_root / "outputs/theta-selection-raw.csv")
    shutil.copy2(root / "confirmation.tsv", theta_root / "manifests/confirmation-lower.tsv")
    shutil.copy2(root / "dataset.json", dataset_root / "manifest.json")
    shutil.copy2(root / "gpu-selection.json", theta_root / "outputs/gpu-selection.json")
    (runtime / "dt-environment.tar.zst").write_bytes(b"fixture")
    (runtime / "ubuntu-24.04.sqsh").write_bytes(b"fixture")
    environment = os.environ.copy()
    environment.update({
        "THETA_REMOTE_REPO": str(ROOT), "THETA_REMOTE_ASSETS": str(assets),
        "THETA_RESULT_ROOT": str(theta_root), "SIGMA_MARGIN_RESULT_ROOT": str(sigma_root),
        "SIGMA_MARGIN_CONTROL_PYTHON": sys.executable,
        "THETA_CHECKPOINT_SHA256": "checkpoint-sha",
    })
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh")],
        env=environment, text=True, capture_output=True, check=True,
    )
    assert "Expected runs: 470" in result.stdout
    assert "Pending runs: 470" in result.stdout
    assert "Dry preparation complete" in result.stdout
    assert len(read_manifest(sigma_root / "manifests/expected_runs.tsv")) == 470
    with (sigma_root / "manifests/pilot.tsv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle, dialect="excel-tab"))) == 6


def write_log(path: Path, spec, *, correct: int) -> None:
    enabled = spec.stage == "sigma_margin"
    ratio = spec.time_noise_std_abs / 1.0e-12 if enabled else 0.0
    site = ""
    if enabled:
        site = (
            "Gaussian[layernorm.log_positive] events=100, misses=10 (rate=0.1), "
            "deadline_events=20 (rate=0.2), deadline_ulp_min=1e-12, "
            "deadline_ulp_max=2e-12, std_to_ulp_min=10, std_to_ulp_max=20, "
            "outputs=50, underflows=1 (rate=0.02), overflows=2 (rate=0.04)\n"
        )
    seed = spec.seed if spec.seed is not None else 0
    path.write_text(
        "Slurm identity — job_id: 1, task_id: 0, node: fixture, gpu_family: rtxa6000\n"
        "Artifact identity — source_commit: experiment-commit, checkpoint_sha256: checkpoint-sha\n"
        "GPU model: NVIDIA RTX A6000\n"
        f"Gaussian time noise — enabled: {enabled}, std_frac: {spec.time_noise_std_frac}, "
        f"identity_window: {2 * spec.theta}, std_abs: {spec.time_noise_std_abs}, "
        f"mean_abs: 0.0, seed: {seed}, identity_deadline_ulp: 1e-12, "
        f"std_to_identity_ulp: {ratio}, deadline_margin_std: {spec.deadline_margin_std}, "
        f"deadline_margin_abs: {spec.deadline_margin_abs}\n"
        "Static threshold mismatch — enabled: False, theta_std: 0.0, seed: 0\n"
        "Evaluation metadata — model: checkpoint, dataset: imagenet-1k, split: validation, "
        f"samples: 5000, theta: {spec.theta}, precision: float64, source: disk:/fixture, "
        "fingerprint: quick-fingerprint\n"
        f"Correct: {correct}\nEvaluated samples: 5000\n"
        f"Prediction SHA256: {hashlib.sha256(spec.run_id.encode()).hexdigest()}\n"
        f"Accuracy: {correct / 5000:.8f}\n" + site,
        encoding="utf-8",
    )


def verify_manifest_aggregation_and_resume(root: Path) -> None:
    rows = build_rows(theta=Decimal("40"), common=common_identity(),
                      fractions=("1.000e-10",), margins=("0", "1"), seeds=(0, 1, 2))
    manifest = root / "manifest.tsv"
    manifest.write_text(serialized_tsv(rows), encoding="utf-8")
    specs = read_manifest(manifest, require_canonical=False)
    for spec in specs:
        if spec.stage == "baseline":
            correct = 4500 if spec.backend == "spiking" else 4550
        elif spec.deadline_margin_std == 0.0:
            correct = 4000 + int(spec.seed or 0)
        else:
            correct = 4475 + int(spec.seed or 0)
        write_log(root / spec.log_file, spec, correct=correct)
    runs = [parse_run_log(spec, root) for spec in specs]
    summary = aggregate_runs(runs)
    site_rows = aggregate_sites(runs)
    frontier = build_frontier(summary)
    assert len(summary) == 4 and len(site_rows) == 2
    assert frontier["frontier"][0]["minimum_recovery_margin_std"] == 1.0
    stochastic = [row for row in summary if row["stage"] == "sigma_margin"]
    assert all(row["replicas"] == 3 for row in stochastic)
    assert all(float(row["accuracy_ci95_half_width"]) >= 0.0 for row in stochastic)
    assert all(float(row["miss_rate"]) == 0.1 for row in stochastic)
    figure = root / "figure"
    plot_summary(summary, frontier, figure)
    assert figure.with_suffix(".pdf").is_file() and figure.with_suffix(".png").is_file()
    pending = root / "pending.tsv"
    assert write_pending_manifest(manifest, specs, root, pending) == 0
    (root / specs[-1].log_file).unlink()
    assert write_pending_manifest(manifest, specs, root, pending) == 1
    write_log(root / specs[-1].log_file, specs[-1], correct=4477)
    provenance = root / "provenance.json"
    write_provenance(manifest, specs, provenance)
    assert json.loads(provenance.read_text(encoding="utf-8"))["runs"] == len(specs)
    bad = replace(specs[2], gpu_family="a10")
    try:
        parse_run_log(bad, root)
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("mixed GPU-family log was accepted")


def verify_canonical_cardinality() -> None:
    rows = build_rows(theta=Decimal("40"), common=common_identity())
    assert len(rows) == 470 and len({row["run_id"] for row in rows}) == 470
    assert len([row for row in rows if row["stage"] == "sigma_margin"]) == 468
    assert all(Decimal(row["time_noise_std_abs"]) == Decimal("80") * Decimal(row["time_noise_std_frac"])
               for row in rows)
    with TemporaryDirectory() as directory:
        manifest = Path(directory) / "expected.tsv"
        manifest.write_text(serialized_tsv(rows), encoding="utf-8")
        assert len(read_manifest(manifest)) == 470


def verify_slurm_contract() -> None:
    task = (ROOT / "scripts/experiments/ubai/sigma_margin_task.sbatch").read_text()
    submit = (ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh").read_text()
    reducer = (ROOT / "scripts/experiments/ubai/sigma_margin_reduce.sbatch").read_text()
    continuation = (ROOT / "scripts/experiments/ubai/continue_sigma_margin_ubai.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in task
    assert "#SBATCH --cpus-per-task=4" in task and "#SBATCH --mem=64G" in task
    assert "DataParallel" not in task and "/usr/bin/env -u WANDB_API_KEY" in task
    assert "WANDB_MODE=disabled" in task and '--experiment_name "$run_id"' in task
    assert "WANDB_RUN_ID" not in task and "WANDB_RESUME" not in task
    assert '$HOME:$HOME' in task
    assert '--array="0-${array_end}%8"' in submit and '--array="0-5%6"' in submit
    assert 'mode="pilot"' in submit and "60000000000" in submit
    assert "/home1/sizz1997/miniconda3/bin/python" in submit
    assert "--theta-confirmation-manifest" in submit and "--wandb-dir" not in submit
    assert '--dependency="afterany:$array_job"' in submit
    assert "--wandb-run-manifest" not in reducer and "--provenance-json" in reducer
    assert 'glob("disabled-batch-*.tsv")' in continuation
    assert '--array="0-${array_end}%8"' in continuation
    assert "WANDB" not in continuation
    for path in (
        ROOT / "scripts/experiments/ubai/sigma_margin_task.sbatch",
        ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh",
        ROOT / "scripts/experiments/ubai/sigma_margin_reduce.sbatch",
        ROOT / "scripts/experiments/ubai/continue_sigma_margin_ubai.sh",
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)


def main() -> None:
    # @lat: [[lat.md/noise#Sigma and Deadline-Margin Grid]]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        verify_confirmation_gate(root)
        verify_builder_cli(root)
        verify_submit_dry_run(root)
        verify_manifest_aggregation_and_resume(root)
    verify_canonical_cardinality()
    verify_slurm_contract()
    print("sigma-margin sweep verification passed")


if __name__ == "__main__":
    main()
