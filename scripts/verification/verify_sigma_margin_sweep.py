#!/usr/bin/env python3
"""Dataset-free verification for the UBAI sigma/deadline-margin sweep."""

from __future__ import annotations

import csv
from dataclasses import replace
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
)
from scripts.experiments.ubai.build_sigma_margin_manifest import (
    build_rows,
    resolve_approved_theta,
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
        "theta_full_manifest_sha256": "c" * 64,
        "gpu_selection_sha256": "d" * 64,
    }


def verify_approval_gate(root: Path) -> None:
    selection = root / "selection.json"
    selection.write_text(
        json.dumps({"status": "approved", "selected_theta": 640.0}),
        encoding="utf-8",
    )
    full_manifest = root / "full.tsv"
    fields = (
        "run_id", "stage", "backend", "theta", "expected_samples",
        "checkpoint_sha256", "dataset_fingerprint", "source_commit",
    )
    with full_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        writer.writerows([
            {
                "run_id": "full_spiking", "stage": "full", "backend": "spiking",
                "theta": "640", "expected_samples": "50000",
                "checkpoint_sha256": "checkpoint-sha",
                "dataset_fingerprint": "full-fingerprint", "source_commit": "theta-commit",
            },
            {
                "run_id": "full_hf", "stage": "full", "backend": "hf",
                "theta": "2000", "expected_samples": "50000",
                "checkpoint_sha256": "checkpoint-sha",
                "dataset_fingerprint": "full-fingerprint", "source_commit": "theta-commit",
            },
        ])
    raw_csv = root / "theta-raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "stage", "backend", "theta", "samples", "checkpoint_sha256",
            "dataset_fingerprint", "source_commit",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {
                "stage": "full", "backend": "spiking", "theta": "640.0",
                "samples": "50000", "checkpoint_sha256": "checkpoint-sha",
                "dataset_fingerprint": "full-fingerprint", "source_commit": "theta-commit",
            },
            {
                "stage": "full", "backend": "hf", "theta": "",
                "samples": "50000", "checkpoint_sha256": "checkpoint-sha",
                "dataset_fingerprint": "full-fingerprint", "source_commit": "theta-commit",
            },
        ])
    theta, source = resolve_approved_theta(
        selection_json=selection,
        theta_raw_csv=raw_csv,
        theta_full_manifest=full_manifest,
        checkpoint_sha256="checkpoint-sha",
        validation_fingerprint="full-fingerprint",
    )
    assert float(theta) == 640.0 and source == "theta-commit"
    selection.write_text(
        json.dumps({"status": "confirmed", "selected_theta": 640.0}),
        encoding="utf-8",
    )
    try:
        resolve_approved_theta(
            selection_json=selection,
            theta_raw_csv=raw_csv,
            theta_full_manifest=full_manifest,
            checkpoint_sha256="checkpoint-sha",
            validation_fingerprint="full-fingerprint",
        )
    except ValueError as error:
        assert "approved" in str(error)
    else:
        raise AssertionError("unapproved theta selection was accepted")
    selection.write_text(
        json.dumps({"status": "approved", "selected_theta": 640.0}),
        encoding="utf-8",
    )


def verify_builder_cli(root: Path) -> None:
    dataset_manifest = root / "dataset.json"
    dataset_manifest.write_text(
        json.dumps({
            "validation": {
                "quick_prefix_samples": 5000,
                "quick_prefix_fingerprint": "quick-fingerprint",
                "fingerprint": "full-fingerprint",
            }
        }),
        encoding="utf-8",
    )
    gpu_selection = root / "gpu-selection.json"
    gpu_selection.write_text(
        json.dumps({
            "selected_family": "rtxa6000",
            "selected_partition": "gpu4,gpu5",
        }),
        encoding="utf-8",
    )
    output = root / "canonical.tsv"
    experiment = root / "experiment.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiments/ubai/build_sigma_margin_manifest.py"),
            "--output", str(output),
            "--experiment-json", str(experiment),
            "--selection-json", str(root / "selection.json"),
            "--theta-raw-csv", str(root / "theta-raw.csv"),
            "--theta-full-manifest", str(root / "full.tsv"),
            "--dataset-manifest", str(dataset_manifest),
            "--gpu-selection", str(gpu_selection),
            "--source-commit", "experiment-commit",
            "--checkpoint-path", "/data/ubai-assets/checkpoint",
            "--checkpoint-sha256", "checkpoint-sha",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert len(read_manifest(output)) == 470
    contract = json.loads(experiment.read_text(encoding="utf-8"))
    assert contract["runs"] == 470 and contract["stochastic_runs"] == 468
    assert contract["selected_theta"] == 640.0


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
    shutil.copy2(root / "full.tsv", theta_root / "manifests/full.tsv")
    shutil.copy2(root / "dataset.json", dataset_root / "manifest.json")
    (theta_root / "outputs/gpu-selection.json").write_text(
        json.dumps({
            "selected_family": "rtxa6000",
            "selected_partition": "gpu4,gpu5",
        }),
        encoding="utf-8",
    )
    (runtime / "dt-environment.tar.zst").write_bytes(b"fixture")
    (runtime / "ubuntu-24.04.sqsh").write_bytes(b"fixture")
    environment = os.environ.copy()
    environment.update({
        "THETA_REMOTE_REPO": str(ROOT),
        "THETA_REMOTE_ASSETS": str(assets),
        "THETA_RESULT_ROOT": str(theta_root),
        "SIGMA_MARGIN_RESULT_ROOT": str(sigma_root),
        "THETA_CHECKPOINT_SHA256": "checkpoint-sha",
    })
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Expected runs: 470" in result.stdout
    assert "Pending runs: 470" in result.stdout
    assert "Dry preparation complete" in result.stdout
    assert len(read_manifest(sigma_root / "manifests/expected_runs.tsv")) == 470


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
        f"Correct: {correct}\n"
        "Evaluated samples: 5000\n"
        f"Prediction SHA256: {hashlib.sha256(spec.run_id.encode()).hexdigest()}\n"
        f"Accuracy: {correct / 5000:.8f}\n"
        + site,
        encoding="utf-8",
    )


def verify_manifest_aggregation_and_resume(root: Path) -> None:
    rows = build_rows(
        theta=__import__("decimal").Decimal("640"),
        common=common_identity(),
        fractions=("1.000e-10",),
        margins=("0", "1"),
        seeds=(0, 1, 2),
    )
    manifest = root / "manifest.tsv"
    manifest.write_text(serialized_tsv(rows), encoding="utf-8")
    specs = read_manifest(manifest, require_canonical=False)
    assert len(specs) == 8
    assert all(spec.seed is None for spec in specs[:2])
    assert len({spec.run_id for spec in specs}) == 8
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
    assert figure.with_suffix(".pdf").is_file()
    assert figure.with_suffix(".png").is_file()

    pending = root / "pending.tsv"
    assert write_pending_manifest(manifest, specs, root, pending) == 0
    (root / specs[-1].log_file).unlink()
    assert write_pending_manifest(manifest, specs, root, pending) == 1
    with pending.open(newline="", encoding="utf-8") as handle:
        pending_rows = list(csv.DictReader(handle, dialect="excel-tab"))
    assert pending_rows[0]["run_id"] == specs[-1].run_id

    bad = replace(specs[2], gpu_family="a10")
    try:
        parse_run_log(bad, root)
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("mixed GPU-family log was accepted")


def verify_canonical_cardinality() -> None:
    rows = build_rows(
        theta=__import__("decimal").Decimal("640"),
        common=common_identity(),
    )
    assert len(rows) == 470
    assert len({row["run_id"] for row in rows}) == 470
    assert len([row for row in rows if row["stage"] == "sigma_margin"]) == 468
    assert all("\r" not in line for line in serialized_tsv(rows).splitlines())
    with TemporaryDirectory() as directory:
        manifest = Path(directory) / "expected.tsv"
        manifest.write_text(serialized_tsv(rows), encoding="utf-8")
        assert len(read_manifest(manifest)) == 470


def verify_slurm_contract() -> None:
    task = (ROOT / "scripts/experiments/ubai/sigma_margin_task.sbatch").read_text()
    submit = (ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh").read_text()
    reducer = (ROOT / "scripts/experiments/ubai/sigma_margin_reduce.sbatch").read_text()
    assert "#SBATCH --gres=gpu:1" in task
    assert "#SBATCH --cpus-per-task=4" in task and "#SBATCH --mem=64G" in task
    assert "DataParallel" not in task
    assert '--array="0-${array_end}%8"' in submit
    assert 'if [[ "$submit" == "0" ]]' in submit
    assert "--write-pending" in submit
    assert "--time=03:00:00" in submit
    assert '--dependency="afterany:$array_job"' in submit
    assert "jobs are already active" in submit
    assert "--frontier-json" in reducer and "--site-csv" in reducer
    assert "Reducer source commit mismatch" in reducer
    assert "--mismatch-theta-std 0" in task
    for path in (
        ROOT / "scripts/experiments/ubai/sigma_margin_task.sbatch",
        ROOT / "scripts/experiments/ubai/submit_sigma_margin_ubai.sh",
        ROOT / "scripts/experiments/ubai/sigma_margin_reduce.sbatch",
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)


def main() -> None:
    # @lat: [[lat.md/noise#Sigma and Deadline-Margin Grid]]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        verify_approval_gate(root)
        verify_builder_cli(root)
        verify_submit_dry_run(root)
        verify_manifest_aggregation_and_resume(root)
    verify_canonical_cardinality()
    verify_slurm_contract()
    print("sigma-margin sweep verification passed")


if __name__ == "__main__":
    main()
