"""Prepare UBAI evaluation files without submitting GPU evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
TAG = "vit_base_noise_calibrated_theta40_float64_v1"
CANONICAL_REPO = Path("/data/delayed-temporal")
CANONICAL_SOURCE = Path("/data/delayed-temporal-worktrees/gelu-timeconstant-noise")
ASSET_RELATIVE = Path("artifacts/assets/theta-selection-v1")
SOURCE_COMMIT = "648af9bbbea796bcbc6589449b1f9714e156ebc8"
CONTROL_COMMIT = "8dfe57c253d91d8c1f81bdb9458c786cdf493b15"
ARCHIVE_SHA256 = "3ac55cc182fa8f0671110c45c29cca9bb7e04e7d05c2bf6239b17be75ebc6762"
VALIDATION_SHA256 = "1bd5444893dce45589a315943bde9e3f356a1d1ab9a7c0e5a02b3123786bd42b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace a different preparation file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return fields, rows


def serialize(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def validate_calibration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = json.loads((root / "experiment.json").read_text())
    evidence = json.loads((root / "calibration-evidence.json").read_text())
    table = json.loads((root / "calibration.json").read_text())
    if experiment["source_commit"] != SOURCE_COMMIT:
        raise ValueError("The numerical source does not match this deployment")
    if evidence["experiment_sha256"] != sha256(root / "experiment.json"):
        raise ValueError("Calibration evidence has a different experiment")
    if evidence["calibration_sha256"] != sha256(root / "calibration.json"):
        raise ValueError("Calibration table checksum mismatch")
    if evidence["collection_log_sha256"] != sha256(root / "logs/calibration_collect.log"):
        raise ValueError("Calibration collection log checksum mismatch")
    if len(table["layers"]) != 48 or evidence["sites"] != 48:
        raise ValueError("Expected 48 calibrated layer ranges")
    options = dict(table["metadata"]["model_options"])
    for key in ("source_commit", "checkpoint_sha256", "gelu_cubic_implementation", "gelu_cubic_floor", "calibration_dataset_fingerprint"):
        if options[key] != experiment[key]:
            raise ValueError(f"Calibration metadata mismatch: {key}")
    if table["metadata"]["model_id"] != experiment["checkpoint_path"]:
        raise ValueError("Calibration checkpoint path mismatch")
    expected = {
        "precision": "float64", "batch_size": 32, "theta": 40,
        "calibration_samples": 5000, "calibration_seed": 0, "calibration_bins": 2048,
        "calibration_lower_quantile": 0.0, "calibration_upper_quantile": 1.0,
        "calibration_margin_fraction": 0.05,
        "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
    }
    if any(experiment.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected evaluation settings")
    wrapper = REPO / "scripts/analysis/evaluate_calibrated_vit.py"
    if sha256(wrapper) != experiment["calibration_evaluator_sha256"]:
        raise ValueError("Calibrated evaluator changed")
    return experiment, evidence


# @lat: [[evaluation#Evaluation and Verification#Calibrated ViT UBAI Preparation]]
def prepare(root: Path, output: Path, *, image: Path, host_base: Path) -> dict[str, Any]:
    experiment, evidence = validate_calibration(root)
    fields, rows = read_rows(root / "manifests/grid.tsv")
    if len(rows) != 65 or len({row["run_id"] for row in rows}) != 65:
        raise ValueError("Expected 65 unique conditions")
    for row in rows:
        if row["source_commit"] != SOURCE_COMMIT or row["calibration_sha256"] != evidence["calibration_sha256"]:
            raise ValueError("Manifest identity mismatch")
        if row["gpu_family"] != "rtxa6000":
            raise ValueError("Only the validated RTX A6000 family is supported")
    # This is intentionally not an allocation: the live local runner remains in charge.
    candidates = [row for row in rows if row["stage"] == "sigma_margin"]
    host_assets = host_base / "delayed-temporal-assets/theta-selection-v1"
    host_experiment = host_assets / "results" / TAG
    host_deployment = host_experiment / "ubai" / output.name
    scripts = ("calibrated_noise_task.sbatch", "calibrated_noise_prep.sbatch", "run_calibrated_noise_task.py")
    for name in scripts:
        immutable(output / "code" / name, (REPO / "scripts/experiments/ubai" / name).read_bytes())
    runtime_tools = []
    for original, relative in (
        (REPO / "scripts/experiments/ubai/calibrated_git.sh", "tools/git"),
        (Path("/usr/bin/git"), "tools/git.bin"),
        (Path("/lib/x86_64-linux-gnu/libpcre2-8.so.0"), "tools/lib/libpcre2-8.so.0"),
        (Path("/lib/x86_64-linux-gnu/libz.so.1"), "tools/lib/libz.so.1"),
    ):
        immutable(output / relative, original.read_bytes())
        if relative in {"tools/git", "tools/git.bin"}:
            (output / relative).chmod(0o755)
        runtime_tools.append({"path": "/calibrated-deployment/" + relative,
                              "sha256": sha256(output / relative)})
    immutable(output / "manifests/pending.tsv", serialize(fields, candidates))
    # Preserve the scientific result contract independently of deployment metadata.
    for relative in ("experiment.json", "calibration.json", "calibration-evidence.json", "manifests/grid.tsv", "logs/calibration_collect.log", "logs/dense_reference.log"):
        immutable(output / "experiment" / relative, (root / relative).read_bytes())
    clean = root / "logs/clean_spiking_baseline.log"
    clean_hash = None
    if clean.exists():
        immutable(output / "experiment/logs/clean_spiking_baseline.log", clean.read_bytes())
        clean_hash = sha256(clean)
    canonical_root = CANONICAL_REPO / "artifacts/logs/noise_scan" / TAG
    canonical_assets = CANONICAL_REPO / ASSET_RELATIVE
    deployment = {
        "format_version": 1, "state": "prepared", "tag": TAG,
        "source_root": str(CANONICAL_SOURCE), "source_commit": SOURCE_COMMIT,
        "control_commit": CONTROL_COMMIT,
        "wrapper_path": str(CANONICAL_REPO / "scripts/analysis/evaluate_calibrated_vit.py"),
        "experiment_root": str(canonical_root),
        "experiment_sha256": sha256(root / "experiment.json"),
        "calibration_sha256": evidence["calibration_sha256"],
        "calibration_evidence_sha256": sha256(root / "calibration-evidence.json"),
        "grid_sha256": sha256(root / "manifests/grid.tsv"),
        "assigned_manifest_path": "/calibrated-deployment/manifests/pending.tsv",
        "assigned_manifest_sha256": sha256(output / "manifests/pending.tsv"),
        "task_script_sha256": sha256(output / "code/calibrated_noise_task.sbatch"),
        "prep_script_sha256": sha256(output / "code/calibrated_noise_prep.sbatch"),
        "worker_sha256": sha256(output / "code/run_calibrated_noise_task.py"),
        "reference_clean_log_sha256": clean_hash,
        "assignment_required": True, "paper_promotion_allowed": False,
        "runtime_tools": runtime_tools,
        "limits": {"max_running_jobs": 10, "max_submitted_jobs": 20, "max_gpus": 12,
                   "gpu_per_job": 1, "cpus_per_job": 4, "memory_gib_per_job": 64,
                   "partitions": ["gpu4", "gpu5"]},
        "assets": [
            {"path": experiment["checkpoint_path"], "aggregate_sha256": experiment["checkpoint_sha256"]},
            {"path": experiment["calibration_dataset_path"], "aggregate_sha256": experiment["calibration_dataset_sha256"]},
            {"path": str(canonical_assets / "datasets/imagenet_theta_selection_v1/validation_50000"), "aggregate_sha256": VALIDATION_SHA256},
        ],
        "runtime": {
            "host_source_root": str(host_base / "delayed-temporal-gelu-648af9b"),
            "host_repository_root": str(host_base / "delayed-temporal-calibrated-noise"),
            "host_assets_root": str(host_assets), "host_experiment_root": str(host_experiment),
            "host_deployment_root": str(host_deployment),
            "env_archive": str(host_assets / "runtime/dt-environment.tar.zst"),
            "env_archive_sha256": ARCHIVE_SHA256,
            "env_unpacked_bytes": 96 * 1024 ** 3,
            "minimum_scratch_bytes": 8 * 1024 ** 3,
            "expected_python": "3.12.13",
            "container_image": str(host_assets / "runtime/ubuntu-24.04.sqsh"),
            "container_image_sha256": sha256(image),
        },
    }
    immutable(output / "deployment.json", (json.dumps(deployment, indent=2, sort_keys=True) + "\n").encode())
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=REPO / "artifacts/logs/noise_scan" / TAG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--host-base", type=Path, default=Path("/home1/sizz1997/myubai"))
    args = parser.parse_args()
    deployment = prepare(args.experiment_root, args.output, image=args.container_image, host_base=args.host_base)
    print(json.dumps({"state": deployment["state"], "deployment": str(args.output / "deployment.json"), "gpu_evaluations_submitted": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
