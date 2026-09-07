#!/usr/bin/env python3
"""Build the immutable ViT-B/16 timing-sigma/deadline-margin manifest."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


TIME_NOISE_STD_FRACS = (
    "1.000e-10",
    "1.250e-10",
    "1.500e-10",
    "1.750e-10",
    "2.000e-10",
    "2.500e-10",
    "3.162e-10",
    "4.000e-10",
    "5.000e-10",
    "6.300e-10",
    "8.000e-10",
    "1.000e-09",
)
DEADLINE_MARGIN_RATIOS = (
    "0",
    "0.5",
    "1",
    "1.5",
    "2",
    "2.5",
    "3",
    "4",
    "5",
    "6",
    "8",
    "10",
    "12",
)
REPLICA_SEEDS = (0, 1, 2)
FAMILY_PARTITIONS = {
    "rtx3090": "gpu1",
    "a10": "gpu2,gpu6",
    "rtx6000ada": "gpu3",
    "rtxa6000": "gpu4,gpu5",
}
FIELDS = (
    "run_id",
    "stage",
    "backend",
    "theta",
    "time_noise_std_frac",
    "time_noise_std_abs",
    "deadline_margin_std",
    "deadline_margin_abs",
    "seed",
    "split",
    "expected_samples",
    "dataset_path",
    "dataset_fingerprint",
    "precision",
    "source_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "gpu_family",
    "theta_selection_sha256",
    "theta_selection_raw_sha256",
    "theta_confirmation_manifest_sha256",
    "gpu_selection_sha256",
    "log_file",
)

PILOT_RUN_IDS = (
    "clean_spiking_baseline",
    "dense_reference",
    "sigma_1p000em10_margin_0_seed_0",
    "sigma_3p162em10_margin_0_seed_0",
    "sigma_1p000em09_margin_0_seed_0",
    "sigma_1p000em09_margin_12_seed_0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def resolve_confirmed_theta(
    *,
    selection_json: Path,
    theta_raw_csv: Path,
    theta_confirmation_manifest: Path,
    checkpoint_sha256: str,
    validation_prefix_fingerprint: str,
) -> tuple[Decimal, str]:
    """Validate the complete 5k theta evidence and return its threshold."""

    selection = json.loads(selection_json.read_text(encoding="utf-8"))
    if selection.get("status") != "confirmed":
        raise ValueError("theta selection must have status='confirmed'")
    selected_value = selection.get("selected_theta")
    if isinstance(selected_value, bool) or not isinstance(selected_value, (int, float)):
        raise ValueError("confirmed theta selection has no numeric selected_theta")
    if not math.isfinite(float(selected_value)) or float(selected_value) <= 0.0:
        raise ValueError("selected_theta must be finite and positive")
    selected = Decimal(str(selected_value))
    if selected != Decimal("40"):
        raise ValueError(f"this campaign requires selected_theta=40, got {selected}")
    evaluated = [Decimal(str(value)) for value in selection.get("evaluated_thetas", [])]
    if not evaluated or Decimal("10") not in evaluated or Decimal("20") not in evaluated:
        raise ValueError("confirmed theta selection must include lower candidates 10 and 20")
    if min(evaluated) >= selected:
        raise ValueError("confirmed theta selection has an unresolved lower boundary")
    neighbors = [Decimal(str(value)) for value in selection.get("validation_neighbors", [])]
    if neighbors != [Decimal("20"), Decimal("40"), Decimal("80")]:
        raise ValueError("confirmed theta validation neighbors must be 20, 40, and 80")

    manifest_rows = read_tsv(theta_confirmation_manifest)
    expected_run_ids = {
        "replay_theta_40",
        "validation_theta_20",
        "validation_theta_40",
        "validation_theta_80",
        "validation_dense_reference",
    }
    if {row.get("run_id") for row in manifest_rows} != expected_run_ids:
        raise ValueError("theta confirmation manifest does not match the 5k contract")
    if any(row.get("expected_samples") != "5000" for row in manifest_rows):
        raise ValueError("theta confirmation manifest runs must cover 5,000 samples")
    if any(row.get("checkpoint_sha256") != checkpoint_sha256 for row in manifest_rows):
        raise ValueError("checkpoint differs from confirmed theta manifest")
    selection_source_commits = {row.get("source_commit") for row in manifest_rows}
    if len(selection_source_commits) != 1:
        raise ValueError("theta confirmation manifest mixes source commits")
    selection_source_commit = next(iter(selection_source_commits))
    validation_specs = [row for row in manifest_rows if row.get("stage") == "validation"]
    if any(
        row.get("dataset_fingerprint") != validation_prefix_fingerprint
        for row in validation_specs
    ):
        raise ValueError("validation prefix differs from confirmed theta manifest")

    raw_rows = read_csv(theta_raw_csv)
    by_run_id = {row.get("run_id"): row for row in raw_rows}
    if len(by_run_id) != len(raw_rows):
        raise ValueError("theta raw CSV contains duplicate run IDs")
    if not expected_run_ids.issubset(by_run_id):
        raise ValueError("theta raw CSV is missing confirmation runs")
    for spec in manifest_rows:
        run = by_run_id[spec["run_id"]]
        for field in (
            "stage", "backend", "checkpoint_sha256",
            "dataset_fingerprint", "source_commit",
        ):
            if run.get(field) != spec.get(field):
                raise ValueError(
                    f"theta raw CSV {field} mismatch for {spec['run_id']}"
                )
        if run.get("samples") != "5000":
            raise ValueError("theta raw CSV confirmation runs must contain 5,000 samples")
        if (
            spec["backend"] == "spiking"
            and Decimal(run["theta"]) != Decimal(spec["theta"])
        ):
            raise ValueError(f"theta raw CSV threshold mismatch for {spec['run_id']}")

    selection_runs = [
        row for row in raw_rows
        if row.get("stage") == "selection" and row.get("backend") == "spiking"
    ]
    selection_thetas = {Decimal(row["theta"]) for row in selection_runs}
    if selection_thetas != set(evaluated):
        raise ValueError("theta raw CSV candidates differ from selection.json")
    best_accuracy = max(Decimal(row["accuracy"]) for row in selection_runs)
    selected_by_rule = min(
        Decimal(row["theta"])
        for row in selection_runs
        if Decimal(row["accuracy"]) >= best_accuracy - Decimal("0.005")
    )
    if selected_by_rule != selected:
        raise ValueError("theta raw CSV does not reproduce the selection rule")
    selected_run = next(
        row for row in selection_runs if Decimal(row["theta"]) == selected
    )
    replay = by_run_id["replay_theta_40"]
    if (selected_run.get("correct"), selected_run.get("prediction_sha256")) != (
        replay.get("correct"), replay.get("prediction_sha256")
    ):
        raise ValueError("theta replay count or prediction digest mismatch")
    validation_runs = [
        by_run_id[f"validation_theta_{value}"] for value in (20, 40, 80)
    ]
    selected_validation = by_run_id["validation_theta_40"]
    if Decimal(selected_validation["accuracy"]) < (
        max(Decimal(row["accuracy"]) for row in validation_runs) - Decimal("0.005")
    ):
        raise ValueError("selected theta fails 5k validation stability")
    return selected, selection_source_commit


def decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "g")


def slug(value: str) -> str:
    return value.replace(".", "p").replace("+", "").replace("-", "m")


def make_row(
    *,
    run_id: str,
    stage: str,
    backend: str,
    theta: Decimal,
    time_noise_std_frac: str,
    deadline_margin_std: str,
    seed: int | None,
    common: dict[str, str],
) -> dict[str, str]:
    fraction = Decimal(time_noise_std_frac)
    margin_ratio = Decimal(deadline_margin_std)
    sigma_abs = fraction * Decimal(2) * theta
    margin_abs = margin_ratio * sigma_abs
    return {
        "run_id": run_id,
        "stage": stage,
        "backend": backend,
        "theta": decimal_text(theta),
        "time_noise_std_frac": time_noise_std_frac,
        "time_noise_std_abs": decimal_text(sigma_abs),
        "deadline_margin_std": deadline_margin_std,
        "deadline_margin_abs": decimal_text(margin_abs),
        # Use -1 instead of an empty middle TSV field because POSIX-shell IFS
        # parsing collapses adjacent tab whitespace in the Slurm task reader.
        "seed": "-1" if seed is None else str(seed),
        "log_file": f"{run_id}.log",
        **common,
    }


def build_rows(
    *,
    theta: Decimal,
    common: dict[str, str],
    fractions: Sequence[str] = TIME_NOISE_STD_FRACS,
    margins: Sequence[str] = DEADLINE_MARGIN_RATIOS,
    seeds: Sequence[int] = REPLICA_SEEDS,
) -> list[dict[str, str]]:
    rows = [
        make_row(
            run_id="clean_spiking_baseline",
            stage="baseline",
            backend="spiking",
            theta=theta,
            time_noise_std_frac="0",
            deadline_margin_std="0",
            seed=None,
            common=common,
        ),
        make_row(
            run_id="dense_reference",
            stage="baseline",
            backend="hf",
            theta=theta,
            time_noise_std_frac="0",
            deadline_margin_std="0",
            seed=None,
            common=common,
        ),
    ]
    for fraction in fractions:
        for margin in margins:
            for seed in seeds:
                run_id = (
                    f"sigma_{slug(fraction)}_margin_{slug(margin)}_seed_{seed}"
                )
                rows.append(
                    make_row(
                        run_id=run_id,
                        stage="sigma_margin",
                        backend="spiking",
                        theta=theta,
                        time_noise_std_frac=fraction,
                        deadline_margin_std=margin,
                        seed=seed,
                        common=common,
                    )
                )
    return rows


def serialized_tsv(rows: Sequence[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=FIELDS,
        dialect="excel-tab",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace different immutable artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-json", type=Path)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--theta-raw-csv", type=Path, required=True)
    parser.add_argument("--theta-confirmation-manifest", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--gpu-selection", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--dataset-root",
        default="/data/ubai-assets/datasets/imagenet_theta_selection_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    validation = dataset["validation"]
    if int(validation["quick_prefix_samples"]) != 5000:
        raise ValueError("validation artifact must define a fixed 5,000-image prefix")
    selected_theta, selection_source_commit = resolve_confirmed_theta(
        selection_json=args.selection_json,
        theta_raw_csv=args.theta_raw_csv,
        theta_confirmation_manifest=args.theta_confirmation_manifest,
        checkpoint_sha256=args.checkpoint_sha256,
        validation_prefix_fingerprint=validation["quick_prefix_fingerprint"],
    )

    gpu_selection = json.loads(args.gpu_selection.read_text(encoding="utf-8"))
    gpu_family = gpu_selection.get("selected_family")
    if gpu_family not in FAMILY_PARTITIONS:
        raise ValueError("GPU selection has no supported selected_family")
    if gpu_selection.get("selected_partition") != FAMILY_PARTITIONS[gpu_family]:
        raise ValueError("GPU selection partition does not match its family")

    evidence_hashes = {
        "theta_selection_sha256": sha256_file(args.selection_json),
        "theta_selection_raw_sha256": sha256_file(args.theta_raw_csv),
        "theta_confirmation_manifest_sha256": sha256_file(args.theta_confirmation_manifest),
        "gpu_selection_sha256": sha256_file(args.gpu_selection),
    }
    common = {
        "split": "validation",
        "expected_samples": "5000",
        "dataset_path": str(Path(args.dataset_root) / "validation_50000"),
        "dataset_fingerprint": validation["quick_prefix_fingerprint"],
        "precision": "float64",
        "source_commit": args.source_commit,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": args.checkpoint_sha256,
        "gpu_family": gpu_family,
        **evidence_hashes,
    }
    rows = build_rows(theta=selected_theta, common=common)
    if len(rows) != 470:
        raise AssertionError(f"canonical manifest must contain 470 runs, got {len(rows)}")
    write_immutable(args.output, serialized_tsv(rows))
    if args.pilot_output is not None:
        pilot_rows = [row for row in rows if row["run_id"] in PILOT_RUN_IDS]
        if {row["run_id"] for row in pilot_rows} != set(PILOT_RUN_IDS):
            raise AssertionError("canonical manifest is missing pilot conditions")
        write_immutable(args.pilot_output, serialized_tsv(pilot_rows))

    experiment_path = args.experiment_json or args.output.with_name("experiment.json")
    experiment: dict[str, Any] = {
        "format_version": 1,
        "tag": "vit_base_sigma_margin_5k_float64_v1",
        "status": "planned",
        "validation_scope": "fixed-prefix-5000",
        "wandb_mode": "offline",
        "selected_theta": float(selected_theta),
        "theta_selection_source_commit": selection_source_commit,
        "time_noise_std_fracs": list(TIME_NOISE_STD_FRACS),
        "deadline_margin_ratios": list(DEADLINE_MARGIN_RATIOS),
        "replica_seeds": list(REPLICA_SEEDS),
        "recovery_tolerance": 0.01,
        "runs": len(rows),
        "stochastic_runs": len(rows) - 2,
        "source_commit": args.source_commit,
        "checkpoint_sha256": args.checkpoint_sha256,
        "dataset_fingerprint": validation["quick_prefix_fingerprint"],
        "gpu_family": gpu_family,
        "gpu_partition": FAMILY_PARTITIONS[gpu_family],
        **evidence_hashes,
    }
    write_immutable(
        experiment_path,
        json.dumps(experiment, indent=2, sort_keys=True) + "\n",
    )
    print(f"{len(rows)}\t{gpu_family}\t{FAMILY_PARTITIONS[gpu_family]}\t{decimal_text(selected_theta)}")


if __name__ == "__main__":
    main()
