#!/usr/bin/env python3
"""Validate and summarize the three calibrated text-model comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts.experiments.run_full_calibrated_text_comparison import (
    FAMILY_CONFIG, MODEL_CONFIG, TAG, canonical, parse_evaluation, parse_sites, sha256_file,
)


def csv_text(rows: list[dict[str, Any]], empty: tuple[str, ...]) -> str:
    fields = sorted({key for row in rows for key in row}) or list(empty)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".text-summary-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_family(root: Path, family: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = MODEL_CONFIG[family]
    output = root / family
    manifest = json.loads((output / "manifest.json").read_text())
    result = json.loads((output / "result.json").read_text())
    if (manifest.get("tag") != config["tag"] or manifest.get("family") != family
            or result.get("state") != "complete"):
        raise ValueError(f"{family} is not a completed member of {config['tag']}")
    if manifest.get("evaluation_samples") != config["evaluation_samples"]:
        raise ValueError(f"{family} evaluation population differs")
    if manifest.get("theta") != 40.0 or manifest.get("dtype") != "float64":
        raise ValueError(f"{family} numerical contract differs")
    phases = result.get("phases", {})
    if set(phases) != {"collect", "ann", "snn"}:
        raise ValueError(f"{family} phase population differs")
    calibration_path = output / "calibration.json"
    metadata, sites = parse_sites(calibration_path, config["sites"])
    calibration_sha = sha256_file(calibration_path)
    rows: list[dict[str, Any]] = []
    for phase in ("collect", "ann", "snn"):
        evidence = phases[phase]
        log_path = output / evidence["log_file"]
        if sha256_file(log_path) != evidence["log_sha256"]:
            raise ValueError(f"{family} {phase} log hash differs")
        row: dict[str, Any] = {
            "family": family, "phase": phase, "source_commit": manifest["source_commit"],
            "checkpoint_identity_sha256": hashlib.sha256(
                canonical(manifest["checkpoint_files_sha256"]).encode()
            ).hexdigest(),
            "dataset_fingerprint": manifest[
                "calibration_dataset" if phase == "collect" else "evaluation_dataset"
            ]["fingerprint"],
            "calibration_sha256": calibration_sha, "log_sha256": evidence["log_sha256"],
            "elapsed_seconds": evidence["elapsed_seconds"], "theta": 40.0,
            "dtype": "float64", "batch_size": manifest["batch_size"],
        }
        if phase == "collect":
            row.update(samples=manifest["calibration_samples"], calibration_site_count=len(sites))
        else:
            metrics = parse_evaluation(log_path.read_text(), family, sites if phase == "snn" else None)
            row.update(metrics)
        rows.append(row)
    if result.get("calibration_sha256") != calibration_sha:
        raise ValueError(f"{family} final calibration identity differs")
    return rows, {"family": family, "site_count": len(sites),
                  "calibration_sha256": calibration_sha,
                  "metadata_sha256": hashlib.sha256(canonical(metadata).encode()).hexdigest()}


def build(
    root: Path, output: Path, *, require_complete: bool,
    requested_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected = tuple(FAMILY_CONFIG) if requested_models is None else requested_models
    if not expected or len(set(expected)) != len(expected) or any(name not in MODEL_CONFIG for name in expected):
        raise ValueError("requested text models must be unique supported model keys")
    families = [family for family in expected if (root / family / "result.json").exists()]
    if require_complete and set(families) != set(expected):
        raise ValueError("every requested text model must be complete")
    raw, sites = [], []
    for family in expected:
        if family in families:
            family_rows, site = validate_family(root, family)
            raw.extend(family_rows)
            sites.append(site)
    commits = {row["source_commit"] for row in raw}
    if len(commits) > 1:
        raise ValueError("text results mix source commits")
    summary = []
    for family in expected:
        rows = {row["phase"]: row for row in raw if row["family"] == family}
        if not rows:
            continue
        ann, snn = rows["ann"], rows["snn"]
        if family == "gpt2":
            summary.append({
                "family": family, "samples": snn["total"],
                "ann_token_weighted_perplexity": ann["token_weighted_perplexity"],
                "snn_token_weighted_perplexity": snn["token_weighted_perplexity"],
                "ann_compatibility_perplexity": ann["perplexity"],
                "snn_compatibility_perplexity": snn["perplexity"],
                "valid_token_count": snn["valid_token_count"],
            })
        else:
            summary.append({
                "family": family, "samples": snn["total"],
                "ann_correct": ann["correct"], "snn_correct": snn["correct"],
                "ann_accuracy_percent": 100 * ann["accuracy"],
                "snn_accuracy_percent": 100 * snn["accuracy"],
                "accuracy_difference_pp": 100 * (snn["accuracy"] - ann["accuracy"]),
            })
    files = {
        "raw_runs.csv": csv_text(raw, ("family", "phase")),
        "summary.csv": csv_text(summary, ("family",)),
        "calibration_sites.csv": csv_text(sites, ("family", "site_count")),
    }
    provenance = {
        "tag": (MODEL_CONFIG[expected[0]]["tag"]
                if len({MODEL_CONFIG[name]["tag"] for name in expected}) == 1 else "mixed"),
        "complete": set(families) == set(expected),
        "source_commit": next(iter(commits)) if commits else None,
        "families": families,
        "files": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
    }
    for name, value in files.items():
        atomic_text(output / name, value)
    atomic_text(output / "provenance.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_CONFIG))
    args = parser.parse_args()
    requested = None if args.models is None else tuple(args.models)
    print(canonical(build(
        args.root, args.output, require_complete=args.require_complete,
        requested_models=requested,
    )))


if __name__ == "__main__":
    main()
