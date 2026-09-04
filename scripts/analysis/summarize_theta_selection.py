#!/usr/bin/env python3
"""Validate, select, and plot the deterministic ViT theta sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


BASE_THETAS = (40.0, 80.0, 160.0, 320.0, 640.0, 1000.0, 1400.0, 2000.0, 2800.0, 4000.0)
EXTENSION_THETAS = (5600.0, 8000.0)
ACCURACY_TOLERANCE = 0.005
PLATEAU_TOLERANCE = 0.001
STRUCTURAL_CLAMP_NAMES = frozenset({"x_err_neg", "x_err_pos", "multiplication_result"})


@dataclass(frozen=True)
class Run:
    """One complete evaluator result with immutable experiment identity."""

    run_id: str
    stage: str
    backend: str
    theta: float | None
    split: str
    samples: int
    correct: int
    accuracy: float
    prediction_sha256: str
    dataset_fingerprint: str
    source_commit: str
    checkpoint_sha256: str
    gpu_model: str
    semantic_max_rate: float
    semantic_max_site: str
    structural_events: int
    benchmark_seconds_per_image: float | None
    benchmark_peak_memory_bytes: int | None
    log_file: str


_ARTIFACT_RE = re.compile(
    r"^Artifact identity — source_commit: (?P<commit>[^,]+), "
    r"checkpoint_sha256: (?P<checkpoint>\S+)$",
    re.MULTILINE,
)
_METADATA_RE = re.compile(
    r"^Evaluation metadata — model: (?P<model>.*?), dataset: (?P<dataset>.*?), "
    r"split: (?P<split>.*?), samples: (?P<samples>\d+), theta: (?P<theta>[^,]+), "
    r"precision: (?P<precision>[^,]+), source: (?P<source>.*?), "
    r"fingerprint: (?P<fingerprint>\S+)$",
    re.MULTILINE,
)
_GPU_RE = re.compile(r"^GPU model: (?P<gpu>.+)$", re.MULTILINE)
_CORRECT_RE = re.compile(r"^Correct: (?P<value>\d+)$", re.MULTILINE)
_EVALUATED_RE = re.compile(r"^Evaluated samples: (?P<value>\d+)$", re.MULTILINE)
_DIGEST_RE = re.compile(r"^Prediction SHA256: (?P<value>[0-9a-f]{64})$", re.MULTILINE)
_ACCURACY_RE = re.compile(r"^Accuracy: (?P<value>[0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)
_CLAMP_RE = re.compile(
    r"^Clamp\[(?P<site>.+)] values=(?P<values>\d+), "
    r"underflows=(?P<underflows>\d+) \(rate=[^)]+\), "
    r"overflows=(?P<overflows>\d+) \(rate=[^)]+\)$",
    re.MULTILINE,
)
_BENCHMARK_RE = re.compile(
    r"^Benchmark — warmup_batches: (?P<warmup>\d+), "
    r"measure_batches: (?P<measure>\d+), images: (?P<images>\d+), "
    r"seconds: (?P<seconds>[0-9.eE+-]+), "
    r"seconds_per_image: (?P<spi>[0-9.eE+-]+), "
    r"peak_memory_bytes: (?P<memory>\d+)$",
    re.MULTILINE,
)


def _single(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def parse_log(spec: dict[str, str], log_dir: Path) -> Run:
    """Parse one evaluator log and enforce the manifest identity."""

    log_path = log_dir / spec["log_file"]
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    data = log_path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"NUL byte in incomplete log: {log_path}")
    text = data.decode("utf-8")
    if "Traceback (most recent call last)" in text:
        raise ValueError(f"traceback in log: {log_path}")

    artifact = _single(_ARTIFACT_RE, text, "artifact identity")
    metadata = _single(_METADATA_RE, text, "evaluation metadata")
    correct = int(_single(_CORRECT_RE, text, "correct count").group("value"))
    samples = int(_single(_EVALUATED_RE, text, "evaluated sample count").group("value"))
    digest = _single(_DIGEST_RE, text, "prediction digest").group("value")
    accuracy = float(_single(_ACCURACY_RE, text, "accuracy").group("value"))
    gpu_model = _single(_GPU_RE, text, "GPU model").group("gpu")
    if samples <= 0 or not 0 <= correct <= samples:
        raise ValueError(f"invalid correct/total in {log_path}")
    exact_accuracy = correct / samples
    if not math.isclose(accuracy, exact_accuracy, abs_tol=1e-8):
        raise ValueError(
            f"rounded accuracy mismatch in {log_path}: {accuracy} != {exact_accuracy}"
        )

    expected = {
        "source_commit": artifact.group("commit"),
        "checkpoint_sha256": artifact.group("checkpoint"),
        "dataset_fingerprint": metadata.group("fingerprint"),
        "precision": metadata.group("precision"),
        "split": metadata.group("split"),
        "expected_samples": str(samples),
    }
    for field, actual in expected.items():
        planned = spec.get(field, "")
        if planned and planned != actual:
            raise ValueError(
                f"{field} mismatch for {spec['run_id']}: {actual!r} != {planned!r}"
            )

    theta: float | None = None
    if spec["backend"] == "spiking":
        theta = float(spec["theta"])
        if not math.isclose(float(metadata.group("theta")), theta):
            raise ValueError(f"theta mismatch for {spec['run_id']}")

    semantic_max_rate = 0.0
    semantic_max_site = ""
    structural_events = 0
    for match in _CLAMP_RE.finditer(text):
        events = int(match.group("underflows")) + int(match.group("overflows"))
        values = int(match.group("values"))
        clamp_name = match.group("site").rsplit("/", 1)[-1]
        if clamp_name in STRUCTURAL_CLAMP_NAMES:
            structural_events += events
            continue
        rate = events / values if values else 0.0
        if rate > semantic_max_rate:
            semantic_max_rate = rate
            semantic_max_site = match.group("site")

    benchmark = _BENCHMARK_RE.search(text)
    benchmark_spi = float(benchmark.group("spi")) if benchmark else None
    benchmark_memory = int(benchmark.group("memory")) if benchmark else None
    return Run(
        run_id=spec["run_id"],
        stage=spec["stage"],
        backend=spec["backend"],
        theta=theta,
        split=metadata.group("split"),
        samples=samples,
        correct=correct,
        accuracy=exact_accuracy,
        prediction_sha256=digest,
        dataset_fingerprint=metadata.group("fingerprint"),
        source_commit=artifact.group("commit"),
        checkpoint_sha256=artifact.group("checkpoint"),
        gpu_model=gpu_model,
        semantic_max_rate=semantic_max_rate,
        semantic_max_site=semantic_max_site,
        structural_events=structural_events,
        benchmark_seconds_per_image=benchmark_spi,
        benchmark_peak_memory_bytes=benchmark_memory,
        log_file=str(log_path),
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    required = {
        "run_id",
        "stage",
        "backend",
        "theta",
        "split",
        "expected_samples",
        "dataset_fingerprint",
        "precision",
        "source_commit",
        "checkpoint_sha256",
        "gpu_family",
        "log_file",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid theta selection manifest: {path}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("manifest run_id values must be unique")
    return rows


def choose_theta(runs: Sequence[Run]) -> tuple[float | None, str, float]:
    """Apply the preregistered accuracy rule and upper-grid guard."""

    candidates = {
        run.theta: run
        for run in runs
        if run.stage == "selection" and run.backend == "spiking"
    }
    missing = set(BASE_THETAS).difference(candidates)
    if missing:
        raise ValueError(f"missing base theta candidates: {sorted(missing)}")
    if candidates[4000.0].accuracy - candidates[2800.0].accuracy > PLATEAU_TOLERANCE:
        extension_missing = set(EXTENSION_THETAS).difference(candidates)
        if extension_missing:
            return None, "needs_extension", max(run.accuracy for run in candidates.values())
        if candidates[8000.0].accuracy - candidates[5600.0].accuracy > PLATEAU_TOLERANCE:
            return None, "range_insufficient", max(run.accuracy for run in candidates.values())

    best = max(run.accuracy for run in candidates.values())
    selected = min(theta for theta, run in candidates.items() if run.accuracy >= best - ACCURACY_TOLERANCE)
    return selected, "selected", best


def validation_neighbors(selected: float, candidates: Iterable[float]) -> list[float]:
    ordered = sorted(set(candidates))
    index = ordered.index(selected)
    return ordered[max(0, index - 1) : min(len(ordered), index + 2)]


def validate_replay_and_validation(runs: Sequence[Run], selected: float) -> None:
    selection = [
        run
        for run in runs
        if run.stage == "selection" and run.backend == "spiking" and run.theta == selected
    ]
    replay = [run for run in runs if run.stage == "replay" and run.theta == selected]
    if len(selection) != 1 or len(replay) != 1:
        raise ValueError("selection confirmation requires one selected run and one replay")
    if (
        selection[0].correct != replay[0].correct
        or selection[0].prediction_sha256 != replay[0].prediction_sha256
    ):
        raise ValueError("selected theta replay changed correct count or predictions")

    validation = [
        run for run in runs if run.stage == "validation" and run.backend == "spiking"
    ]
    if not validation or selected not in {run.theta for run in validation}:
        raise ValueError("validation runs do not include selected theta")
    selected_accuracy = next(run.accuracy for run in validation if run.theta == selected)
    if selected_accuracy < max(run.accuracy for run in validation) - ACCURACY_TOLERANCE:
        raise ValueError("selected theta failed the validation stability tolerance")


def write_raw_csv(path: Path, runs: Sequence[Run]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for run in runs]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, runs: Sequence[Run]) -> None:
    fields = (
        "stage",
        "backend",
        "theta",
        "samples",
        "correct",
        "accuracy",
        "semantic_max_rate",
        "semantic_max_site",
        "gpu_model",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: getattr(run, field) for field in fields})


def plot_selection(path_prefix: Path, runs: Sequence[Run], selected: float) -> None:
    candidates = sorted(
        (run for run in runs if run.stage == "selection" and run.backend == "spiking"),
        key=lambda run: run.theta or 0.0,
    )
    dense = [run for run in runs if run.stage == "selection" and run.backend == "hf"]
    figure, accuracy_axis = plt.subplots(figsize=(6.4, 3.8))
    thetas = [run.theta for run in candidates]
    accuracy_axis.semilogx(thetas, [100 * run.accuracy for run in candidates], "o-", color="#235789")
    if dense:
        accuracy_axis.axhline(100 * dense[0].accuracy, color="0.35", linestyle="--", label="Dense reference")
    accuracy_axis.axvline(selected, color="#c1292e", linestyle=":", label=fr"Selected $\theta={selected:g}$")
    accuracy_axis.set_xlabel(r"Global threshold $\theta$")
    accuracy_axis.set_ylabel("Top-1 accuracy (%)")
    accuracy_axis.grid(True, which="both", alpha=0.25)
    accuracy_axis.legend(loc="best")

    rail_axis = accuracy_axis.twinx()
    rail_axis.semilogx(thetas, [run.semantic_max_rate for run in candidates], "s--", color="#f1a208", alpha=0.75)
    rail_axis.set_ylabel("Max semantic rail excursion rate")
    figure.tight_layout()
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    specs: list[dict[str, str]] = []
    for manifest in args.manifest:
        specs.extend(read_manifest(manifest))
    runs = [parse_log(spec, args.log_dir) for spec in specs]
    selected, status, best = choose_theta(runs)
    payload: dict[str, object] = {
        "format_version": 1,
        "status": status,
        "base_thetas": list(BASE_THETAS),
        "extension_thetas": list(EXTENSION_THETAS),
        "accuracy_tolerance": ACCURACY_TOLERANCE,
        "plateau_tolerance": PLATEAU_TOLERANCE,
        "best_selection_accuracy": best,
        "selected_theta": selected,
    }
    if selected is not None:
        candidate_values = [run.theta for run in runs if run.stage == "selection" and run.theta is not None]
        payload["validation_neighbors"] = validation_neighbors(selected, candidate_values)
        if args.confirm:
            validate_replay_and_validation(runs, selected)
            payload["status"] = "confirmed"
        if args.require_full:
            if not args.confirm:
                raise ValueError("full approval requires --confirm")
            validate_replay_and_validation(runs, selected)
            full = [run for run in runs if run.stage == "full"]
            if len(full) != 2 or {run.backend for run in full} != {"hf", "spiking"}:
                raise ValueError("full approval requires one dense and one spiking run")
            if any(run.samples != 50000 for run in full):
                raise ValueError("full approval requires exactly 50,000 samples per run")
            payload["status"] = "approved"
            payload["full_accuracy"] = {
                run.backend: run.accuracy for run in full
            }

    write_raw_csv(args.raw_csv, runs)
    write_summary_csv(args.summary_csv, runs)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected is not None:
        plot_selection(args.figure_prefix, runs, selected)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
