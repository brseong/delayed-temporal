#!/usr/bin/env python3
"""Validate and summarize the ViT timing-sigma/deadline-margin grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import statistics
from typing import Sequence


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
REQUIRED_FIELDS = {
    "run_id", "stage", "backend", "theta", "time_noise_std_frac",
    "time_noise_std_abs", "deadline_margin_std", "deadline_margin_abs",
    "seed", "split", "expected_samples", "dataset_fingerprint", "precision",
    "source_commit", "checkpoint_sha256", "gpu_family", "log_file",
}
HASH_FIELDS = (
    "theta_selection_sha256",
    "theta_selection_raw_sha256",
    "theta_full_manifest_sha256",
    "gpu_selection_sha256",
)
GPU_MARKERS = {
    "rtx3090": ("RTX 3090",),
    "a10": (" A10", "A10 ", "A10-SXM"),
    "rtx6000ada": ("RTX 6000 Ada",),
    "rtxa6000": ("RTX A6000",),
}
CANONICAL_FRACTIONS = (
    1.000e-10, 1.250e-10, 1.500e-10, 1.750e-10, 2.000e-10, 2.500e-10,
    3.162e-10, 4.000e-10, 5.000e-10, 6.300e-10, 8.000e-10, 1.000e-9,
)
CANONICAL_MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)

ARTIFACT_RE = re.compile(
    r"^Artifact identity — source_commit: (?P<commit>[^,]+), "
    r"checkpoint_sha256: (?P<checkpoint>\S+)$",
    re.MULTILINE,
)
METADATA_RE = re.compile(
    rf"^Evaluation metadata — model: (?P<model>.*?), dataset: (?P<dataset>.*?), "
    rf"split: (?P<split>.*?), samples: (?P<samples>\d+), theta: (?P<theta>{FLOAT}), "
    rf"precision: (?P<precision>[^,]+), source: (?P<source>.*?), "
    rf"fingerprint: (?P<fingerprint>\S+)$",
    re.MULTILINE,
)
SLURM_RE = re.compile(
    r"^Slurm identity — .* gpu_family: (?P<family>\S+)$",
    re.MULTILINE,
)
GPU_RE = re.compile(r"^GPU model: (?P<gpu>.+)$", re.MULTILINE)
CORRECT_RE = re.compile(r"^Correct: (?P<value>\d+)$", re.MULTILINE)
EVALUATED_RE = re.compile(r"^Evaluated samples: (?P<value>\d+)$", re.MULTILINE)
DIGEST_RE = re.compile(r"^Prediction SHA256: (?P<value>[0-9a-f]{64})$", re.MULTILINE)
ACCURACY_RE = re.compile(rf"^Accuracy: (?P<value>{FLOAT})$", re.MULTILINE)
GAUSSIAN_CONFIG_RE = re.compile(
    rf"^Gaussian time noise — enabled: (?P<enabled>True|False), "
    rf"std_frac: (?P<frac>{FLOAT}), identity_window: (?P<window>{FLOAT}), "
    rf"std_abs: (?P<sigma>{FLOAT}), mean_abs: (?P<mean>{FLOAT}), "
    rf"seed: (?P<seed>\d+), identity_deadline_ulp: (?P<ulp>{FLOAT}), "
    rf"std_to_identity_ulp: (?P<ratio>{FLOAT}), "
    rf"deadline_margin_std: (?P<margin>{FLOAT}), "
    rf"deadline_margin_abs: (?P<margin_abs>{FLOAT})$",
    re.MULTILINE,
)
MISMATCH_RE = re.compile(
    rf"^Static threshold mismatch — enabled: (?P<enabled>True|False), "
    rf"theta_std: (?P<std>{FLOAT}), seed: (?P<seed>\d+)$",
    re.MULTILINE,
)
SITE_RE = re.compile(
    r"^Gaussian\[(?P<site>[^]]+)] events=(?P<events>\d+), "
    r"misses=(?P<misses>\d+) \(rate=[^)]+\), "
    r"deadline_events=(?P<deadline_events>\d+) \(rate=[^)]+\), "
    rf"deadline_ulp_min=(?P<ulp_min>{FLOAT}), "
    rf"deadline_ulp_max=(?P<ulp_max>{FLOAT}), .*?"
    r"outputs=(?P<outputs>\d+), underflows=(?P<underflows>\d+) "
    r"\(rate=[^)]+\), overflows=(?P<overflows>\d+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ManifestRun:
    run_id: str
    stage: str
    backend: str
    theta: float
    time_noise_std_frac: float
    time_noise_std_abs: float
    deadline_margin_std: float
    deadline_margin_abs: float
    seed: int | None
    split: str
    expected_samples: int
    dataset_fingerprint: str
    precision: str
    source_commit: str
    checkpoint_sha256: str
    gpu_family: str
    log_file: str
    row: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class SiteCounts:
    site: str
    events: int
    misses: int
    deadline_events: int
    deadline_ulp_min: float
    deadline_ulp_max: float
    outputs: int
    underflows: int
    overflows: int


@dataclass(frozen=True)
class ParsedRun:
    run_id: str
    stage: str
    backend: str
    theta: float
    time_noise_std_frac: float
    time_noise_std_abs: float
    deadline_margin_std: float
    deadline_margin_abs: float
    seed: int | None
    split: str
    samples: int
    correct: int
    accuracy: float
    prediction_sha256: str
    dataset_fingerprint: str
    precision: str
    source_commit: str
    checkpoint_sha256: str
    gpu_family: str
    gpu_model: str
    identity_deadline_ulp: float
    time_noise_std_to_identity_ulp: float
    events: int
    misses: int
    deadline_events: int
    outputs: int
    underflows: int
    overflows: int
    log_file: str
    sites: tuple[SiteCounts, ...] = field(repr=False)


def _single(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-15):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def read_manifest(path: Path, *, require_canonical: bool = True) -> list[ManifestRun]:
    """Read a manifest and enforce its grid and immutable shared identity."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    if not rows or not REQUIRED_FIELDS.issubset(rows[0]):
        raise ValueError(f"invalid sigma-margin manifest: {path}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("manifest run_id values must be unique")
    if len({row["log_file"] for row in rows}) != len(rows):
        raise ValueError("manifest log_file values must be unique")
    for name in HASH_FIELDS:
        if name not in rows[0] or len({row[name] for row in rows}) != 1:
            raise ValueError(f"manifest must have one shared {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", rows[0][name]):
            raise ValueError(f"invalid {name}")
    for name in (
        "theta", "split", "expected_samples", "dataset_fingerprint", "precision",
        "source_commit", "checkpoint_sha256", "gpu_family",
    ):
        if len({row[name] for row in rows}) != 1:
            raise ValueError(f"manifest mixes {name}")

    parsed: list[ManifestRun] = []
    seen_conditions: set[tuple[str, str, float, float, int | None]] = set()
    for row in rows:
        parsed_seed = int(row["seed"])
        seed = None if parsed_seed == -1 else parsed_seed
        run = ManifestRun(
            run_id=row["run_id"],
            stage=row["stage"],
            backend=row["backend"],
            theta=float(row["theta"]),
            time_noise_std_frac=float(row["time_noise_std_frac"]),
            time_noise_std_abs=float(row["time_noise_std_abs"]),
            deadline_margin_std=float(row["deadline_margin_std"]),
            deadline_margin_abs=float(row["deadline_margin_abs"]),
            seed=seed,
            split=row["split"],
            expected_samples=int(row["expected_samples"]),
            dataset_fingerprint=row["dataset_fingerprint"],
            precision=row["precision"],
            source_commit=row["source_commit"],
            checkpoint_sha256=row["checkpoint_sha256"],
            gpu_family=row["gpu_family"],
            log_file=row["log_file"],
            row=row,
        )
        if Path(run.log_file).name != run.log_file:
            raise ValueError(f"log_file must be a basename: {run.log_file}")
        if run.theta <= 0 or run.expected_samples != 5000 or run.precision != "float64":
            raise ValueError(f"invalid fixed evaluation contract for {run.run_id}")
        if run.gpu_family not in GPU_MARKERS:
            raise ValueError(f"unsupported GPU family: {run.gpu_family}")
        _close(run.time_noise_std_abs, run.time_noise_std_frac * 2.0 * run.theta, "sigma")
        _close(run.deadline_margin_abs, run.deadline_margin_std * run.time_noise_std_abs, "margin")
        condition = (
            run.stage, run.backend, run.time_noise_std_frac,
            run.deadline_margin_std, run.seed,
        )
        if condition in seen_conditions:
            raise ValueError(f"duplicate condition: {condition}")
        seen_conditions.add(condition)
        parsed.append(run)

    baselines = [run for run in parsed if run.stage == "baseline"]
    if len(baselines) != 2 or {run.backend for run in baselines} != {"spiking", "hf"}:
        raise ValueError("manifest requires one clean spiking and one dense baseline")
    if any(
        run.seed is not None or run.time_noise_std_frac != 0.0
        or run.deadline_margin_std != 0.0
        for run in baselines
    ):
        raise ValueError("baseline rows must be deterministic")
    stochastic = [run for run in parsed if run.stage == "sigma_margin"]
    if any(run.backend != "spiking" or run.seed is None for run in stochastic):
        raise ValueError("stochastic rows must be seeded spiking evaluations")

    cells: dict[tuple[float, float], set[int]] = {}
    for run in stochastic:
        assert run.seed is not None
        cells.setdefault(
            (run.time_noise_std_frac, run.deadline_margin_std), set()
        ).add(run.seed)
    if any(seeds != {0, 1, 2} for seeds in cells.values()):
        raise ValueError("every sigma-margin cell must contain seeds 0, 1, and 2")
    if require_canonical:
        expected_cells = {
            (fraction, margin)
            for fraction in CANONICAL_FRACTIONS
            for margin in CANONICAL_MARGINS
        }
        if set(cells) != expected_cells or len(parsed) != 470:
            raise ValueError("manifest does not match the canonical 12x13x3 grid")
    return parsed


def parse_run_log(spec: ManifestRun, log_dir: Path) -> ParsedRun:
    """Parse one complete log and match every experimental parameter."""

    path = log_dir / spec.log_file
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"NUL byte in log: {path}")
    text = data.decode("utf-8")
    if "Traceback (most recent call last)" in text:
        raise ValueError(f"traceback in log: {path}")

    slurm = _single(SLURM_RE, text, "Slurm identity")
    artifact = _single(ARTIFACT_RE, text, "artifact identity")
    metadata = _single(METADATA_RE, text, "evaluation metadata")
    gpu_model = _single(GPU_RE, text, "GPU model").group("gpu")
    correct = int(_single(CORRECT_RE, text, "correct count").group("value"))
    samples = int(_single(EVALUATED_RE, text, "evaluated samples").group("value"))
    digest = _single(DIGEST_RE, text, "prediction digest").group("value")
    accuracy = float(_single(ACCURACY_RE, text, "accuracy").group("value"))
    gaussian = _single(GAUSSIAN_CONFIG_RE, text, "Gaussian configuration")
    mismatch = _single(MISMATCH_RE, text, "mismatch configuration")

    if slurm.group("family") != spec.gpu_family:
        raise ValueError(f"GPU family mismatch in {path}")
    if not any(marker.casefold() in gpu_model.casefold() for marker in GPU_MARKERS[spec.gpu_family]):
        raise ValueError(f"GPU model does not match {spec.gpu_family} in {path}")
    actual_identity = {
        "source_commit": artifact.group("commit"),
        "checkpoint_sha256": artifact.group("checkpoint"),
        "dataset_fingerprint": metadata.group("fingerprint"),
        "precision": metadata.group("precision"),
        "split": metadata.group("split"),
    }
    for name, actual in actual_identity.items():
        if actual != getattr(spec, name):
            raise ValueError(f"{name} mismatch in {path}: {actual!r}")
    if samples != spec.expected_samples or not 0 <= correct <= samples:
        raise ValueError(f"invalid correct/total in {path}")
    _close(float(metadata.group("theta")), spec.theta, "theta")
    _close(accuracy, correct / samples, "accuracy")
    if mismatch.group("enabled") != "False" or float(mismatch.group("std")) != 0.0:
        raise ValueError(f"static mismatch must be disabled in {path}")

    gaussian_enabled = gaussian.group("enabled") == "True"
    expected_enabled = spec.stage == "sigma_margin"
    if gaussian_enabled != expected_enabled:
        raise ValueError(f"Gaussian enabled state mismatch in {path}")
    _close(float(gaussian.group("frac")), spec.time_noise_std_frac, "r_t")
    _close(float(gaussian.group("window")), 2.0 * spec.theta, "identity window")
    _close(float(gaussian.group("sigma")), spec.time_noise_std_abs, "sigma_t")
    _close(float(gaussian.group("mean")), 0.0, "Gaussian mean")
    _close(float(gaussian.group("margin")), spec.deadline_margin_std, "margin ratio")
    _close(float(gaussian.group("margin_abs")), spec.deadline_margin_abs, "absolute margin")
    if spec.seed is not None and int(gaussian.group("seed")) != spec.seed:
        raise ValueError(f"Gaussian seed mismatch in {path}")

    identity_ulp = float(gaussian.group("ulp"))
    std_to_ulp = float(gaussian.group("ratio"))
    sites: list[SiteCounts] = []
    if expected_enabled:
        if identity_ulp <= 0.0 or not math.isfinite(std_to_ulp) or std_to_ulp <= 1.0:
            raise ValueError(f"Gaussian perturbation is not numerically resolved in {path}")
        for match in SITE_RE.finditer(text):
            value = SiteCounts(
                site=match.group("site"),
                events=int(match.group("events")),
                misses=int(match.group("misses")),
                deadline_events=int(match.group("deadline_events")),
                deadline_ulp_min=float(match.group("ulp_min")),
                deadline_ulp_max=float(match.group("ulp_max")),
                outputs=int(match.group("outputs")),
                underflows=int(match.group("underflows")),
                overflows=int(match.group("overflows")),
            )
            if value.misses > value.events or value.deadline_events > value.events:
                raise ValueError(f"invalid event counts for {value.site} in {path}")
            if value.underflows + value.overflows > value.outputs:
                raise ValueError(f"invalid output counts for {value.site} in {path}")
            sites.append(value)
        if not sites:
            raise ValueError(f"missing Gaussian site statistics in {path}")
        if len({site.site for site in sites}) != len(sites):
            raise ValueError(f"duplicate Gaussian site statistics in {path}")
        _close(std_to_ulp, spec.time_noise_std_abs / identity_ulp, "sigma/identity ULP")
    events = sum(site.events for site in sites)
    misses = sum(site.misses for site in sites)
    deadline_events = sum(site.deadline_events for site in sites)
    outputs = sum(site.outputs for site in sites)
    underflows = sum(site.underflows for site in sites)
    overflows = sum(site.overflows for site in sites)
    return ParsedRun(
        run_id=spec.run_id,
        stage=spec.stage,
        backend=spec.backend,
        theta=spec.theta,
        time_noise_std_frac=spec.time_noise_std_frac,
        time_noise_std_abs=spec.time_noise_std_abs,
        deadline_margin_std=spec.deadline_margin_std,
        deadline_margin_abs=spec.deadline_margin_abs,
        seed=spec.seed,
        split=spec.split,
        samples=samples,
        correct=correct,
        accuracy=correct / samples,
        prediction_sha256=digest,
        dataset_fingerprint=spec.dataset_fingerprint,
        precision=spec.precision,
        source_commit=spec.source_commit,
        checkpoint_sha256=spec.checkpoint_sha256,
        gpu_family=spec.gpu_family,
        gpu_model=gpu_model,
        identity_deadline_ulp=identity_ulp,
        time_noise_std_to_identity_ulp=std_to_ulp,
        events=events,
        misses=misses,
        deadline_events=deadline_events,
        outputs=outputs,
        underflows=underflows,
        overflows=overflows,
        log_file=str(path.resolve()),
        sites=tuple(sites),
    )


def aggregate_runs(runs: Sequence[ParsedRun]) -> list[dict[str, object]]:
    """Compute per-cell Student-t intervals and pooled physical rates."""

    from scipy.stats import t as student_t

    baseline = [run for run in runs if run.stage == "baseline"]
    grouped: dict[tuple[float, float], list[ParsedRun]] = {}
    for run in runs:
        if run.stage == "sigma_margin":
            grouped.setdefault(
                (run.time_noise_std_frac, run.deadline_margin_std), []
            ).append(run)
    summary: list[dict[str, object]] = []
    for run in sorted(baseline, key=lambda item: item.backend, reverse=True):
        summary.append({
            "stage": "baseline", "backend": run.backend,
            "time_noise_std_frac": 0.0, "deadline_margin_std": 0.0,
            "theta": run.theta, "time_noise_std_abs": 0.0,
            "deadline_margin_abs": 0.0, "replicas": 1,
            "accuracy_mean": run.accuracy, "accuracy_std": "",
            "accuracy_ci95_low": "", "accuracy_ci95_high": "",
            "accuracy_ci95_half_width": "", "events": 0, "misses": 0,
            "miss_rate": 0.0, "deadline_events": 0,
            "deadline_event_rate": 0.0, "outputs": 0,
            "output_underflows": 0, "output_overflows": 0,
            "rail_saturation_rate": 0.0,
        })
    for (fraction, margin), replicas in sorted(grouped.items()):
        if {run.seed for run in replicas} != {0, 1, 2}:
            raise ValueError(f"incomplete replica set for {(fraction, margin)}")
        accuracies = [run.accuracy for run in replicas]
        mean = statistics.fmean(accuracies)
        std = statistics.stdev(accuracies)
        half_width = float(student_t.ppf(0.975, 2)) * std / math.sqrt(3)
        events = sum(run.events for run in replicas)
        misses = sum(run.misses for run in replicas)
        deadline_events = sum(run.deadline_events for run in replicas)
        outputs = sum(run.outputs for run in replicas)
        underflows = sum(run.underflows for run in replicas)
        overflows = sum(run.overflows for run in replicas)
        summary.append({
            "stage": "sigma_margin", "backend": "spiking",
            "time_noise_std_frac": fraction,
            "deadline_margin_std": margin, "theta": replicas[0].theta,
            "time_noise_std_abs": replicas[0].time_noise_std_abs,
            "deadline_margin_abs": replicas[0].deadline_margin_abs,
            "replicas": 3, "accuracy_mean": mean, "accuracy_std": std,
            "accuracy_ci95_low": mean - half_width,
            "accuracy_ci95_high": mean + half_width,
            "accuracy_ci95_half_width": half_width,
            "events": events, "misses": misses,
            "miss_rate": misses / events if events else 0.0,
            "deadline_events": deadline_events,
            "deadline_event_rate": deadline_events / events if events else 0.0,
            "outputs": outputs, "output_underflows": underflows,
            "output_overflows": overflows,
            "rail_saturation_rate": (underflows + overflows) / outputs if outputs else 0.0,
        })
    return summary


def aggregate_sites(runs: Sequence[ParsedRun]) -> list[dict[str, object]]:
    """Pool physical counts by grid cell and named injection/readout site."""

    grouped: dict[tuple[float, float, str], list[SiteCounts]] = {}
    for run in runs:
        for site in run.sites:
            grouped.setdefault(
                (run.time_noise_std_frac, run.deadline_margin_std, site.site), []
            ).append(site)
    rows: list[dict[str, object]] = []
    for (fraction, margin, name), values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError(f"site {name} is missing a replica at {(fraction, margin)}")
        events = sum(value.events for value in values)
        misses = sum(value.misses for value in values)
        deadline_events = sum(value.deadline_events for value in values)
        outputs = sum(value.outputs for value in values)
        underflows = sum(value.underflows for value in values)
        overflows = sum(value.overflows for value in values)
        rows.append({
            "time_noise_std_frac": fraction,
            "deadline_margin_std": margin,
            "site": name,
            "replicas": 3,
            "events": events,
            "misses": misses,
            "miss_rate": misses / events if events else 0.0,
            "deadline_events": deadline_events,
            "deadline_event_rate": deadline_events / events if events else 0.0,
            "deadline_ulp_min": min(
                (value.deadline_ulp_min for value in values if value.deadline_ulp_min > 0),
                default=0.0,
            ),
            "deadline_ulp_max": max(value.deadline_ulp_max for value in values),
            "outputs": outputs,
            "underflows": underflows,
            "overflows": overflows,
            "rail_saturation_rate": (underflows + overflows) / outputs if outputs else 0.0,
        })
    return rows


def build_frontier(
    summary: Sequence[dict[str, object]], *, tolerance: float = 0.01
) -> dict[str, object]:
    """Find the first preregistered margin within tolerance of clean accuracy."""

    clean_rows = [
        row for row in summary
        if row["stage"] == "baseline" and row["backend"] == "spiking"
    ]
    if len(clean_rows) != 1:
        raise ValueError("summary requires exactly one clean spiking baseline")
    clean = float(clean_rows[0]["accuracy_mean"])
    threshold = clean - tolerance
    fractions = sorted({
        float(row["time_noise_std_frac"])
        for row in summary if row["stage"] == "sigma_margin"
    })
    records: list[dict[str, object]] = []
    nonmonotonic: list[dict[str, float]] = []
    for fraction in fractions:
        cells = sorted(
            (
                (float(row["deadline_margin_std"]), float(row["accuracy_mean"]))
                for row in summary
                if row["stage"] == "sigma_margin"
                and float(row["time_noise_std_frac"]) == fraction
            ),
        )
        passing = [margin for margin, accuracy in cells if accuracy >= threshold]
        recovered = min(passing) if passing else None
        if recovered is not None:
            for margin, accuracy in cells:
                if margin > recovered and accuracy < threshold:
                    nonmonotonic.append({
                        "time_noise_std_frac": fraction,
                        "deadline_margin_std": margin,
                        "accuracy_mean": accuracy,
                    })
        records.append({
            "time_noise_std_frac": fraction,
            "minimum_recovery_margin_std": recovered,
            "status": "recovered" if recovered is not None else "unrecovered",
        })
    return {
        "format_version": 1,
        "status": "complete",
        "clean_spiking_accuracy": clean,
        "recovery_tolerance": tolerance,
        "recovery_threshold": threshold,
        "frontier": records,
        "nonmonotonic_after_first_recovery": nonmonotonic,
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_outputs(
    runs: Sequence[ParsedRun],
    summary: Sequence[dict[str, object]],
    site_rows: Sequence[dict[str, object]],
    frontier: dict[str, object],
    *,
    raw_csv: Path,
    summary_csv: Path,
    site_csv: Path,
    frontier_json: Path,
) -> None:
    raw_rows: list[dict[str, object]] = []
    for run in runs:
        row = asdict(run)
        row.pop("sites")
        raw_rows.append(row)
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary)
    write_csv(site_csv, site_rows)
    frontier_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = frontier_json.with_suffix(frontier_json.suffix + ".tmp")
    temporary.write_text(json.dumps(frontier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(frontier_json)


def plot_summary(
    summary: Sequence[dict[str, object]],
    frontier: dict[str, object],
    figure_prefix: Path,
) -> None:
    """Render accuracy, confidence-width, and miss-rate heatmaps."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = [row for row in summary if row["stage"] == "sigma_margin"]
    fractions = sorted({float(row["time_noise_std_frac"]) for row in cells})
    margins = sorted({float(row["deadline_margin_std"]) for row in cells})
    index = {
        (float(row["time_noise_std_frac"]), float(row["deadline_margin_std"])): row
        for row in cells
    }
    matrices = []
    for field_name, scale in (
        ("accuracy_mean", 100.0),
        ("accuracy_ci95_half_width", 100.0),
        ("miss_rate", 100.0),
    ):
        matrices.append(np.array([
            [scale * float(index[(fraction, margin)][field_name]) for fraction in fractions]
            for margin in margins
        ]))

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    titles = ("Mean top-1 accuracy (%)", "95% CI half-width (pp)", "Pooled miss rate (%)")
    cmaps = ("viridis", "magma", "cividis")
    for axis, matrix, title, cmap in zip(axes, matrices, titles, cmaps, strict=True):
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel(r"Timing scale $r_t$")
        axis.set_xticks(range(len(fractions)))
        axis.set_xticklabels([f"{value / 1e-10:g}" for value in fractions], rotation=45, ha="right")
        axis.set_yticks(range(len(margins)))
        axis.set_yticklabels([f"{value:g}" for value in margins])
        figure.colorbar(image, ax=axis, shrink=0.82)
    axes[0].set_ylabel(r"Deadline grace $k=m/\sigma_t$")
    axes[1].set_ylabel(r"Deadline grace $k=m/\sigma_t$")
    axes[2].set_ylabel(r"Deadline grace $k=m/\sigma_t$")
    axes[0].text(
        1.0, -0.24, r"Tick labels show $r_t/10^{-10}$",
        transform=axes[0].transAxes, ha="right", fontsize=8,
    )

    frontier_y: list[float] = []
    frontier_x: list[int] = []
    for record in frontier["frontier"]:
        recovered = record["minimum_recovery_margin_std"]
        if recovered is not None:
            frontier_x.append(fractions.index(float(record["time_noise_std_frac"])))
            frontier_y.append(margins.index(float(recovered)))
    for axis in (axes[0], axes[2]):
        if frontier_x:
            axis.plot(frontier_x, frontier_y, "w.-", linewidth=1.5, markersize=6, label="1 pp recovery frontier")
            axis.legend(frameon=True, fontsize=7, loc="upper left")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(figure_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_pending_manifest(
    manifest_path: Path,
    specs: Sequence[ManifestRun],
    log_dir: Path,
    output: Path,
) -> int:
    pending: list[dict[str, str]] = []
    for spec in specs:
        try:
            parse_run_log(spec, log_dir)
        except (FileNotFoundError, UnicodeDecodeError, ValueError):
            pending.append(spec.row)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = tuple(next(csv.reader(handle, dialect="excel-tab")))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pending)
    temporary.replace(output)
    return len(pending)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--check-run-id")
    parser.add_argument("--write-pending", type=Path)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--site-csv", type=Path)
    parser.add_argument("--frontier-json", type=Path)
    parser.add_argument("--figure-prefix", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    specs = read_manifest(args.manifest)
    if args.check_run_id:
        matching = [spec for spec in specs if spec.run_id == args.check_run_id]
        if len(matching) != 1:
            raise ValueError(f"manifest does not contain one run_id={args.check_run_id}")
        parse_run_log(matching[0], args.log_dir)
        print(f"complete\t{args.check_run_id}")
        return
    if args.write_pending:
        count = write_pending_manifest(args.manifest, specs, args.log_dir, args.write_pending)
        print(f"pending\t{count}")
        return
    outputs = (
        args.raw_csv, args.summary_csv, args.site_csv,
        args.frontier_json, args.figure_prefix,
    )
    if any(path is None for path in outputs):
        raise ValueError("aggregation requires all CSV, JSON, and figure output paths")
    runs = [parse_run_log(spec, args.log_dir) for spec in specs]
    summary = aggregate_runs(runs)
    site_rows = aggregate_sites(runs)
    frontier = build_frontier(summary)
    write_outputs(
        runs, summary, site_rows, frontier,
        raw_csv=args.raw_csv,
        summary_csv=args.summary_csv,
        site_csv=args.site_csv,
        frontier_json=args.frontier_json,
    )
    plot_summary(summary, frontier, args.figure_prefix)
    print(json.dumps(frontier, sort_keys=True))


if __name__ == "__main__":
    main()
