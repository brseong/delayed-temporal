"""Validate, aggregate, and plot the maintained ViT noise fine scan."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ACCURACY_PATTERN = re.compile(rf"^Accuracy: (?P<value>{_FLOAT_PATTERN})$", re.MULTILINE)
_GAUSSIAN_CONFIG_PATTERN = re.compile(
    rf"std_frac: (?P<frac>{_FLOAT_PATTERN}), .*?"
    rf"identity_window: (?P<window>{_FLOAT_PATTERN}), .*?"
    rf"std_abs: (?P<std_abs>{_FLOAT_PATTERN}), .*?"
    rf"seed: (?P<seed>-?\d+), .*?"
    rf"identity_deadline_ulp: (?P<identity_ulp>{_FLOAT_PATTERN}), .*?"
    rf"std_to_identity_ulp: (?P<std_to_identity_ulp>{_FLOAT_PATTERN})"
)
_MISMATCH_CONFIG_PATTERN = re.compile(
    rf"^Static threshold mismatch — enabled: True, "
    rf"theta_std: (?P<std>{_FLOAT_PATTERN}), seed: (?P<seed>-?\d+)$",
    re.MULTILINE,
)
_EVALUATION_METADATA_PATTERN = re.compile(
    rf"^Evaluation metadata — model: (?P<model>.*?), "
    rf"dataset: (?P<dataset>.*?), split: (?P<split>.*?), "
    rf"samples: (?P<samples>\d+), theta: (?P<theta>{_FLOAT_PATTERN}), "
    rf"precision: (?P<precision>\S+)$",
    re.MULTILINE,
)
_GAUSSIAN_STATS_PATTERN = re.compile(
    r"^Gaussian\[(?P<site>[^]]+)] "
    r"events=(?P<events>\d+), misses=(?P<misses>\d+) .*?"
    r"deadline_events=(?P<deadline_events>\d+) .*?"
    rf"deadline_ulp_min=(?P<ulp_min>{_FLOAT_PATTERN}), "
    rf"deadline_ulp_max=(?P<ulp_max>{_FLOAT_PATTERN}), .*?"
    r"outputs=(?P<outputs>\d+), underflows=(?P<underflows>\d+) .*?"
    r"overflows=(?P<overflows>\d+)",
    re.MULTILINE,
)
_WANDB_URL_PATTERN = re.compile(
    r"View run .*? at: (?P<url>https://wandb\.ai/\S+)"
)


@dataclass(frozen=True)
class ExpectedRun:
    """One planned scan condition loaded from the shell-generated manifest."""

    axis: str
    magnitude: float
    seed: int | None
    experiment_name: str
    log_file: str


@dataclass(frozen=True)
class ParsedRun:
    """Validated measurements extracted from one completed evaluator log."""

    axis: str
    magnitude: float
    seed: int | None
    experiment_name: str
    accuracy: float
    model_id: str
    dataset_id: str
    dataset_split: str
    evaluation_samples: int
    theta: float
    precision: str
    time_noise_std_frac: float
    identity_time_window: float
    time_noise_std_abs: float
    identity_deadline_ulp: float
    time_noise_std_to_identity_ulp: float
    events: int
    misses: int
    deadline_events: int
    deadline_ulp_min: float
    deadline_ulp_max: float
    outputs: int
    output_underflows: int
    output_overflows: int
    wandb_url: str
    log_path: str


def read_manifest(path: Path) -> list[ExpectedRun]:
    """Load and validate the complete expected-run manifest.

    The manifest is the single source of truth shared by scheduling, resume, and
    aggregation. Strict validation prevents duplicate log names or an accidentally
    edited condition from being hidden by otherwise successful output files.

    Args:
        path: Tab-separated manifest written by ``noise_scan_vit.sh``.

    Returns:
        Expected conditions in their deterministic presentation order.

    Raises:
        ValueError: If a row, axis, seed, or uniqueness constraint is invalid.
    """
    # Parse the tab-separated file explicitly instead of inferring conditions from
    # filenames. This keeps scientific parameters independent of naming format.
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    if not rows:
        raise ValueError(f"scan manifest is empty: {path}")

    # Normalize numeric fields once and reject path traversal before any log is read.
    expected: list[ExpectedRun] = []
    seen_logs: set[str] = set()
    seen_conditions: set[tuple[str, float, int | None]] = set()
    for row in rows:
        axis = row["axis"]
        if axis not in {"baseline", "gaussian", "mismatch"}:
            raise ValueError(f"unknown scan axis {axis!r}")
        magnitude = float(row["magnitude"])
        if not math.isfinite(magnitude) or magnitude < 0.0:
            raise ValueError(f"invalid scan magnitude {row['magnitude']!r}")
        seed_text = row["seed"].strip()
        seed = int(seed_text) if seed_text else None
        if axis in {"gaussian", "mismatch"} and seed is None:
            raise ValueError(f"{axis} manifest rows require a replica seed")
        if axis == "baseline" and seed is not None:
            raise ValueError("baseline manifest rows must not declare a seed")
        log_file = row["log_file"]
        if Path(log_file).name != log_file:
            raise ValueError(f"manifest log_file must be a basename: {log_file!r}")

        condition = (axis, magnitude, seed)
        if log_file in seen_logs or condition in seen_conditions:
            raise ValueError(f"duplicate scan condition or log file: {row}")
        seen_logs.add(log_file)
        seen_conditions.add(condition)
        expected.append(
            ExpectedRun(
                axis=axis,
                magnitude=magnitude,
                seed=seed,
                experiment_name=row["experiment_name"],
                log_file=log_file,
            )
        )

    # Lock the intended experimental design: one baseline and a common seed set at
    # every magnitude on both stochastic axes.
    baselines = [run for run in expected if run.axis == "baseline"]
    if len(baselines) != 1 or baselines[0].magnitude != 0.0:
        raise ValueError("manifest must contain exactly one zero-magnitude baseline")
    seeds_by_axis: dict[str, dict[float, set[int]]] = {
        "gaussian": defaultdict(set),
        "mismatch": defaultdict(set),
    }
    for run in expected:
        if run.axis in seeds_by_axis:
            assert run.seed is not None
            seeds_by_axis[run.axis][run.magnitude].add(run.seed)
    active_seed_sets: list[frozenset[int]] = []
    for axis, magnitude_seeds in seeds_by_axis.items():
        if not magnitude_seeds:
            if axis == "gaussian":
                raise ValueError("manifest requires at least one gaussian condition")
            continue
        unique_seed_sets = {frozenset(value) for value in magnitude_seeds.values()}
        if len(unique_seed_sets) != 1:
            raise ValueError(f"every {axis} magnitude must use the same seed set")
        if len(next(iter(magnitude_seeds.values()))) < 2:
            raise ValueError(f"{axis} confidence intervals require at least two seeds")
        active_seed_sets.append(frozenset(next(iter(magnitude_seeds.values()))))
    if len(set(active_seed_sets)) != 1:
        raise ValueError("Gaussian and mismatch axes must use the same replica seeds")
    return expected


def parse_run_log(expected: ExpectedRun, log_dir: Path) -> ParsedRun:
    """Parse one evaluator log and enforce condition-specific completeness.

    Every run must report model and dataset identity. Gaussian runs additionally
    report the derived timing scale plus per-site physical statistics; mismatch
    runs report their independently seeded frozen-offset configuration.

    Args:
        expected: Manifest condition the log is required to represent.
        log_dir: Directory containing the manifest's log basenames.

    Returns:
        A normalized measurement row ready for aggregation.

    Raises:
        ValueError: If evaluation failed, output is incomplete, or logged Gaussian
            parameters disagree with the manifest.
    """
    log_path = log_dir / expected.log_file
    if not log_path.is_file():
        raise ValueError(f"missing scan log: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # A traceback invalidates the run even if buffered output happens to contain an
    # older-looking metric. The final exact Accuracy line is the completion marker.
    if "Traceback (most recent call last)" in text:
        raise ValueError(f"evaluator traceback in {log_path}")
    accuracy_matches = list(_ACCURACY_PATTERN.finditer(text))
    if not accuracy_matches:
        raise ValueError(f"missing final accuracy in {log_path}")
    accuracy = float(accuracy_matches[-1].group("value"))
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy outside [0, 1] in {log_path}: {accuracy}")

    # W&B emits the run URL near startup and completion. Retain the last occurrence
    # so the raw CSV points directly to the finalized external record.
    url_matches = list(_WANDB_URL_PATTERN.finditer(text))
    if not url_matches:
        raise ValueError(f"missing W&B run URL in {log_path}")
    wandb_url = url_matches[-1].group("url")

    metadata_matches = list(_EVALUATION_METADATA_PATTERN.finditer(text))
    if not metadata_matches:
        raise ValueError(f"missing evaluation metadata in {log_path}")
    metadata = metadata_matches[-1]
    model_id = metadata.group("model")
    dataset_id = metadata.group("dataset")
    dataset_split = metadata.group("split")
    evaluation_samples = int(metadata.group("samples"))
    theta = float(metadata.group("theta"))
    precision = metadata.group("precision")

    time_noise_std_frac = 0.0
    identity_time_window = 0.0
    time_noise_std_abs = 0.0
    identity_deadline_ulp = 0.0
    time_noise_std_to_identity_ulp = 0.0
    events = misses = outputs = underflows = overflows = 0
    deadline_events = 0
    deadline_ulp_min = math.inf
    deadline_ulp_max = 0.0
    if expected.axis == "gaussian":
        # The evaluator owns fraction-to-absolute conversion. Parse and verify its
        # emitted values rather than recomputing silently in the summarizer.
        config_matches = list(_GAUSSIAN_CONFIG_PATTERN.finditer(text))
        if not config_matches:
            raise ValueError(f"missing Gaussian configuration in {log_path}")
        config = config_matches[-1]
        time_noise_std_frac = float(config.group("frac"))
        identity_time_window = float(config.group("window"))
        time_noise_std_abs = float(config.group("std_abs"))
        identity_deadline_ulp = float(config.group("identity_ulp"))
        time_noise_std_to_identity_ulp = float(
            config.group("std_to_identity_ulp")
        )
        logged_seed = int(config.group("seed"))
        if not math.isclose(
            time_noise_std_frac,
            expected.magnitude,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"Gaussian fraction mismatch in {log_path}")
        if logged_seed != expected.seed:
            raise ValueError(f"Gaussian seed mismatch in {log_path}")
        if not math.isclose(
            time_noise_std_abs,
            time_noise_std_frac * identity_time_window,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"Gaussian absolute std mismatch in {log_path}")
        if identity_deadline_ulp <= 0.0:
            raise ValueError(f"invalid identity deadline ULP in {log_path}")
        if not math.isclose(
            time_noise_std_to_identity_ulp,
            time_noise_std_abs / identity_deadline_ulp,
            rel_tol=1.0e-9,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"Gaussian identity ULP ratio mismatch in {log_path}")

        # Sum raw counts across sites before forming rates. This weights every event
        # or physical output equally instead of averaging incomparable site rates.
        stat_matches = list(_GAUSSIAN_STATS_PATTERN.finditer(text))
        if not stat_matches:
            raise ValueError(f"missing Gaussian physical statistics in {log_path}")
        for match in stat_matches:
            site_events = int(match.group("events"))
            site_ulp_min = float(match.group("ulp_min"))
            site_ulp_max = float(match.group("ulp_max"))
            events += site_events
            misses += int(match.group("misses"))
            deadline_events += int(match.group("deadline_events"))
            if site_events:
                if site_ulp_min <= 0.0 or site_ulp_max < site_ulp_min:
                    raise ValueError(f"invalid Gaussian deadline ULP range in {log_path}")
                deadline_ulp_min = min(deadline_ulp_min, site_ulp_min)
                deadline_ulp_max = max(deadline_ulp_max, site_ulp_max)
            outputs += int(match.group("outputs"))
            underflows += int(match.group("underflows"))
            overflows += int(match.group("overflows"))
        if events and (deadline_ulp_min <= 0.0 or deadline_ulp_max < deadline_ulp_min):
            raise ValueError(f"invalid Gaussian deadline ULP range in {log_path}")
    elif expected.axis == "mismatch":
        mismatch_matches = list(_MISMATCH_CONFIG_PATTERN.finditer(text))
        if not mismatch_matches:
            raise ValueError(f"missing static mismatch configuration in {log_path}")
        mismatch = mismatch_matches[-1]
        if not math.isclose(
            float(mismatch.group("std")),
            expected.magnitude,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"static mismatch magnitude mismatch in {log_path}")
        if int(mismatch.group("seed")) != expected.seed:
            raise ValueError(f"static mismatch seed mismatch in {log_path}")

    # Store an absolute path so result tables remain traceable when opened outside
    # the repository root or copied beside publication artifacts.
    return ParsedRun(
        axis=expected.axis,
        magnitude=expected.magnitude,
        seed=expected.seed,
        experiment_name=expected.experiment_name,
        accuracy=accuracy,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        evaluation_samples=evaluation_samples,
        theta=theta,
        precision=precision,
        time_noise_std_frac=time_noise_std_frac,
        identity_time_window=identity_time_window,
        time_noise_std_abs=time_noise_std_abs,
        identity_deadline_ulp=identity_deadline_ulp,
        time_noise_std_to_identity_ulp=time_noise_std_to_identity_ulp,
        events=events,
        misses=misses,
        deadline_events=deadline_events,
        deadline_ulp_min=0.0 if math.isinf(deadline_ulp_min) else deadline_ulp_min,
        deadline_ulp_max=deadline_ulp_max,
        outputs=outputs,
        output_underflows=underflows,
        output_overflows=overflows,
        wandb_url=wandb_url,
        log_path=str(log_path.resolve()),
    )


def aggregate_runs(runs: Sequence[ParsedRun]) -> list[dict[str, object]]:
    """Aggregate raw runs by axis and magnitude with seed-level uncertainty.

    Both stochastic axes receive a two-sided 95% Student-t confidence interval over
    replica seeds. Gaussian physical rates pool raw counts across sites and replicas;
    the deterministic baseline remains a singleton with blank intervals.

    Args:
        runs: Complete parsed measurements from one manifest.

    Returns:
        Ordered dictionaries suitable for the summary CSV and figure.

    Raises:
        ValueError: If a Gaussian group has fewer than two replicas.
    """
    # Group only conditions that share both physical axis and magnitude. This keeps
    # component-specific parameters from being treated as interchangeable units.
    grouped: dict[tuple[str, float], list[ParsedRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.axis, run.magnitude)].append(run)

    # Emit baseline first, followed by increasing Gaussian and mismatch magnitudes,
    # matching both the experiment manifest and the two-panel plot presentation.
    axis_order = {"baseline": 0, "gaussian": 1, "mismatch": 2}
    summary: list[dict[str, object]] = []
    for (axis, magnitude), replicas in sorted(
        grouped.items(), key=lambda item: (axis_order[item[0][0]], item[0][1])
    ):
        accuracies = [run.accuracy for run in replicas]
        accuracy_mean = statistics.fmean(accuracies)
        accuracy_std: float | None = None
        ci_low: float | None = None
        ci_high: float | None = None
        if axis in {"gaussian", "mismatch"}:
            if len(replicas) < 2:
                raise ValueError(f"{axis} groups require at least two replicas")
            accuracy_std = statistics.stdev(accuracies)
            critical = float(student_t.ppf(0.975, df=len(replicas) - 1))
            half_width = critical * accuracy_std / math.sqrt(len(replicas))
            ci_low = accuracy_mean - half_width
            ci_high = accuracy_mean + half_width

        # Pool event and output denominators before division. Empty denominators are
        # represented as zero only for axes that do not produce Gaussian statistics.
        events = sum(run.events for run in replicas)
        misses = sum(run.misses for run in replicas)
        outputs = sum(run.outputs for run in replicas)
        underflows = sum(run.output_underflows for run in replicas)
        overflows = sum(run.output_overflows for run in replicas)
        summary.append(
            {
                "axis": axis,
                "magnitude": magnitude,
                "replicas": len(replicas),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "model_id": replicas[0].model_id,
                "dataset_id": replicas[0].dataset_id,
                "dataset_split": replicas[0].dataset_split,
                "evaluation_samples": replicas[0].evaluation_samples,
                "theta": replicas[0].theta,
                "precision": replicas[0].precision,
                "time_noise_std_abs_mean": statistics.fmean(
                    run.time_noise_std_abs for run in replicas
                ),
                "events": events,
                "misses": misses,
                "miss_rate": misses / events if events else 0.0,
                "deadline_events": sum(run.deadline_events for run in replicas),
                "deadline_event_rate": (
                    sum(run.deadline_events for run in replicas) / events
                    if events
                    else 0.0
                ),
                "deadline_ulp_min": min(
                    (run.deadline_ulp_min for run in replicas if run.deadline_ulp_min > 0.0),
                    default=0.0,
                ),
                "deadline_ulp_max": max(
                    (run.deadline_ulp_max for run in replicas),
                    default=0.0,
                ),
                "outputs": outputs,
                "output_underflows": underflows,
                "output_underflow_rate": underflows / outputs if outputs else 0.0,
                "output_overflows": overflows,
                "output_overflow_rate": overflows / outputs if outputs else 0.0,
            }
        )
    return summary


def write_csv_files(
    runs: Sequence[ParsedRun],
    summary: Sequence[dict[str, object]],
    *,
    raw_csv: Path,
    summary_csv: Path,
) -> None:
    """Write traceable raw and aggregate CSV files using atomic replacement.

    Args:
        runs: Validated per-run records.
        summary: Aggregate records produced by :func:`aggregate_runs`.
        raw_csv: Destination for one row per evaluator process.
        summary_csv: Destination for one row per axis and magnitude.
    """
    raw_fields = (
        "axis", "magnitude", "seed", "experiment_name", "accuracy",
        "model_id", "dataset_id", "dataset_split", "evaluation_samples",
        "theta", "precision",
        "time_noise_std_frac", "identity_time_window", "time_noise_std_abs",
        "identity_deadline_ulp", "time_noise_std_to_identity_ulp",
        "events", "misses", "miss_rate", "deadline_events",
        "deadline_event_rate", "deadline_ulp_min", "deadline_ulp_max",
        "outputs", "output_underflows",
        "output_underflow_rate", "output_overflows", "output_overflow_rate",
        "wandb_url", "log_path",
    )

    # Construct detached row dictionaries first so a serialization error cannot
    # leave either destination half-written.
    raw_rows: list[dict[str, object]] = []
    for run in runs:
        raw_rows.append(
            {
                "axis": run.axis,
                "magnitude": run.magnitude,
                "seed": "" if run.seed is None else run.seed,
                "experiment_name": run.experiment_name,
                "accuracy": run.accuracy,
                "model_id": run.model_id,
                "dataset_id": run.dataset_id,
                "dataset_split": run.dataset_split,
                "evaluation_samples": run.evaluation_samples,
                "theta": run.theta,
                "precision": run.precision,
                "time_noise_std_frac": run.time_noise_std_frac,
                "identity_time_window": run.identity_time_window,
                "time_noise_std_abs": run.time_noise_std_abs,
                "identity_deadline_ulp": run.identity_deadline_ulp,
                "time_noise_std_to_identity_ulp": run.time_noise_std_to_identity_ulp,
                "events": run.events,
                "misses": run.misses,
                "miss_rate": run.misses / run.events if run.events else 0.0,
                "deadline_events": run.deadline_events,
                "deadline_event_rate": (
                    run.deadline_events / run.events if run.events else 0.0
                ),
                "deadline_ulp_min": run.deadline_ulp_min,
                "deadline_ulp_max": run.deadline_ulp_max,
                "outputs": run.outputs,
                "output_underflows": run.output_underflows,
                "output_underflow_rate": (
                    run.output_underflows / run.outputs if run.outputs else 0.0
                ),
                "output_overflows": run.output_overflows,
                "output_overflow_rate": (
                    run.output_overflows / run.outputs if run.outputs else 0.0
                ),
                "wandb_url": run.wandb_url,
                "log_path": run.log_path,
            }
        )

    # Write beside each target, flush through close, and replace atomically. A failed
    # aggregation therefore preserves the last complete tables from an earlier run.
    for destination, fields, rows in (
        (raw_csv, raw_fields, raw_rows),
        (summary_csv, tuple(summary[0].keys()), list(summary)),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)


def plot_results(
    summary: Sequence[dict[str, object]],
    *,
    figure_prefix: Path,
    model_label: str = "ViT-B/16",
    archive_existing: bool = True,
) -> None:
    """Render Gaussian robustness and an optional static-mismatch panel.

    Gaussian and static-mismatch points display the seed mean and 95% Student-t
    interval. Both panels share the same deterministic evaluation baseline.

    Args:
        summary: Aggregate records containing baseline and both maintained axes.
        figure_prefix: Output path without extension; PNG and PDF are written.
        model_label: Human-readable architecture name used only in the figure title.
        archive_existing: Preserve pre-Gaussian canonical figures before replacing
            them. Verification disables this side effect for temporary fixtures.

    Raises:
        ValueError: If the baseline or Gaussian plotting axis is absent.
    """
    # Reject an empty display label before creating artifacts. This keeps a
    # follow-up architecture explicit without coupling parsing to checkpoint paths.
    model_label = model_label.strip()
    if not model_label:
        raise ValueError("model_label must not be empty")

    # Split component-specific magnitudes instead of implying that timing fraction
    # and threshold mismatch share one physical x-axis merely because both are small.
    baseline_rows = [row for row in summary if row["axis"] == "baseline"]
    gaussian_rows = [row for row in summary if row["axis"] == "gaussian"]
    mismatch_rows = [row for row in summary if row["axis"] == "mismatch"]
    if len(baseline_rows) != 1 or not gaussian_rows:
        raise ValueError("plot requires baseline and Gaussian results")
    baseline = float(baseline_rows[0]["accuracy_mean"])

    # Preserve ignored legacy artifacts once, then update canonical PNG/PDF only
    # after every log and table has already validated successfully.
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    if archive_existing:
        for extension in ("png", "pdf"):
            current = figure_prefix.with_suffix(f".{extension}")
            legacy = figure_prefix.with_name(
                f"{figure_prefix.name}_legacy_pre_gaussian"
            ).with_suffix(f".{extension}")
            if current.exists() and not legacy.exists():
                shutil.copy2(current, legacy)

    plt.rcParams.update(
        {"font.size": 10.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.8}
    )
    # A Gaussian-only refinement gets one full-width panel. The canonical scan
    # retains two panels when static mismatch is present, so no empty subplot or
    # implied rerun appears in the lower-scale figure.
    panel_count = 2 if mismatch_rows else 1
    fig, axes_object = plt.subplots(
        1,
        panel_count,
        figsize=(9.2 if panel_count == 2 else 5.2, 4.2),
        sharey=True,
    )
    axes = [axes_object] if panel_count == 1 else list(axes_object)

    # The Gaussian panel uses asymmetric arrays so the exact raw Student-t bounds
    # remain visible even when seed variation is not symmetric around sampled points.
    gx = [float(row["magnitude"]) for row in gaussian_rows]
    gy = [float(row["accuracy_mean"]) for row in gaussian_rows]
    glow = [float(row["accuracy_ci95_low"]) for row in gaussian_rows]
    ghigh = [float(row["accuracy_ci95_high"]) for row in gaussian_rows]
    axes[0].errorbar(
        gx,
        gy,
        yerr=(
            [mean - low for mean, low in zip(gy, glow, strict=True)],
            [high - mean for mean, high in zip(gy, ghigh, strict=True)],
        ),
        color="#0072B2",
        marker="o",
        markersize=5.5,
        linewidth=1.8,
        capsize=3,
        label="Gaussian mean ± 95% t-CI",
    )
    axes[0].set_title("Gaussian spike-time noise")
    axes[0].set_xlabel(r"time-noise fraction $r_t$")
    axes[0].legend(frameon=False, fontsize=8.5)

    if mismatch_rows:
        # Static offsets are frozen within a replica but independently resampled
        # across seeds, so uncertainty is presented identically to timing noise.
        mx = [float(row["magnitude"]) for row in mismatch_rows]
        my = [float(row["accuracy_mean"]) for row in mismatch_rows]
        mlow = [float(row["accuracy_ci95_low"]) for row in mismatch_rows]
        mhigh = [float(row["accuracy_ci95_high"]) for row in mismatch_rows]
        axes[1].errorbar(
            mx,
            my,
            yerr=(
                [mean - low for mean, low in zip(my, mlow, strict=True)],
                [high - mean for mean, high in zip(my, mhigh, strict=True)],
            ),
            color="#009E73",
            marker="^",
            markersize=5.5,
            linewidth=1.8,
            capsize=3,
            label="Mismatch mean ± 95% t-CI",
        )
        axes[1].set_title("Static threshold mismatch")
        axes[1].set_xlabel(r"relative threshold std $\sigma_\theta$")
        axes[1].legend(frameon=False, fontsize=8.5)

    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(baseline, color="#8a8a8a", linestyle="--", linewidth=1.0)
        axis.grid(True, which="major", color="#e6e6e6", linewidth=0.7)
        axis.grid(True, which="minor", color="#f2f2f2", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    evaluation_samples = int(baseline_rows[0]["evaluation_samples"])
    sample_label = (
        "5k subset" if evaluation_samples == 5000 else f"{evaluation_samples:,} images"
    )
    axes[0].set_ylabel(f"ImageNet top-1 accuracy ({sample_label})")
    axes[0].set_ylim(-0.02, min(1.0, max(0.9, baseline + 0.04)))
    fig.suptitle(
        f"Spiking {model_label} robustness under maintained non-idealities"
    )
    fig.tight_layout()

    # Save both inspection and publication formats from the same figure state, then
    # close it so repeated summarization does not retain Matplotlib global objects.
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_noise_scan(
    *,
    log_dir: Path,
    manifest: Path,
    raw_csv: Path,
    summary_csv: Path,
    figure_prefix: Path,
    model_label: str = "ViT-B/16",
    archive_existing: bool = True,
) -> tuple[list[ParsedRun], list[dict[str, object]]]:
    """Validate a complete scan and publish its tables and figures.

    No output is updated until every manifest row has a complete matching log. This
    makes the function both the final experiment validator and the resume gate that
    prevents partial scans from being presented as finished results.

    Returns:
        The validated raw and aggregate records, primarily for permanent tests.
    """
    # Parse the complete manifest before touching output artifacts. Any duplicate or
    # malformed planned condition is therefore rejected independently of log state.
    expected_runs = read_manifest(manifest)
    parsed_runs = [parse_run_log(run, log_dir) for run in expected_runs]
    identities = {
        (
            run.model_id,
            run.dataset_id,
            run.dataset_split,
            run.evaluation_samples,
            run.theta,
            run.precision,
        )
        for run in parsed_runs
    }
    if len(identities) != 1:
        raise ValueError("scan logs do not share one model/dataset/numerical identity")

    # Aggregate in memory first, then publish tables before figures. Plotting only
    # sees complete, serialized scientific results rather than ad-hoc log parsing.
    summary = aggregate_runs(parsed_runs)
    write_csv_files(
        parsed_runs,
        summary,
        raw_csv=raw_csv,
        summary_csv=summary_csv,
    )
    plot_results(
        summary,
        figure_prefix=figure_prefix,
        model_label=model_label,
        archive_existing=archive_existing,
    )
    return parsed_runs, summary


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit scan input and artifact destinations."""
    # Keep paths explicit so the same aggregator can validate tagged reruns or
    # synthetic verification fixtures without relying on the process directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument("--model-label", default="ViT-B/16")
    parser.add_argument(
        "--archive-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preserve an existing canonical figure before replacement",
    )

    # Accept an injected argv in verification while preserving ordinary command-line
    # behavior for the sweep shell script.
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run complete scan validation, aggregation, and rendering."""
    # Resolve arguments once and pass named paths into the side-effectful boundary so
    # tests can call the underlying function without patching process globals.
    args = parse_arguments(argv)
    runs, summary = summarize_noise_scan(
        log_dir=args.log_dir,
        manifest=args.manifest,
        raw_csv=args.raw_csv,
        summary_csv=args.summary_csv,
        figure_prefix=args.figure_prefix,
        model_label=args.model_label,
        archive_existing=args.archive_existing,
    )

    # Print a compact completion record for the shell log and human monitoring. CSVs
    # remain the authoritative detailed output.
    print(
        f"Validated {len(runs)} runs across {len(summary)} conditions; "
        f"wrote {args.raw_csv}, {args.summary_csv}, and {args.figure_prefix}.{{png,pdf}}"
    )


if __name__ == "__main__":
    main()
