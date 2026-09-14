"""Summarize the adaptive ViT deadline-margin recovery sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ACCURACY = re.compile(rf"^Accuracy: (?P<value>{FLOAT})$", re.MULTILINE)
CONFIG = re.compile(
    rf"std_frac: (?P<frac>{FLOAT}), .*?seed: (?P<seed>\d+), .*?"
    rf"deadline_margin_std: (?P<margin>{FLOAT}), .*?"
    rf"deadline_margin_abs: (?P<margin_abs>{FLOAT})"
)
STATS = re.compile(
    r"^Gaussian\[[^]]+] events=(?P<events>\d+), "
    r"misses=(?P<misses>\d+)",
    re.MULTILINE,
)


def parse_logs(log_dir: Path) -> list[dict[str, float | int | str]]:
    """Parse every complete margin log and reject duplicate conditions."""
    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[float, int]] = set()
    for path in sorted(log_dir.glob("margin_*_seed_*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        accuracy = ACCURACY.search(text)
        config = CONFIG.search(text)
        if accuracy is None or config is None or "Traceback (most recent call last)" in text:
            continue
        condition = (float(config.group("margin")), int(config.group("seed")))
        if condition in seen:
            raise ValueError(f"duplicate margin/seed condition: {condition}")
        seen.add(condition)
        physical = list(STATS.finditer(text))
        if not physical:
            raise ValueError(f"missing Gaussian statistics in {path}")
        events = sum(int(match.group("events")) for match in physical)
        misses = sum(int(match.group("misses")) for match in physical)
        rows.append(
            {
                "margin_std": condition[0],
                "margin_abs": float(config.group("margin_abs")),
                "seed": condition[1],
                "time_noise_std_frac": float(config.group("frac")),
                "accuracy": float(accuracy.group("value")),
                "events": events,
                "misses": misses,
                "miss_rate": misses / events,
                "log_path": str(path),
            }
        )
    if not rows:
        raise ValueError(f"no complete margin logs in {log_dir}")
    return sorted(rows, key=lambda row: (float(row["margin_std"]), int(row["seed"])))


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    """Aggregate available replicas and compute Student-t intervals when possible."""
    groups: dict[float, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[float(row["margin_std"])].append(row)
    summary: list[dict[str, float | int]] = []
    for margin, replicas in sorted(groups.items()):
        accuracies = [float(row["accuracy"]) for row in replicas]
        mean = statistics.fmean(accuracies)
        if len(accuracies) >= 2:
            standard_error = statistics.stdev(accuracies) / math.sqrt(len(accuracies))
            half_width = float(student_t.ppf(0.975, len(accuracies) - 1)) * standard_error
            low, high = mean - half_width, mean + half_width
        else:
            low = high = math.nan
        events = sum(int(row["events"]) for row in replicas)
        misses = sum(int(row["misses"]) for row in replicas)
        summary.append(
            {
                "margin_std": margin,
                "margin_abs": float(replicas[0]["margin_abs"]),
                "replicas": len(replicas),
                "accuracy_mean": mean,
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "events": events,
                "misses": misses,
                "miss_rate": misses / events,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one homogeneous table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_summary(
    summary: list[dict[str, float | int]],
    *,
    baseline: float,
    recovery_tolerance: float,
    figure_prefix: Path,
) -> None:
    """Plot exploratory seed-zero points and replicated recovery candidates."""
    margins = [float(row["margin_std"]) for row in summary]
    means = [100.0 * float(row["accuracy_mean"]) for row in summary]
    fig, axis = plt.subplots(figsize=(5.2, 3.5))
    axis.plot(margins, means, color="#666666", marker="o", linewidth=1.2)
    replicated = [row for row in summary if int(row["replicas"]) >= 2]
    if replicated:
        x = [float(row["margin_std"]) for row in replicated]
        y = [100.0 * float(row["accuracy_mean"]) for row in replicated]
        yerr = [
            [100.0 * (float(row["accuracy_mean"]) - float(row["accuracy_ci95_low"])) for row in replicated],
            [100.0 * (float(row["accuracy_ci95_high"]) - float(row["accuracy_mean"])) for row in replicated],
        ]
        axis.errorbar(x, y, yerr=yerr, fmt="o", color="#1f77b4", capsize=3, label="3-seed confirmation")
    axis.axhline(100.0 * baseline, color="#222222", linestyle="--", label="clean")
    axis.axhline(100.0 * (baseline - recovery_tolerance), color="#999999", linestyle=":", label="recovery threshold")
    axis.set_xlabel(r"Deadline grace $m/\sigma_t$")
    axis.set_ylabel("ImageNet-1k top-1 (%)")
    axis.set_ylim(bottom=0.0, top=100.0)
    axis.grid(True, color="#eeeeee")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_prefix.with_suffix(".pdf"))
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and materialize the recovery artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--recovery-tolerance", type=float, default=0.01)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_logs(args.log_dir)
    summary = aggregate(rows)
    write_csv(args.raw_csv, rows)
    write_csv(args.summary_csv, summary)
    plot_summary(
        summary,
        baseline=args.baseline,
        recovery_tolerance=args.recovery_tolerance,
        figure_prefix=args.figure_prefix,
    )


if __name__ == "__main__":
    main()
