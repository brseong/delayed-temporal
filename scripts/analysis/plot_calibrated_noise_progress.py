"""Draw a calibrated noise snapshot without modifying the running experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.summarize_sigma_margin_sweep import (
    aggregate_runs, parse_run_log, read_manifest, sha256_file, write_csv,
)


def collect_snapshot(root: Path):
    """Validate finished logs; never treat partial batch results as replicas."""
    experiment = json.loads((root / "experiment.json").read_text())
    manifest = root / "manifests/grid.tsv"
    specs = read_manifest(manifest, require_canonical=False)
    table_hash = sha256_file(root / "calibration.json")
    accepted = []
    log_hashes = {}
    for spec in specs:
        if spec.source_commit != experiment["source_commit"]:
            raise ValueError("Experiment and manifest source identity mismatch")
        if spec.row.get("calibration_sha256") != table_hash:
            raise ValueError("Calibration identity mismatch")
        path = root / "logs" / spec.log_file
        if not path.exists():
            continue
        run = parse_run_log(spec, root / "logs")
        if spec.backend == "hf":
            if sha256_file(path) != experiment["dense_reference_log_sha256"]:
                raise ValueError("Dense reference identity mismatch")
        else:
            expected = f"Calibration identity — mode: validate, sha256: {table_hash}"
            if path.read_text().count(expected) != 1:
                raise ValueError("Completed log has a different calibration identity")
        accepted.append(run)
        log_hashes[spec.run_id] = sha256_file(path)
    baselines = [run for run in accepted if run.stage == "baseline"]
    if {run.backend for run in baselines} != {"spiking", "hf"}:
        raise ValueError("Both completed baselines are required")
    return experiment, specs, accepted, table_hash, log_hashes


def plot_snapshot(root: Path, output: Path, fraction: float) -> dict:
    """Keep incomplete seed groups separate from three-seed estimates."""
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("Timing noise fraction must be finite and positive")
    experiment, specs, runs, table_hash, log_hashes = collect_snapshot(root)
    selected = [run for run in runs if run.stage == "sigma_margin"
                and run.time_noise_std_frac == fraction]
    if not selected:
        raise ValueError("No completed replica at this timing noise fraction")
    groups = {}
    for run in selected:
        groups.setdefault(run.deadline_margin_std, []).append(run)
    baselines = [run for run in runs if run.stage == "baseline"]
    complete = {margin: replicas for margin, replicas in groups.items()
                if len(replicas) == 3 and {r.seed for r in replicas} == {0, 1, 2}}
    summary = aggregate_runs(baselines + [r for rs in complete.values() for r in rs])
    estimates = {float(row["deadline_margin_std"]): row for row in summary
                 if row["stage"] == "sigma_margin"}
    source_commit = experiment["source_commit"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.25))
    fig.subplots_adjust(left=.058, right=.988, top=.75, bottom=.24, wspace=.30)
    color = "#2466a8"
    margins = sorted(groups)
    xleft, xright = margins[0] - .16, margins[-1] + .16
    if len(margins) == 1:
        xleft, xright = margins[0] - .35, margins[0] + .35
    for ax in axes:
        ax.set_xlim(xleft, xright)
        ax.set_xticks(margins)
        ax.set_xticklabels([f"{x:g}" for x in margins])
        ax.set_xlabel("Deadline margin / noise standard deviation", labelpad=10)
        ax.grid(axis="y", color="#dfe5eb", alpha=.85)
        ax.set_axisbelow(True)

    for ax in axes[:2]:
        for margin, replicas in sorted(groups.items()):
            done = margin in complete
            ax.scatter([margin] * len(replicas), [100*r.accuracy for r in replicas],
                       s=34, facecolors="#8193a8" if done else "white",
                       edgecolors="#65768a", linewidths=1.3, zorder=3)
            if done:
                estimate = estimates[margin]
                ax.errorbar(margin, 100*estimate["accuracy_mean"],
                            yerr=100*estimate["accuracy_ci95_half_width"],
                            color=color, marker="D", markersize=6, capsize=4,
                            linewidth=1.8, zorder=4)
        ax.set_ylabel("Top-1 accuracy (%)")
    clean = next(100*r.accuracy for r in baselines if r.backend == "spiking")
    dense = next(100*r.accuracy for r in baselines if r.backend == "hf")
    axes[0].axhline(dense, color="#b5651d", linestyle=":", linewidth=1.6,
                   label=f"Dense reference: {dense:.2f}%")
    axes[0].axhline(clean, color="#437c54", linestyle="--", linewidth=1.6,
                   label=f"Clean spiking: {clean:.2f}%")
    axes[0].set_ylim(-2, min(102, max(90, dense + 5)))
    axes[0].set_title("Accuracy vs. clean references", loc="left", fontweight="bold", pad=14)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5,.89), frameon=False, fontsize=9)
    upper = max([100*r.accuracy for r in selected] +
                [100*(s["accuracy_mean"] + s["accuracy_ci95_half_width"]) for s in estimates.values()])
    lower = min([0.] + [100*(s["accuracy_mean"] - s["accuracy_ci95_half_width"])
                       for s in estimates.values()])
    axes[1].set_ylim(min(-.025, lower - .025), max(.4, upper * 1.25))
    axes[1].set_title("Noisy accuracy: detail", loc="left", fontweight="bold", pad=14)
    for margin, replicas in sorted(groups.items()):
        axes[1].text(margin, .98, f"{len(replicas)}/3 seeds",
                     transform=axes[1].get_xaxis_transform(), ha="center", va="top",
                     fontsize=9, color="#566577")

    for kind, label, shade, marker in (
        ("miss", "Deadline-miss rate", "#d37521", "o"),
        ("saturation", "Pre-clamp rail-saturation rate", "#8160a4", "^"),
    ):
        for margin, replicas in sorted(groups.items()):
            if kind == "miss":
                rate = sum(r.misses for r in replicas) / sum(r.events for r in replicas)
            else:
                rate = sum(r.underflows+r.overflows for r in replicas) / sum(r.outputs for r in replicas)
            if rate > 0:
                axes[2].plot(margin, rate*100, marker=marker, markersize=7,
                             markerfacecolor=shade if margin in complete else "white",
                             markeredgecolor=shade, linestyle="none")
        axes[2].plot([], [], color=shade, marker=marker, linestyle="none", label=label)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Pooled rate (%) — logarithmic scale")
    axes[2].set_title("Physical counts", loc="left", fontweight="bold", pad=14)
    axes[2].legend(loc="upper right", bbox_to_anchor=(1, -.23), frameon=False, fontsize=8.5)
    axes[2].grid(axis="y", which="minor", color="#edf0f4", alpha=.6)

    theta = float(experiment["theta"])
    fig.suptitle("ViT-B/16 — calibrated noise experiment: results so far",
                 x=.058, ha="left", y=.97, fontsize=16, fontweight="bold")
    fig.text(.058, .855, f"Validation 5k  |  float64  |  theta = {theta:g}  |  "
             f"Timing noise r_t = {fraction:g}  |  sigma_t = {2*theta*fraction:g}",
             color="#465366", fontsize=10.5)
    handles = [
        Line2D([], [], color=color, marker="D", linestyle="none", label="3-seed mean + 95% Student-t interval"),
        Line2D([], [], color="#65768a", marker="o", markerfacecolor="white",
               linestyle="none", label="Incomplete condition: finished seeds only"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05,.04),
               frameon=False, fontsize=9, ncol=2)
    suffix = " — before the bound update" if source_commit.startswith("648af9b") else ""
    fig.text(.058, .01, f"Source {source_commit[:7]}{suffix}  |  "
             f"{len(runs)}/{len(specs)} runs complete  |  {stamp}.  "
             "Unfinished runs are not accuracy results.", fontsize=9, color="#596575")

    output.mkdir(parents=True, exist_ok=False)
    prefix = output / "calibrated-noise-progress"
    for extension in ("png", "pdf"):
        fig.savefig(prefix.with_suffix("."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    raw = []
    for run in runs:
        row = asdict(run)
        row.pop("sites")
        row["calibration_sha256"] = table_hash
        raw.append(row)
    write_csv(output / "raw_runs.csv", raw)
    write_csv(output / "summary.csv", summary)
    snapshot = {
        "generated_at_utc": stamp, "source_commit": source_commit,
        "calibration_sha256": table_hash, "manifest_sha256": sha256_file(root / "manifests/grid.tsv"),
        "validated_runs": len(runs), "total_runs": len(specs),
        "time_noise_std_frac": fraction, "complete_stochastic_cells": len(complete),
        "log_sha256": log_hashes, "paper_promotion_allowed": False,
        "raw_csv_sha256": sha256_file(output / "raw_runs.csv"),
        "summary_csv_sha256": sha256_file(output / "summary.csv"),
    }
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=1e-5)
    args = parser.parse_args()
    print(json.dumps(plot_snapshot(args.experiment_root, args.output_directory, args.fraction),
                     sort_keys=True))


if __name__ == "__main__":
    main()
