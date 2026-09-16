"""Validate and plot completed calibrated theta, timing-noise, and margin sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime import identity

SCIENCE_KINDS = {"theta_train", "theta_validation", "theta_replay", "dense", "noise"}
COUNTS = ("events", "misses", "deadline_events", "outputs", "underflows", "overflows")
T95_THREE_SEEDS = 4.302652729911275


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0.0)


def _condition(run: dict[str, Any]) -> tuple:
    if run["kind"] == "noise":
        return ("noise", float(run["theta"]), float(run["time_noise_std_frac"]),
                float(run["deadline_margin_std"]))
    return (run["kind"], float(run["theta"]))


def _rates(counts: dict[str, Any], *, measured: bool) -> dict[str, float | None]:
    return {
        "miss_rate": counts["misses"] / counts["events"]
        if measured and counts["events"] else None,
        "rail_saturation_rate": (counts["underflows"] + counts["overflows"]) / counts["outputs"]
        if measured and counts["outputs"] else None,
    }


def aggregate_results(runs: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Pool raw counts; only complete three-seed cells receive confidence intervals."""
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        if run["kind"] in {"noise", "theta_validation", "dense"}:
            groups.setdefault(_condition(run), []).append(run)
    summary, sites = [], []
    for _, replicas in sorted(groups.items()):
        first = replicas[0]
        noisy = first["kind"] == "noise"
        seeds = sorted(int(run["seed"]) for run in replicas) if noisy else []
        if noisy and (len(set(seeds)) != len(seeds) or not set(seeds).issubset({0, 1, 2})):
            raise ValueError("Duplicate or invalid replica seed")
        if not noisy and len(replicas) != 1:
            raise ValueError("Duplicate deterministic condition")
        values = [float(run["accuracy"]) for run in replicas]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Accuracy must be finite and lie between zero and one")
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else None
        width = T95_THREE_SEEDS * std / math.sqrt(3) if seeds == [0, 1, 2] else None
        row = {name: first.get(name) for name in (
            "kind", "backend", "theta", "theta_index", "time_noise_std_frac",
            "time_noise_std_abs", "deadline_margin_std", "deadline_margin_abs",
            "split", "samples", "source_commit", "checkpoint_sha256", "calibration_sha256",
        )}
        row.update(replicas=len(replicas), seeds=" ".join(map(str, seeds)),
                   accuracy_mean=mean, accuracy_std=std,
                   accuracy_ci95_low=None if width is None else mean - width,
                   accuracy_ci95_high=None if width is None else mean + width,
                   accuracy_ci95_half_width=width,
                   provisional=noisy and seeds != [0, 1, 2])
        row.update({name: sum(int(run.get(name, 0)) for run in replicas) for name in COUNTS})
        row.update(_rates(row, measured=noisy))
        summary.append(row)
        by_site: dict[str, dict] = {}
        for run in replicas:
            if not noisy:
                continue
            seen_sites = set()
            for site in run["sites"]:
                name = site["site"]
                if name in seen_sites:
                    raise ValueError("Duplicate site in one completed replica")
                seen_sites.add(name)
                pooled = by_site.setdefault(name, {"site": name, **{key: 0 for key in COUNTS}})
                for key in COUNTS:
                    pooled[key] += int(site.get(key, 0))
        for name, counts in sorted(by_site.items()):
            sites.append({**{key: row[key] for key in (
                "theta", "time_noise_std_frac", "deadline_margin_std", "replicas", "seeds",
                "source_commit", "calibration_sha256",
            )}, **counts, **_rates(counts, measured=True)})
    return summary, sites


def _write_csv(path: Path, rows: list[dict], default_fields: tuple[str, ...]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(default_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _accuracy_footer(summary: list[dict], progress: dict) -> str:
    """Describe the seed means and intervals present in this figure."""
    noise = [row for row in summary if row["kind"] == "noise"]
    seed_sets = {tuple(int(seed) for seed in row["seeds"].split()) for row in noise}
    if not noise:
        estimate = "no completed noise evaluations"
    elif len(seed_sets) == 1:
        seeds = next(iter(seed_sets))
        estimate = (f"seed {seeds[0]} result" if len(seeds) == 1
                    else "mean of seeds " + ", ".join(map(str, seeds)))
    else:
        estimate = "means of available seeds"
    intervals = sum(row["accuracy_ci95_half_width"] is not None for row in noise)
    if not intervals:
        uncertainty = "confidence intervals not shown"
    elif intervals == len(noise):
        uncertainty = "95% Student-t confidence intervals"
    else:
        uncertainty = "95% Student-t confidence intervals only where all 3 seeds are complete"
    return (f"{progress['completed_noise_runs']}/{progress.get('expected_noise_runs', 51)} "
            f"noise evaluations; {estimate}; {uncertainty}")


def _plot(output: Path, summary: list[dict], progress: dict, selected_theta: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    theta_rows = sorted((r for r in summary if r["kind"] == "theta_validation"),
                        key=lambda r: r["theta"])
    noise = [r for r in summary if r["kind"] == "noise"]
    rt_rows = sorted((r for r in noise if _same(r["deadline_margin_std"], 4)),
                     key=lambda r: r["time_noise_std_frac"])
    ratio_rows = sorted((r for r in noise if _same(r["time_noise_std_frac"], 1e-5)),
                        key=lambda r: r["deadline_margin_std"])
    dense = next((r["accuracy_mean"] for r in summary if r["kind"] == "dense"), None)
    clean = next((r["accuracy_mean"] for r in theta_rows
                  if selected_theta is not None and _same(r["theta"], selected_theta)), None)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for index, (ax, rows, key, label) in enumerate(zip(axes, (theta_rows, rt_rows, ratio_rows),
            ("theta", "time_noise_std_frac", "deadline_margin_std"),
            (r"Threshold $\theta$", r"Timing noise $r_t$", "Deadline margin / noise standard deviation"))):
        if index < 2:
            ax.set_xscale("log")
        else:
            ax.set_xlim(-.15, 6.15)
        ax.set_xlabel(label)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.2)
        if rows:
            ax.plot([r[key] for r in rows], [100*r["accuracy_mean"] for r in rows],
                    color="#2368a2", linewidth=1.5, alpha=.8)
            for row in rows:
                width = row["accuracy_ci95_half_width"]
                ax.errorbar(row[key], 100*row["accuracy_mean"],
                            yerr=None if width is None else 100*width, fmt="o", color="#2368a2",
                            markerfacecolor="white" if row["provisional"] else "#2368a2",
                            capsize=3, markersize=5)
        else:
            ax.text(.5, .5, "No completed evaluations", transform=ax.transAxes, ha="center")
        if dense is not None:
            ax.axhline(100*dense, color="#b46825", linestyle=":", label="Dense reference")
        if index and clean is not None:
            ax.axhline(100*clean, color="#36805c", linestyle="--", label="Clean spiking")
        if index == 0 and selected_theta is not None:
            ax.axvline(selected_theta, color="#36805c", linestyle="--", alpha=.6)
        if dense is not None or (index and clean is not None):
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    title = "ViT-B/16 calibrated sweeps — validation 5k"
    suffix = _accuracy_footer(summary, progress)
    fig.suptitle(title, fontsize=14)
    fig.text(.5, .025, suffix, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .93))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"latest_accuracy.{extension}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for index, (ax, rows, key, label) in enumerate(zip(axes, (rt_rows, ratio_rows),
            ("time_noise_std_frac", "deadline_margin_std"),
            (r"Timing noise $r_t$", "Deadline margin / noise standard deviation"))):
        if not index:
            ax.set_xscale("log")
        else:
            ax.set_xlim(-.15, 6.15)
        positive = [100*r[field] for r in rows for field in ("miss_rate", "rail_saturation_rate")
                    if r[field] is not None and r[field] > 0]
        linear_limit = min(positive) / 10 if positive else 1e-8
        ax.set_yscale("symlog", linthresh=linear_limit, linscale=.8)
        for field, legend, color, marker in (
            ("miss_rate", "Deadline-miss rate", "#d07525", "o"),
            ("rail_saturation_rate", "Pre-clamp rail-saturation rate", "#8560a3", "^"),
        ):
            measured = [r for r in rows if r[field] is not None]
            ax.plot([r[key] for r in measured], [100*r[field] for r in measured],
                    color=color, marker=marker, label=legend)
        ax.set_xlabel(label)
        ax.set_ylabel("Pooled rate (%)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Completed noise evaluations: pooled physical counts", fontsize=13)
    fig.text(.5, .025, "Zero rates are shown at zero on a symmetric logarithmic scale. "
             "Clean physical rates are not inferred.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"latest_physical_rates.{extension}", dpi=170)
    plt.close(fig)


def summarize(root: Path, *, snapshot_seed: int | None = None,
              require_complete: bool = False) -> dict[str, Any]:
    """Validate available completions and produce live or immutable seed-barrier outputs."""
    from scripts.experiments.calibrated_three_sweeps import (
        validate_result, theta_grid, rt_grid, RATIOS, select_theta, confirm_selection,
    )

    root = Path(root).resolve()
    if snapshot_seed is not None and snapshot_seed not in (0, 1, 2):
        raise ValueError("Snapshot seed must be zero, one, or two")
    experiment = _read(root / "experiment.json")
    tasks: dict[str, dict] = {}
    for path in sorted((root / "tasks").glob("*.json")):
        task = _read(path)
        if task["run_id"] in tasks or path.stem != task["run_id"]:
            raise ValueError("Duplicate or incorrectly named task")
        tasks[task["run_id"]] = task
    runs, completed, hashes, conditions = [], [], {}, set()
    for path in sorted((root / "results").glob("*.json")):
        result = _read(path)
        run_id = result.get("run_id")
        if run_id not in tasks or path.stem != run_id:
            raise ValueError("Result does not have exactly one matching task")
        task = tasks[run_id]
        validate_result(task, result, experiment, root)
        if snapshot_seed is not None and result["kind"] in {"noise", "smoke_noise"}:
            if int(result["seed"]) > snapshot_seed:
                continue
        if result["kind"] in SCIENCE_KINDS:
            condition = (*_condition(result), result.get("seed"))
            if condition in conditions:
                raise ValueError("Duplicate completed scientific condition")
            conditions.add(condition)
            if not _same(result["accuracy"], result["correct"] / result["samples"]):
                raise ValueError("Accuracy does not match correct and total counts")
            runs.append(result)
        completed.append(result)
        hashes[run_id] = identity.sha256_file(path)
    noisy = [run for run in runs if run["kind"] == "noise"]
    expected_cells = {(float(rt), 4.0) for rt in rt_grid()} | {(1e-5, float(k)) for k in RATIOS}
    expected_thetas = set(map(float, theta_grid()))
    if len(expected_cells) != 17 or len(expected_thetas) != 9:
        raise ValueError("Expected exactly nine theta points and seventeen unique noise cells")
    by_seed = {seed: set() for seed in (0, 1, 2)}
    noise_thetas, tables = set(), {}
    for run in runs:
        if run["kind"] != "dense":
            theta = float(run["theta"])
            table = run["calibration_sha256"]
            if theta in tables and tables[theta] != table:
                raise ValueError("Mixed calibration identities for one theta")
            tables[theta] = table
        if run["kind"] == "noise":
            cell = (float(run["time_noise_std_frac"]), float(run["deadline_margin_std"]))
            if cell not in expected_cells:
                raise ValueError("Noise condition is outside the declared sweep")
            by_seed[int(run["seed"])].add(cell)
            noise_thetas.add(float(run["theta"]))
    if len(noise_thetas) > 1:
        raise ValueError("Noise sweeps must share one selected theta")
    for seed in (1, 2):
        if by_seed[seed] and by_seed[seed-1] != expected_cells:
            raise ValueError("Later seed started before the preceding global seed barrier")
    selection_path = root / "selection.json"
    selection = _read(selection_path) if selection_path.exists() else {}
    selected = selection.get("theta")
    if noisy and (selected is None or noise_thetas != {float(selected)}):
        raise ValueError("Noise theta does not match the recorded selection")
    if noisy:
        if selection.get("status") != "confirmed":
            raise ValueError("Noise requires confirmed theta evidence")
        training = [run for run in runs if run["kind"] == "theta_train"]
        validation = [run for run in runs if run["kind"] == "theta_validation"]
        replay = [run for run in runs if run["kind"] == "theta_replay"]
        if len(replay) != 1:
            raise ValueError("Exactly one selected theta replay is required")
        verified = confirm_selection(select_theta(training), validation, replay[0], training)
        if any(selection.get(key) != value for key, value in verified.items()):
            raise ValueError("Confirmed theta evidence does not match current completed evaluations")
    seed_complete = [seed for seed in (0, 1, 2) if by_seed[seed] == expected_cells]
    required_seed = 2 if require_complete else snapshot_seed
    if required_seed is not None:
        if not all(seed in seed_complete for seed in range(required_seed + 1)):
            raise ValueError("A global seed barrier is incomplete")
        for kind in ("theta_train", "theta_validation", "collect"):
            observed = {float(run["theta"]) for run in completed if run["kind"] == kind}
            if observed != expected_thetas:
                raise ValueError(f"Incomplete {kind} results")
        for kind in ("theta_replay", "dense"):
            if sum(run["kind"] == kind for run in completed) != 1:
                raise ValueError(f"Exactly one completed {kind} is required")
    summary, sites = aggregate_results(runs)
    base_complete = all(
        {float(run["theta"]) for run in completed if run["kind"] == kind} == expected_thetas
        for kind in ("theta_train", "theta_validation", "collect")
    ) and all(sum(run["kind"] == kind for run in completed) == 1
              for kind in ("theta_replay", "dense"))
    progress = {
        "format_version": 1, "source_commit": experiment["source_commit"],
        "experiment_sha256": identity.sha256_file(root / "experiment.json"),
        "selection_sha256": identity.sha256_file(selection_path) if selection else None,
        "selected_theta": selected, "snapshot_seed": snapshot_seed,
        "completed_evaluations": len(runs), "expected_evaluations": 71,
        "completed_noise_runs": len(noisy), "expected_noise_runs": 51,
        "completed_seed_barriers": seed_complete,
        "noise_runs_by_seed": {str(seed): len(by_seed[seed]) for seed in (0, 1, 2)},
        "completed_collections": sum(run["kind"] == "collect" for run in completed),
        "result_sha256": hashes, "complete": 2 in seed_complete and base_complete,
        "paper_promotion_allowed": False,
    }
    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root if snapshot_seed is None else output_root / f"seed-{snapshot_seed}"
    if snapshot_seed is not None and destination.exists():
        previous = _read(destination / "progress.json")
        if {key: value for key, value in previous.items() if key != "artifact_sha256"} != progress:
            raise ValueError("Refusing to replace an immutable snapshot with different evidence")
        for name, digest in previous.get("artifact_sha256", {}).items():
            if Path(name).name != name or identity.sha256_file(destination / name) != digest:
                raise ValueError("Saved snapshot output changed")
        return previous
    with tempfile.TemporaryDirectory(prefix=".summary-", dir=output_root) as temporary:
        staging = Path(temporary)
        raw = [{key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in run.items() if key != "sites"} for run in runs]
        _write_csv(staging / "raw_runs.csv", raw, ("run_id", "kind", "accuracy"))
        _write_csv(staging / "theta_selection.csv", [r for r in raw if r["kind"] == "theta_train"],
                   ("run_id", "theta", "correct", "samples", "accuracy"))
        _write_csv(staging / "summary.csv", summary, ("kind", "theta", "accuracy_mean", "replicas"))
        _write_csv(staging / "site_summary.csv", sites, ("site", "events", "misses", "miss_rate"))
        _plot(staging, summary, progress, selected)
        progress["artifact_sha256"] = {
            path.name: identity.sha256_file(path) for path in sorted(staging.iterdir())
        }
        (staging / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        if snapshot_seed is not None:
            os.replace(staging, destination)
        else:
            for path in sorted(staging.iterdir(), key=lambda p: p.name == "progress.json"):
                os.replace(path, destination / path.name)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--snapshot-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.experiment_root, snapshot_seed=args.snapshot_seed,
                               require_complete=args.require_complete), sort_keys=True))


if __name__ == "__main__":
    main()
