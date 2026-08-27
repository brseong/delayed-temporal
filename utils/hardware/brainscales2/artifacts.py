"""Artifact serialization for reproducible BrainScaleS-2 pooling runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import csv
import json
import math

import torch

from utils.transforms.types import Potential

from .config import BrainScaleS2PoolConfig, CADCDiagnosticResult, PoolRunResult


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    return value


def condition_key(result: PoolRunResult) -> str:
    return f"M{result.pool_size}_{result.placement}_{result.routing}"


def write_cadc_diagnostic_artifacts(
    output_dir: Path,
    *,
    config: BrainScaleS2PoolConfig,
    result: CADCDiagnosticResult,
    analysis: dict[str, object],
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    """Persist paired CADC traces and the next-stage recommendation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = {
        "key": "cadc_single_psp",
        "pool_size": len(result.physical_coordinates),
        "placement": "same-quadrant",
        "routing": "broadcast",
        "physical_coordinates": list(result.physical_coordinates),
        "metadata": result.metadata,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase": "diagnose-cadc",
        "config": config.to_manifest_dict(),
        "conditions": [condition],
        "diagnostic": analysis.get("aggregate"),
        "viable": analysis.get("viable"),
        "reason": analysis.get("reason"),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = list(analysis.get("per_neuron", []))
    _write_rows(output_dir / "summary.csv", rows)
    torch.save(
        {
            "baseline_cadc": result.baseline_cadc,
            "stimulated_cadc": result.stimulated_cadc,
            "baseline_spikes": result.baseline_spikes,
            "stimulated_spikes": result.stimulated_spikes,
            "time_s": result.time_s,
            "stimulus_time_s": result.stimulus_time_s,
            "physical_coordinates": result.physical_coordinates,
        },
        output_dir / "cadc_traces.pt",
    )
    recommendation = {
        "viable": analysis.get("viable"),
        "reason": analysis.get("reason"),
        "selected": analysis.get("selected"),
        "aggregate": analysis.get("aggregate"),
        "config": config.to_manifest_dict(),
    }
    (output_dir / "recommended_operating_point.json").write_text(
        json.dumps(_json_value(recommendation), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    time_us = result.time_s * 1.0e6
    baseline = result.baseline_cadc.mean(dim=(0, 2))
    stimulated = result.stimulated_cadc.mean(dim=(0, 2))
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(time_us, baseline, label="no input")
    axis.plot(time_us, stimulated, label="single input")
    axis.axvline(result.stimulus_time_s * 1.0e6, color="black", linestyle="--")
    axis.set_xlabel("hardware time [µs]")
    axis.set_ylabel("mean membrane CADC [a.u.]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cadc_trace.png", dpi=180)
    plt.close(figure)


def write_experiment_artifacts(
    output_dir: Path,
    *,
    config: BrainScaleS2PoolConfig,
    potential: Potential,
    results: list[PoolRunResult],
    summaries: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    """Write the stable manifest, event, tensor, summary, and fit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "config": config.to_manifest_dict(),
        "input": {
            "shape": list(potential.value.shape),
            "potential_min": float(potential.domain.min),
            "potential_max": float(potential.domain.max),
        },
        "conditions": [
            {
                "key": condition_key(result),
                "pool_size": result.pool_size,
                "placement": result.placement,
                "routing": result.routing,
                "physical_coordinates": list(result.physical_coordinates),
                "metadata": result.metadata,
            }
            for result in results
        ],
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    event_fields = (
        "condition",
        "trial",
        "sample",
        "neuron",
        "physical_coordinate",
        "ideal_input_s",
        "nominal_input_s",
        "first_spike_s",
        "fired",
        "spike_count",
    )
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for result in results:
            key = condition_key(result)
            trials, samples, neurons = result.first_spike_s.shape
            for trial in range(trials):
                for sample in range(samples):
                    for neuron in range(neurons):
                        fired = bool(result.fired[trial, sample, neuron])
                        writer.writerow(
                            {
                                "condition": key,
                                "trial": trial,
                                "sample": sample,
                                "neuron": neuron,
                                "physical_coordinate": result.physical_coordinates[neuron],
                                "ideal_input_s": float(result.ideal_input_s[sample]),
                                "nominal_input_s": float(result.nominal_input_s[sample]),
                                "first_spike_s": (
                                    float(result.first_spike_s[trial, sample, neuron])
                                    if fired
                                    else ""
                                ),
                                "fired": int(fired),
                                "spike_count": int(
                                    result.spike_count[trial, sample, neuron]
                                ),
                            }
                        )

    torch.save(
        {
            condition_key(result): {
                "first_spike_s": result.first_spike_s,
                "fired": result.fired,
                "spike_count": result.spike_count,
                "ideal_input_s": result.ideal_input_s,
                "nominal_input_s": result.nominal_input_s,
                "original_input_shape": result.original_input_shape,
            }
            for result in results
        },
        output_dir / "events.pt",
    )

    _write_rows(output_dir / "summary.csv", summaries)
    _write_rows(output_dir / "variance_fit.csv", fits)
    _plot_variance_fit(output_dir / "variance_fit.png", summaries, fits)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in row.items()
                }
            )


def _plot_variance_fit(
    path: Path,
    summaries: list[dict[str, Any]],
    fits: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[
            (
                str(summary["placement"]),
                str(summary["routing"]),
                str(summary["estimator"]),
            )
        ].append(summary)
    fit_lookup = {
        (str(row["placement"]), str(row["routing"]), str(row["estimator"])): row
        for row in fits
    }
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["pool_size"]))
        inverse_size = [1.0 / int(row["pool_size"]) for row in rows]
        variance = [float(row["variance_s2"]) for row in rows]
        label = "/".join(key)
        axis.scatter(inverse_size, variance, label=label)
        fit = fit_lookup.get(key)
        if fit is not None and math.isfinite(float(fit["a_s2"])):
            x_line = torch.linspace(0.0, max(inverse_size), 100)
            y_line = float(fit["a_s2"]) * x_line + float(fit["c_s2"])
            axis.plot(x_line, y_line)
    axis.set_xlabel("1 / pool size")
    axis.set_ylabel("held-out latency variance [s²]")
    if grouped:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

