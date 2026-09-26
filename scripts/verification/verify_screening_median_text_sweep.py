#!/usr/bin/env python3
"""Verify encoder-specific text timing controls and sweep reduction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.summarize_screening_median_text_sweep import (
    ComparisonFigureStyle,
    PAPER_PANEL_FONT_SIZE,
    PAPER_PANEL_HEIGHT_IN,
    PAPER_PANEL_LEGEND_SIZE,
    PAPER_PANEL_WIDTH_IN,
    aligned_zero_limits,
    relative_inverse_perplexity_change,
    render,
    render_model_comparison,
    summarize,
    write_csv,
)
from scripts.evaluation import error_analysis_gpt2, error_analysis_roberta
from scripts.experiments.run_screening_median_text_condition import evaluator_command
from scripts.experiments.run_vit_local_range_noise_condition import parse_physical_counts
from scripts.experiments.run_screening_median_text_sweep import (
    preparation_artifacts_root,
    preparation_root,
    preparation_runtime_root,
)
from scripts.experiments.screening_median_text_sweep import (
    TEXT_INITIAL_ALPHAS,
    TEXT_MODELS,
    TEXT_SEEDS,
    expected_text_cells,
)


def verify_evaluator_arguments() -> None:
    """Require distinct NP/NL fractions and a shared deadline margin."""

    options = [
        "--gaussian-time-noise",
        "--time-noise-std-frac",
        "0",
        "--linear-time-noise-std-frac",
        "0.001",
        "--log-time-noise-std-frac",
        "0.002",
        "--time-noise-deadline-margin-std",
        "4",
    ]
    with patch("sys.argv", ["error_analysis_roberta.py", *options]):
        roberta = error_analysis_roberta.parse_arguments()
    with patch("sys.argv", ["error_analysis_gpt2.py", *options]):
        gpt2 = error_analysis_gpt2.parse_arguments()
    for parsed in (roberta, gpt2):
        assert parsed.gaussian_time_noise
        assert parsed.time_noise_std_frac == 0.0
        assert parsed.linear_time_noise_std_frac == 0.001
        assert parsed.log_time_noise_std_frac == 0.002
        assert parsed.time_noise_deadline_margin_std == 4.0


def verify_grid_and_command() -> None:
    """Require the complete two-model, seven-alpha, three-seed population."""

    formal = expected_text_cells(phase="formal", alphas=TEXT_INITIAL_ALPHAS)
    smoke = expected_text_cells(phase="smoke", alphas=("1",))
    assert len(formal) == len(TEXT_MODELS) * len(TEXT_INITIAL_ALPHAS) * len(TEXT_SEEDS) == 42
    assert len(smoke) == len(TEXT_MODELS) * len(TEXT_SEEDS) == 6
    assert len({cell.identity for cell in formal}) == len(formal)

    args = SimpleNamespace(
        python_bin="python",
        phase="formal",
        seed=2,
    )
    protocol = {"resources": {"source_root": str(ROOT)}}
    resource = {
        "evaluator_family": "roberta",
        "model_id": "/checkpoint",
        "task": "sst2",
        "cache_dir": "/cache",
        "activation": "gelu",
        "calibration_dataset_path": "/calibration",
        "calibration_dataset_fingerprint": "cal",
        "evaluation_dataset_path": "/evaluation",
        "evaluation_dataset_fingerprint": "eval",
    }
    command = evaluator_command(
        args,
        protocol,
        resource,
        Path("/frozen-calibration.json"),
        linear_fraction=0.001,
        log_fraction=0.002,
        run_id="fixture",
    )
    assert command[command.index("--linear-time-noise-std-frac") + 1] == "0.001"
    assert command[command.index("--log-time-noise-std-frac") + 1] == "0.002"
    assert command[command.index("--time-noise-deadline-margin-std") + 1] == "4.0"


def verify_preparation_layout() -> None:
    """Require campaign isolation without weakening the canonical fixed layout."""

    args = SimpleNamespace(output_root=Path("/artifacts/results/text-campaign"))
    artifacts = preparation_artifacts_root(args)
    assert artifacts == args.output_root / "preparation_artifacts"
    assert preparation_root(args, "roberta_base") == (
        artifacts
        / "logs/conversion_comparison"
        / "conversion_comparison_end_to_end_local_ranges_float64_v1"
        / "text/roberta"
    )
    assert preparation_root(args, "gpt2") == (
        artifacts
        / "logs/conversion_comparison"
        / "gpt2_end_to_end_local_ranges_float64_v1"
        / "text/gpt2"
    )
    assert preparation_runtime_root(args, "gpt2") == (
        artifacts / "runtime/gpt2_end_to_end_local_ranges_float64_v1/text"
    )


def verify_text_physical_count_report() -> None:
    """Require the text report to retain deadline and output count populations."""

    fixture = (
        "Gaussian[linear.data] events=100, misses=2 (rate=0.02), "
        "deadline_events=3 (rate=0.03), deadline_ulp_min=1e-12, "
        "deadline_ulp_max=2e-12, outputs=50, underflows=4 (rate=0.08), "
        "overflows=5 (rate=0.1)"
    )
    counts = parse_physical_counts(fixture)
    assert counts["events"] == 100
    assert counts["misses"] == 2
    assert counts["deadline_events"] == 3
    assert counts["outputs"] == 50
    assert counts["underflows"] == 4
    assert counts["overflows"] == 5
    for evaluator in (error_analysis_roberta, error_analysis_gpt2):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        assert 'f"deadline_events={counts[' in source
        assert 'f"deadline_ulp_min={ulp_min:' in source


def fixture_protocol() -> dict:
    return {
        "resources": {
            "models": {
                "roberta_base": {
                    "converted_clean": {"accuracy": 0.94},
                    "ann_reference": {"accuracy": 0.945},
                },
                "gpt2": {
                    "converted_clean": {
                        "token_weighted_loss": 3.1,
                        "token_weighted_perplexity": 22.0,
                    },
                    "ann_reference": {"token_weighted_perplexity": 21.9},
                },
            }
        }
    }


def fixture_rows() -> list[dict]:
    rows = []
    for model in TEXT_MODELS:
        for alpha in (0.01, 1.0):
            for seed in TEXT_SEEDS:
                row = {
                    "model": model,
                    "alpha": alpha,
                    "seed": seed,
                    "linear_time_noise_std_fraction": 0.01 * alpha,
                    "log_time_noise_std_fraction": 0.02 * alpha,
                    "events": 100,
                    "misses": seed,
                }
                if model == "roberta_base":
                    row["accuracy"] = 0.94 - 0.01 * alpha - 0.001 * seed
                else:
                    row["token_weighted_loss"] = 3.1 + 0.01 * alpha + 0.001 * seed
                    row["token_weighted_perplexity"] = 22.0 + alpha + 0.1 * seed
                rows.append(row)
    return rows


# @lat: [[evaluation#Evaluation and Verification#Screening Median Text Timing Noise Sweep#Verification]]
def main() -> None:
    verify_evaluator_arguments()
    verify_grid_and_command()
    verify_preparation_layout()
    verify_text_physical_count_report()
    summary = summarize(fixture_protocol(), fixture_rows())
    assert len(summary) == 4
    assert {row["metric"] for row in summary} == {"accuracy", "token_weighted_perplexity"}
    incomplete = [
        row
        for row in fixture_rows()
        if not (row["model"] == "gpt2" and row["alpha"] == 1.0 and row["seed"] == 2)
    ]
    try:
        summarize(fixture_protocol(), incomplete)
    except ValueError as error:
        assert "seeds differ" in str(error)
    else:
        raise AssertionError("formal reduction accepted an incomplete three-seed cell")
    draft = summarize(fixture_protocol(), incomplete, allow_incomplete=True)
    assert len(draft) == 3
    assert not any(
        row["model"] == "gpt2" and row["alpha"] == 1.0 for row in draft
    )
    gpt2 = next(
        row for row in summary
        if row["model"] == "gpt2" and row["alpha"] == 1.0
    )
    mean, low, high = relative_inverse_perplexity_change(gpt2)
    clean = gpt2["converted_clean_metric"]
    expected = [
        -100.0 * (1.0 - clean / value) for value in (23.0, 23.1, 23.2)
    ]
    assert mean == sum(expected) / len(expected)
    assert -100.0 < low <= mean <= high
    twice_clean = {
        "metric": "token_weighted_perplexity",
        "relative_inverse_perplexity_change_mean": -50.0,
        "relative_inverse_perplexity_change_ci_low": -50.0,
        "relative_inverse_perplexity_change_ci_high": -50.0,
    }
    assert relative_inverse_perplexity_change(twice_clean) == (
        -50.0,
        -50.0,
        -50.0,
    )
    paper_style = ComparisonFigureStyle()
    assert paper_style.width_in == PAPER_PANEL_WIDTH_IN == 3.35
    assert paper_style.height_in == PAPER_PANEL_HEIGHT_IN == 2.15
    assert paper_style.font_size == PAPER_PANEL_FONT_SIZE == 8.0
    assert paper_style.axis_label_size == 12.5
    assert paper_style.legend_size == PAPER_PANEL_LEGEND_SIZE == 8.0
    assert paper_style.legend_columns == 5
    assert paper_style.legend_above
    assert paper_style.compact_labels
    assert not paper_style.show_title
    left_limits = aligned_zero_limits([-20.0, 0.2])
    right_limits = aligned_zero_limits([-50.0, 1.0])
    assert abs((-left_limits[0]) / (left_limits[1] - left_limits[0]) - 0.88) < 1e-12
    assert abs((-right_limits[0]) / (right_limits[1] - right_limits[0]) - 0.88) < 1e-12
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_csv(root / "summary.csv", summary)
        render(summary, root / "figure")
        vision = []
        for model in ("cct7", "imagenet_vit_small", "imagenet_vit_base"):
            for alpha in (0.01, 1.0):
                vision.append(
                    {
                        "model": model,
                        "alpha": alpha,
                        "accuracy_change_pp_mean": -alpha,
                        "accuracy_change_pp_ci_low": -alpha - 0.1,
                        "accuracy_change_pp_ci_high": -alpha + 0.1,
                    }
                )
        write_csv(root / "aggregate.csv", vision)
        with patch(
            "matplotlib.axes.Axes.axvline",
            side_effect=AssertionError("combined figure must not imply a threshold"),
        ):
            render_model_comparison(summary, root, root / "combined")
            render_model_comparison(
                summary,
                root,
                root / "combined_compact",
                style=ComparisonFigureStyle(
                    width_in=3.2,
                    height_in=2.1,
                    font_size=7.6,
                    legend_size=5.8,
                    legend_columns=2,
                    legend_above=True,
                    gpt2_max_alpha=0.01,
                    show_title=False,
                    compact_labels=True,
                ),
            )
        assert (root / "figure.png").is_file()
        assert (root / "figure.pdf").is_file()
        assert (root / "combined.png").is_file()
        assert (root / "combined.pdf").is_file()
        assert (root / "combined_compact.png").is_file()
        assert (root / "combined_compact.pdf").is_file()
    print("Screening median text timing-noise sweep verification passed.")


if __name__ == "__main__":
    main()
