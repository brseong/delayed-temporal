"""Dataset-independent verification for ViT noise-scan aggregation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Make direct execution resolve the repository's namespace packages without an
# editable install or caller-provided PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis.summarize_noise_scan import (
    parse_run_log,
    plot_results,
    read_manifest,
    summarize_noise_scan,
)


def verify_noise_scan_summary() -> None:
    """Verify manifest validation, pooled counts, Student-t CI, and rendering.

    A synthetic seven-run scan avoids datasets and GPUs while exercising the same
    log syntax emitted by the ViT evaluator. The fixture contains one baseline,
    three Gaussian seeds, and three static-mismatch seeds.

    Raises:
        AssertionError: If parsing, aggregation, output creation, or incomplete-log
            rejection violates the maintained sweep contract.
    """
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        log_dir = root / "logs"
        log_dir.mkdir()
        manifest = log_dir / "expected_runs.tsv"

        # Write the complete expected design first. The Gaussian rows intentionally
        # share one magnitude and vary only the dedicated timing-noise seed.
        manifest_rows = [
            {
                "axis": "baseline",
                "magnitude": "0",
                "seed": "",
                "experiment_name": "test-baseline",
                "log_file": "noise_off_baseline.log",
            },
            *[
                {
                    "axis": "gaussian",
                    "magnitude": "1e-6",
                    "seed": str(seed),
                    "experiment_name": f"test-gaussian-{seed}",
                    "log_file": f"gaussian_{seed}.log",
                }
                for seed in (0, 1, 2)
            ],
            *[
                {
                    "axis": "mismatch",
                    "magnitude": "1e-5",
                    "seed": str(seed),
                    "experiment_name": f"test-mismatch-{seed}",
                    "log_file": f"mismatch_{seed}.log",
                }
                for seed in (0, 1, 2)
            ],
        ]
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "axis", "magnitude", "seed", "experiment_name", "log_file"
                ),
                dialect="excel-tab",
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        metadata = (
            "Evaluation metadata — model: /models/vit-base, dataset: imagenet-1k, "
            "split: validation, samples: 5000, theta: 2000.0, precision: float32\n"
        )
        (log_dir / "noise_off_baseline.log").write_text(
            metadata + "wandb: 🚀 View run baseline at: "
            "https://wandb.ai/CIDA/project/runs/baseline\n"
            "Accuracy: 0.82\n",
            encoding="utf-8",
        )
        for seed, accuracy in zip((0, 1, 2), (0.69, 0.70, 0.71), strict=True):
            (log_dir / f"mismatch_{seed}.log").write_text(
                metadata
                + f"Static threshold mismatch — enabled: True, theta_std: 1e-05, seed: {seed}\n"
                "wandb: 🚀 View run mismatch at: "
                f"https://wandb.ai/CIDA/project/runs/mismatch-{seed}\n"
                f"Accuracy: {accuracy}\n",
                encoding="utf-8",
            )

        # Each Gaussian replica contributes two sites so pooling is tested across
        # both operator sites and seeds rather than merely copying emitted rates.
        for seed, accuracy in zip((0, 1, 2), (0.79, 0.80, 0.81), strict=True):
            (log_dir / f"gaussian_{seed}.log").write_text(
                metadata + "Gaussian time noise — enabled: True, std_frac: 1e-06, "
                f"identity_window: 4000.0, std_abs: 0.004, mean_abs: 0.0, seed: {seed}, "
                "identity_deadline_ulp: 0.00048828125, std_to_identity_ulp: 8.192\n"
                "Gaussian[linear.data] events=60, misses=1 (rate=0.0166667), "
                "deadline_events=3 (rate=0.05), deadline_ulp_min=0.000244140625, "
                "deadline_ulp_max=0.00048828125, std_to_ulp_min=8.192, "
                "std_to_ulp_max=16.384, "
                "outputs=30, underflows=1 (rate=0.0333333), "
                "overflows=2 (rate=0.0666667)\n"
                "Gaussian[linear.reference] events=40, misses=0 (rate=0), "
                "deadline_events=0 (rate=0), deadline_ulp_min=0.00048828125, "
                "deadline_ulp_max=0.00048828125, std_to_ulp_min=8.192, "
                "std_to_ulp_max=8.192, "
                "outputs=20, underflows=0 (rate=0), overflows=0 (rate=0)\n"
                f"wandb: 🚀 View run gaussian at: "
                f"https://wandb.ai/CIDA/project/runs/gaussian-{seed}\n"
                f"Accuracy: {accuracy}\n",
                encoding="utf-8",
            )

        raw_csv = root / "raw.csv"
        summary_csv = root / "summary.csv"
        figure_prefix = root / "noise_robustness"
        raw_runs, summary = summarize_noise_scan(
            log_dir=log_dir,
            manifest=manifest,
            raw_csv=raw_csv,
            summary_csv=summary_csv,
            figure_prefix=figure_prefix,
            model_label="Synthetic ViT",
            archive_existing=False,
        )

        # The three accuracies have mean 0.8 and sample standard deviation 0.01.
        # Student-t with df=2 supplies the exact expected 95% half-width.
        assert len(raw_runs) == 7
        gaussian = next(row for row in summary if row["axis"] == "gaussian")
        assert gaussian["replicas"] == 3
        assert math.isclose(float(gaussian["accuracy_mean"]), 0.8, abs_tol=1e-15)
        assert math.isclose(float(gaussian["accuracy_std"]), 0.01, abs_tol=1e-15)
        expected_half_width = 4.302652729696142 * 0.01 / math.sqrt(3.0)
        assert math.isclose(
            float(gaussian["accuracy_ci95_low"]),
            0.8 - expected_half_width,
            rel_tol=1e-12,
        )
        mismatch = next(row for row in summary if row["axis"] == "mismatch")
        assert mismatch["replicas"] == 3
        assert math.isclose(float(mismatch["accuracy_mean"]), 0.70)
        assert math.isclose(float(mismatch["accuracy_std"]), 0.01)
        assert math.isclose(
            float(gaussian["accuracy_ci95_high"]),
            0.8 + expected_half_width,
            rel_tol=1e-12,
        )

        # Counts pool to 300 events and 150 outputs across three replicas. Rates
        # must use those totals, and every requested artifact must be materialized.
        assert gaussian["events"] == 300
        assert gaussian["misses"] == 3
        assert math.isclose(float(gaussian["miss_rate"]), 0.01)
        assert gaussian["deadline_events"] == 9
        assert math.isclose(float(gaussian["deadline_event_rate"]), 0.03)
        assert math.isclose(float(gaussian["deadline_ulp_min"]), 0.000244140625)
        assert math.isclose(float(gaussian["deadline_ulp_max"]), 0.00048828125)
        assert gaussian["outputs"] == 150
        assert gaussian["output_underflows"] == 3
        assert gaussian["output_overflows"] == 6
        assert raw_csv.is_file()
        assert summary_csv.is_file()
        assert figure_prefix.with_suffix(".png").is_file()
        assert figure_prefix.with_suffix(".pdf").is_file()

        # A lower-scale timing refinement deliberately omits static mismatch.
        # Rendering that subset must produce a valid single-panel figure instead
        # of requiring an unrelated axis to be rerun.
        gaussian_only_prefix = root / "gaussian_only"
        plot_results(
            [row for row in summary if row["axis"] != "mismatch"],
            figure_prefix=gaussian_only_prefix,
            model_label="Synthetic ViT",
            archive_existing=False,
        )
        assert gaussian_only_prefix.with_suffix(".png").is_file()
        assert gaussian_only_prefix.with_suffix(".pdf").is_file()

        # Remove one required physical-statistics line set and prove that parsing
        # fails instead of publishing a Gaussian accuracy without mechanism data.
        invalid_log = log_dir / "gaussian_0.log"
        invalid_log.write_text(
            metadata + "Gaussian time noise — enabled: True, std_frac: 1e-06, "
            "identity_window: 4000.0, std_abs: 0.004, mean_abs: 0.0, seed: 0, "
            "identity_deadline_ulp: 0.00048828125, std_to_identity_ulp: 8.192\n"
            "wandb: 🚀 View run gaussian at: "
            "https://wandb.ai/CIDA/project/runs/gaussian-0\n"
            "Accuracy: 0.79\n",
            encoding="utf-8",
        )
        gaussian_expected = next(
            run for run in read_manifest(manifest) if run.axis == "gaussian"
        )
        try:
            parse_run_log(gaussian_expected, log_dir)
        except ValueError as error:
            assert "physical statistics" in str(error)
        else:
            raise AssertionError("accepted Gaussian log without physical statistics")


if __name__ == "__main__":
    verify_noise_scan_summary()
    print("Noise-scan summary verification passed.")
