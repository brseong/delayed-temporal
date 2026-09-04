"""Dataset-independent verification for the ViT theta-noise appendix summary."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis.summarize_theta_noise_scan import summarize_theta_noise


def verify_theta_noise_summary() -> None:
    """Verify three-theta identity checks, combined CSV output, and rendering."""
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        inputs: list[str] = []
        fields = (
            "axis",
            "magnitude",
            "replicas",
            "accuracy_mean",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "model_id",
            "dataset_id",
            "dataset_split",
            "evaluation_samples",
            "theta",
            "precision",
        )
        for theta, scale in ((40, 50.0), (400, 5.0), (2000, 1.0)):
            summary = root / f"theta_{theta}.csv"
            rows = [
                {
                    "axis": "baseline",
                    "magnitude": 0,
                    "replicas": 1,
                    "accuracy_mean": 0.85,
                    "accuracy_ci95_low": "",
                    "accuracy_ci95_high": "",
                    "model_id": "/models/vit-base",
                    "dataset_id": "imagenet-1k",
                    "dataset_split": "validation",
                    "evaluation_samples": 5000,
                    "theta": theta,
                    "precision": "float64",
                },
                *[
                    {
                        "axis": "gaussian",
                        "magnitude": magnitude * scale,
                        "replicas": 3,
                        "accuracy_mean": accuracy,
                        "accuracy_ci95_low": accuracy - 0.01,
                        "accuracy_ci95_high": accuracy + 0.01,
                        "model_id": "/models/vit-base",
                        "dataset_id": "imagenet-1k",
                        "dataset_split": "validation",
                        "evaluation_samples": 5000,
                        "theta": theta,
                        "precision": "float64",
                    }
                    for magnitude, accuracy in ((1.0e-10, 0.84), (4.0e-10, 0.40))
                ],
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            inputs.append(f"{theta}={summary}")

        output_csv = root / "combined.csv"
        figure_prefix = root / "theta_noise"
        combined = summarize_theta_noise(
            inputs,
            output_csv=output_csv,
            figure_prefix=figure_prefix,
        )
        assert len(combined) == 6
        assert {float(row["theta"]) for row in combined} == {40.0, 400.0, 2000.0}
        assert output_csv.is_file()
        assert figure_prefix.with_suffix(".png").is_file()
        assert figure_prefix.with_suffix(".pdf").is_file()


if __name__ == "__main__":
    verify_theta_noise_summary()
    print("Theta-noise summary verification passed.")
