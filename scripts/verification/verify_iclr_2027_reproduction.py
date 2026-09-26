#!/usr/bin/env python3
"""Verify the integrated ICLR 2027 evidence reproduction driver."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import reproduce_iclr_2027 as reproduction  # noqa: E402


def write_table(path: Path) -> None:
    """Write a compact fixture matching the seven publication rows."""

    fields = (
        "model",
        "family",
        "samples",
        "ann_metric",
        "snn_metric",
        "snn_minus_ann",
        "metric",
        "calibration_sha256",
        "source_commit",
    )
    values = (
        ("cifar10_vit_small", 0.9848, 0.9847),
        ("imagenet_vit_small", 0.8228, 0.8228),
        ("imagenet_vit_base", 0.8606, 0.86),
        ("imagenet_vit_large", 0.8638, 0.8638),
        ("roberta", 0.9449541284, 0.9438073394),
        ("roberta_large", 0.9644495413, 0.9644495413),
        ("gpt2", 21.9841797962, 21.9843868967),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, ann, snn in values:
            writer.writerow(
                {
                    "model": model,
                    "family": (
                        "text"
                        if model in {"roberta", "roberta_large", "gpt2"}
                        else "vit"
                    ),
                    "samples": 1,
                    "ann_metric": ann,
                    "snn_metric": snn,
                    "snn_minus_ann": snn - ann,
                    "metric": (
                        "token_weighted_perplexity"
                        if model == "gpt2"
                        else "accuracy"
                    ),
                    "calibration_sha256": "0" * 64,
                    "source_commit": "0" * 40,
                }
            )


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation#ICLR 2027 Evidence Reproduction]]
def main() -> None:
    assert len(reproduction.ITEMS) == 7
    assert len({item.key for item in reproduction.ITEMS}) == len(reproduction.ITEMS)
    paths = reproduction.artifact_paths(Path("/tmp/evidence"))
    assert all(str(path).startswith("/tmp/evidence/") for path in paths.values())
    assert reproduction.format_delta(-0.0001, scale=100.0) == "-0.01"
    assert reproduction.format_delta(0.0002, scale=1.0) == "0.00"

    with tempfile.TemporaryDirectory(prefix="iclr-reproduction-") as directory:
        root = Path(directory)
        table = root / "table.csv"
        tex = root / "experiment.tex"
        write_table(table)
        fragments = reproduction.expected_table_fragments(table)
        assert len(fragments) == 7
        tex.write_text("\n".join(fragments), encoding="utf-8")
        reproduction.verify_table_binding(table, tex)
        tex.write_text("\n".join(fragments[:-1]), encoding="utf-8")
        try:
            reproduction.verify_table_binding(table, tex)
        except ValueError:
            pass
        else:
            raise AssertionError("a missing GPT-2 publication value was accepted")
    print("ICLR 2027 reproduction driver verification passed")


if __name__ == "__main__":
    main()
