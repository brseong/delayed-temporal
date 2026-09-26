#!/usr/bin/env python3
"""Check the Llama appendix diagnostic result contract without GPU inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_llama_appendix_noise as sweep


def expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("invalid appendix evidence was accepted")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        calibration = root / "calibration.json"
        calibration.write_text("{}", encoding="utf-8")
        model = root / "model"
        model.mkdir()
        cleans = {}
        for dataset, baseline in (("wikitext2", 10.0), ("imdb", 20.0)):
            directory = root / dataset
            identity = sweep.expected_identity(
                model_id=model, calibration_path=calibration,
                dataset=dataset, implementation_digest="implementation-test",
                device_count=2,
            )
            shared = {
                **identity, "dataset_fingerprint": f"{dataset}-test",
                "dataset_path": f"{dataset}/test",
                "calibration_dataset_fingerprint": "calibration-test",
                "noise_enabled": False,
            }
            hf = {**shared, "backend": "hf", "perplexity": baseline - 0.01}
            converted = {**shared, "backend": "spiking", "perplexity": baseline}
            write_json(directory / "hf_clean.json", hf)
            write_json(directory / "spiking_clean.json", converted)
            cleans[dataset] = sweep.clean_pair(directory, identity)
            for alpha_index, alpha in enumerate(sweep.ALPHAS):
                for seed in (0, 1, 2):
                    perplexity = baseline + 0.1 * (alpha_index + 1) * (seed + 1)
                    row = {
                        **converted, "noise_enabled": True, "alpha": alpha,
                        "seed": seed, "perplexity": perplexity,
                        "hardware_summary_sha256": sweep.HARDWARE_SUMMARY_SHA256,
                        "deadline_margin_std_ratio": sweep.DEADLINE_MARGIN_SIGMA_RATIO,
                        "relative_inverse_perplexity_percent":
                            100.0 * (baseline / perplexity - 1.0),
                    }
                    write_json(sweep.condition_path(directory, alpha, seed), row)

        one = sweep.summarize(output_root=root, datasets=sweep.DATASETS,
                              cleans=cleans, seeds=(0,))
        table = sweep.render_table(one)
        assert "WikiText-2" in table and "IMDb" in table
        assert "10^{-5}" in table and "10^{-4}" in table and "10^{-3}" in table
        assert "\\alpha" not in table and "seed 0 only" in table
        assert all("perplexity_ci_95" not in cell
                   and "relative_inverse_perplexity_ci_95_percent" not in cell
                   for row in one["datasets"] for cell in row["conditions"])
        three = sweep.summarize(output_root=root, datasets=sweep.DATASETS,
                                cleans=cleans, seeds=(0, 1, 2))
        assert all("perplexity_ci_95" in cell
                   and "relative_inverse_perplexity_ci_95_percent" in cell
                   for row in three["datasets"] for cell in row["conditions"])
        three_table = sweep.render_table(three)
        assert r"\pm" in three_table and r"95\% Student-$t$" in three_table
        assert r"\begin{tabular}{@{}lcc@{}}" in three_table
        assert "Condition & WikiText-2 & IMDb" in three_table
        assert "The three noisy rows" in three_table

        noise_path = sweep.condition_path(root / "imdb", "0.0001", 0)
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        noise["calibration_sha256"] = "different"
        write_json(noise_path, noise)
        expect_value_error(lambda: sweep.summarize(
            output_root=root, datasets=sweep.DATASETS, cleans=cleans, seeds=(0,)))
        noise["calibration_sha256"] = cleans["imdb"][1]["calibration_sha256"]
        write_json(noise_path, noise)

        converted_path = root / "imdb" / "spiking_clean.json"
        converted = json.loads(converted_path.read_text(encoding="utf-8"))
        converted["dataset_fingerprint"] = "different"
        write_json(converted_path, converted)
        expect_value_error(lambda: sweep.clean_pair(root / "imdb", sweep.expected_identity(
            model_id=model, calibration_path=calibration,
            dataset="imdb", implementation_digest="implementation-test", device_count=2,
        )))
    print("Llama appendix noise contract: PASS")


if __name__ == "__main__":
    main()
