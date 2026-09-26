#!/usr/bin/env python3
"""Verify clean Llama shard boundaries and corpus-loss reduction without GPUs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from datasets import Dataset
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.error_analysis_llama import evaluate, prepare_batches
from scripts.experiments.run_llama_iclr_campaign import merge_clean_shards


class Tokenizer:
    def __call__(self, texts: list[str], *, padding: str, truncation: bool, max_length: int) -> dict:
        assert padding == "max_length" and truncation
        return {
            "input_ids": [[1, 2] + [0] * (max_length - 2) for _ in texts],
            "attention_mask": [[1, 1] + [0] * (max_length - 2) for _ in texts],
        }


class ToyModel:
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 labels: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        del attention_mask, use_cache
        counts = (labels[:, 1:] != -100).sum(dim=1).to(torch.float64)
        losses = input_ids[:, 0].to(torch.float64)
        loss = (losses * counts).sum() / counts.sum()
        return SimpleNamespace(loss=loss, logits=torch.zeros((len(input_ids), 1)))


def main() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        "attention_mask": torch.ones((3, 4), dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4], [2, 3, 4, -100], [3, 4, -100, -100]]),
    }
    full = evaluate(ToyModel(), [batch], torch.device("cpu"), condition="full", microbatch_size=3)
    split = evaluate(ToyModel(), [batch], torch.device("cpu"), condition="split", microbatch_size=1)
    assert full["valid_token_count"] == split["valid_token_count"] == 6
    assert math.isclose(full["loss"], split["loss"], rel_tol=1e-15)
    assert math.isclose(full["token_weighted_loss"], split["token_weighted_loss"], rel_tol=1e-15)

    dataset = Dataset.from_dict({"text": [str(index) for index in range(11)]})
    populations = []
    for index in range(3):
        loader, population = prepare_batches(
            Tokenizer(), dataset=dataset, source="test",
            expected_size=11, expected_fingerprint=dataset._fingerprint,
            max_examples=11, batch_size=2, max_length=4,
            shard_index=index, shard_count=3,
        )
        assert len(loader) == (population["examples"] + 1) // 2
        populations.append(population)
    assert [(row["shard_start"], row["shard_stop"]) for row in populations] == [
        (0, 4), (4, 8), (8, 11),
    ]
    assert all(row["dataset_fingerprint"] == dataset._fingerprint for row in populations)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        expected = {
            "model_id": "/checkpoint", "calibration_sha256": "calibration",
            "implementation_sha256": "implementation", "calibration_examples": 5_000,
            "evaluation_dataset": "wikitext2", "examples": 11,
            "batch_size": 2, "max_length": 4, "device": "cuda:0,cuda:1",
        }
        for index, population in enumerate(populations):
            directory = root / "shards" / f"shard_{index}"
            directory.mkdir(parents=True)
            for backend, filename in (("hf", "hf_clean.json"), ("spiking", "spiking_clean.json")):
                examples = population["examples"]
                row = {
                    **expected, **population,
                    "dtype": "float64", "backend": backend,
                    "noise_enabled": False,
                    "loss_aggregation": "mean_of_batch_losses",
                    "microbatch_size": 2 if index < 2 else 1,
                    "loss": float(index + 1),
                    "token_weighted_loss": float(index + 1),
                    "batch_count": len(range(population["shard_start"], population["shard_stop"], 2)),
                    "valid_token_count": examples * 10,
                    "example_count": examples,
                    "elapsed_seconds": 1.0,
                }
                if backend == "spiking":
                    row["calibration_clipping"] = [{
                        "module_name": "model.layers.0", "tensor_name": "output",
                        "num_values": examples * 100, "underflows": index,
                        "overflows": 0, "underflow_rate": index / (examples * 100),
                        "overflow_rate": 0.0,
                    }]
                (directory / filename).write_text(json.dumps(row), encoding="utf-8")
        hf, converted = merge_clean_shards(
            root, shard_count=3, gpu_groups=[[0, 1], [2, 3], [4, 5]], expected=expected,
        )
        weighted = (4 * 1 + 4 * 2 + 3 * 3) / 11
        assert hf["examples"] == converted["examples"] == 11
        assert hf["batch_count"] == converted["batch_count"] == 6
        assert hf["valid_token_count"] == converted["valid_token_count"] == 110
        assert math.isclose(hf["perplexity"], math.exp(2.0))
        assert math.isclose(hf["token_weighted_perplexity"], math.exp(weighted))
        assert converted["calibration_clipping"][0]["num_values"] == 1100
        assert converted["calibration_clipping"][0]["underflows"] == 3
        assert converted["evaluation_shards"] == 3
        assert converted["physical_microbatch_sizes"] == [2, 2, 1]

        for filename in ("hf_clean.json", "spiking_clean.json"):
            remote = root / "shards" / "shard_2" / filename
            row = json.loads(remote.read_text(encoding="utf-8"))
            row["device"] = "cuda:0,cuda:1,cuda:2,cuda:3"
            row["dataset_path"] = "/cluster/test"
            remote.write_text(json.dumps(row), encoding="utf-8")
        hybrid_expected = {key: value for key, value in expected.items() if key != "device"}
        hf, converted = merge_clean_shards(
            root, shard_count=3,
            gpu_groups=[["local:0", "local:1"], ["local:2", "local:3"],
                        ["cluster:0", "cluster:1", "cluster:2", "cluster:3"]],
            expected=hybrid_expected,
        )
        assert hf["device"] == converted["device"] == "distributed"
        assert math.isclose(converted["token_weighted_perplexity"], math.exp(weighted))

        corrupted = root / "shards" / "shard_1" / "hf_clean.json"
        row = json.loads(corrupted.read_text(encoding="utf-8"))
        row["shard_start"] += 1
        corrupted.write_text(json.dumps(row), encoding="utf-8")
        try:
            merge_clean_shards(
                root, shard_count=3, gpu_groups=[[0, 1], [2, 3], [4, 5]],
                expected=hybrid_expected,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect shard boundary was accepted")
    print("Llama clean shard contract: PASS")


if __name__ == "__main__":
    main()
