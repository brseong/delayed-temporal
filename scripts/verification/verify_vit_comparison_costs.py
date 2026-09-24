#!/usr/bin/env python3
"""Independent checks for the current ViT SOP accounting contract."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.vit_comparison_costs import (
    COST_MODEL_VERSION,
    ENERGY_PJ_PER_SOP,
    estimate_vit_cost,
)


def configuration(
    *,
    image_size: int = 8,
    patch_size: int = 4,
    hidden: int = 4,
    intermediate: int = 8,
    heads: int = 2,
    layers: int = 2,
    classes: int = 3,
) -> dict[str, int | str]:
    """Return a complete checkpoint-style ViT configuration."""
    return {
        "model_type": "vit",
        "hidden_act": "gelu",
        "image_size": image_size,
        "patch_size": patch_size,
        "num_channels": 3,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "num_labels": classes,
    }


def enumerate_small_circuit(config: dict[str, int | str]) -> dict[str, Counter[str]]:
    """Count every destination independently for a deliberately small graph."""
    image_size = int(config["image_size"])
    patch_size = int(config["patch_size"])
    channels = int(config["num_channels"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    head_count = int(config["num_attention_heads"])
    layer_count = int(config["num_hidden_layers"])
    class_count = int(config["num_labels"])
    patches = list(range((image_size // patch_size) ** 2))
    tokens = list(range(len(patches) + 1))
    features = list(range(hidden))
    mlp_features = list(range(intermediate))
    patch_features = list(range(channels * patch_size**2))
    heads = list(range(head_count))
    head_features = list(range(hidden // head_count))
    classes = list(range(class_count))
    totals: dict[str, Counter[str]] = {}

    def event(stage: str, kind: str) -> None:
        totals.setdefault(stage, Counter())[kind] += 1

    def linear(stage: str, rows: list[int], inputs: list[int], outputs: list[int]) -> None:
        event(stage, "global")
        for _ in rows:
            for _ in inputs:
                event(stage, "global")
                for _ in outputs:
                    event(stage, "data")
            for _ in outputs:
                event(stage, "global")

    def multiply(stage: str, coordinates: list[tuple[int, int]]) -> None:
        event(stage, "global")
        for _ in coordinates:
            event(stage, "global")
            event(stage, "data")
            event(stage, "global")

    def layernorm(prefix: str) -> None:
        coordinates = [(token, feature) for token in tokens for feature in features]
        for _ in ("positive", "negative"):
            multiply(prefix + ".square", coordinates)
        stage = prefix + ".log_and_exponential_difference"
        for _ in tokens:
            event(stage, "global")
            for _ in ("positive", "negative"):
                for _ in features:
                    event(stage, "data")
                    event(stage, "global")
                    event(stage, "data")
                    event(stage, "global")
                    event(stage, "data")
        stage = prefix + ".gamma"
        event(stage, "global")
        for _ in features:
            event(stage, "global")
            for _ in tokens:
                event(stage, "data")
                event(stage, "global")

    linear("patch_embedding", patches, patch_features, features)
    for _ in range(layer_count):
        layernorm("block.layernorm")
        layernorm("block.layernorm")
        for _ in ("query", "key", "value"):
            linear("block.attention.qkv", tokens, features, features)

        stage = "block.attention.score"
        event(stage, "global")
        for _ in heads:
            for _ in tokens:
                for _ in head_features:
                    event(stage, "global")
                    for _ in tokens:
                        event(stage, "data")
            for _ in tokens:
                for _ in tokens:
                    event(stage, "global")

        for _ in heads:
            for _ in tokens:
                event("block.attention.division", "global")
                for _ in tokens:
                    event("block.attention.exponential", "global")
                    event("block.attention.exponential", "data")
                    event("block.attention.division", "global")
                    event("block.attention.division", "global")
                    for _ in ("numerator", "denominator", "internal"):
                        event("block.attention.division", "data")

        stage = "block.attention.value"
        event(stage, "global")
        for _ in tokens:
            for _ in features:
                event(stage, "global")
                for _ in tokens:
                    event(stage, "data")
        for _ in tokens:
            for _ in features:
                event(stage, "global")

        linear("block.attention.output_projection", tokens, features, features)
        linear("block.mlp.input_projection", tokens, features, mlp_features)
        coordinates = [(token, feature) for token in tokens for feature in mlp_features]
        stage = "block.mlp.gelu.cubic"
        event(stage, "global")
        for _ in coordinates:
            for _ in ("positive", "negative"):
                event(stage, "global")
                event(stage, "data")
                event(stage, "global")
                event(stage, "global")
                event(stage, "data")
            event("block.mlp.gelu.exponential", "global")
            event("block.mlp.gelu.exponential", "data")
            for _ in ("numerator", "denominator", "internal"):
                event("block.mlp.gelu.division", "global")
                event("block.mlp.gelu.division", "data")
        multiply("block.mlp.gelu.product", coordinates)
        linear("block.mlp.output_projection", tokens, mlp_features, features)

    layernorm("final_layernorm")
    linear("classification_head", [0], features, classes)
    return totals


class CostTests(unittest.TestCase):
    """Check the estimator without reusing its component polynomials."""

    # @lat: [[comparison-costs#Comparison Costs and Source Audit#Verification]]
    def test_enumerated_oracle(self) -> None:
        config = configuration()
        actual = estimate_vit_cost(config)
        expected = enumerate_small_circuit(config)
        breakdown = {row["component"]: row for row in actual["breakdown"]}
        self.assertEqual(set(breakdown), set(expected))
        for component, counts in expected.items():
            with self.subTest(component=component):
                self.assertEqual(breakdown[component]["data_sop"], counts["data"])
                self.assertEqual(breakdown[component]["global_sop"], counts["global"])
                self.assertEqual(
                    breakdown[component]["total_sop"], counts["data"] + counts["global"]
                )

    def test_classification_head_matches_spiking_linear_mapping(self) -> None:
        hidden, classes = 384, 1000
        result = estimate_vit_cost(configuration(hidden=hidden, classes=classes))
        head = next(row for row in result["breakdown"] if row["component"] == "classification_head")
        self.assertEqual(head["source"], "utils/transformers/models/spiking_ops.py:SpikingLinear")
        self.assertEqual(head["data_sop"], hidden * classes)
        self.assertEqual(head["global_sop"], hidden + classes + 1)

    def test_paper_model_totals(self) -> None:
        cases = {
            "cifar_vit_s": (384, 1536, 6, 12, 10, 4_750_469_300),
            "imagenet_vit_s": (384, 1536, 6, 12, 1000, 4_750_850_450),
            "imagenet_vit_b": (768, 3072, 12, 12, 1000, 17_867_607_866),
            "imagenet_vit_l": (1024, 4096, 16, 24, 1000, 62_360_701_602),
        }
        for name, (hidden, intermediate, heads, layers, classes, total) in cases.items():
            with self.subTest(model=name):
                result = estimate_vit_cost(
                    configuration(
                        image_size=224,
                        patch_size=16,
                        hidden=hidden,
                        intermediate=intermediate,
                        heads=heads,
                        layers=layers,
                        classes=classes,
                    )
                )
                self.assertEqual(result["cost_model_version"], COST_MODEL_VERSION)
                self.assertEqual(result["total_sop"], total)
                self.assertAlmostEqual(result["energy_mj"], total * ENERGY_PJ_PER_SOP * 1e-9)

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_vit_cost(configuration(image_size=10, patch_size=4))
        with self.assertRaises(ValueError):
            estimate_vit_cost(configuration(hidden=5, heads=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
