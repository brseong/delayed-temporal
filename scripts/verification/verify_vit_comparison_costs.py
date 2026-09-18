"""CPU verification of ViT comparison cost accounting and artifact generation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.vit_comparison_costs import estimate_vit_cost
from scripts.analysis.summarize_vit_comparison import (
    MODEL_KEYS, build_outputs, validate_results, verify_publication_bundle,
)
from scripts.analysis.publish_vit_comparison import prepare_paper_update
from scripts.experiments.vit_comparison import theta_grid


def configuration(*, classes: int = 1000, hidden: int = 384, layers: int = 12) -> dict:
    return dict(model_type="vit", hidden_act="gelu", image_size=224, patch_size=16,
                num_channels=3, hidden_size=hidden, intermediate_size=4 * hidden,
                num_attention_heads=hidden // 64, num_hidden_layers=layers,
                num_labels=classes)


def enumerate_small_circuit(config: dict) -> dict[str, Counter]:
    """Count individual deliveries through independent nested graph traversals.

    Each increment denotes one destination synapse. No component polynomial or
    function from the estimator is used by this deliberately small test oracle.
    """
    patches = list(range((config["image_size"] // config["patch_size"]) ** 2))
    tokens = list(range(len(patches) + 1))
    features = list(range(config["hidden_size"]))
    mlp_features = list(range(config["intermediate_size"]))
    patch_features = list(range(config["num_channels"] * config["patch_size"] ** 2))
    heads = list(range(config["num_attention_heads"]))
    head_features = list(range(len(features) // len(heads)))
    totals: dict[str, Counter] = {}

    def event(stage: str, kind: str) -> None:
        totals.setdefault(stage, Counter())[kind] += 1

    def linear(stage: str, rows: list, inputs: list, outputs: list) -> None:
        event(stage, "global")  # shared scalar reference source
        for _ in rows:
            for _ in inputs:
                event(stage, "global")  # input encoder
                for _ in outputs:
                    event(stage, "data")
            for _ in outputs:
                event(stage, "global")  # reference delivered to the sum

    def multiply(stage: str, coordinates: list) -> None:
        event(stage, "global")
        for _ in coordinates:
            event(stage, "global")  # data encoder synchronization
            event(stage, "data")
            event(stage, "global")  # reference delivered to product

    def layernorm(prefix: str) -> None:
        coordinates = [(token, feature) for token in tokens for feature in features]
        for _ in ("positive", "negative"):
            multiply(prefix + ".square", coordinates)
        stage = prefix + ".log_and_exponential_difference"
        for _ in tokens:
            event(stage, "global")  # variance encoder
            for _ in ("positive", "negative"):
                for _ in features:
                    event(stage, "data")  # denominator fan-out
                    event(stage, "global")  # residual log encoder
                    event(stage, "data")  # numerator delivery
                    event(stage, "global")  # internal encoder
                    event(stage, "data")
        stage = prefix + ".gamma"
        event(stage, "global")  # gamma product reference source
        for _ in features:
            event(stage, "global")  # gamma encoder, reused across tokens
            for _ in tokens:
                event(stage, "data")
                event(stage, "global")

    linear("patch_embedding", patches, patch_features, features)
    for _ in range(config["num_hidden_layers"]):
        layernorm("block.layernorm")
        layernorm("block.layernorm")
        for _ in ("q", "k", "v"):
            linear("block.attention.qkv", tokens, features, features)
        stage = "block.attention.score"
        event(stage, "global")
        for _ in heads:
            for _ in tokens:  # keys
                for _ in head_features:
                    event(stage, "global")
                    for _ in tokens:  # queries
                        event(stage, "data")
            for _ in tokens:
                for _ in tokens:
                    event(stage, "global")
        for _ in heads:
            for _ in tokens:
                event("block.attention.division", "global")  # denominator encoder
                for _ in tokens:
                    event("block.attention.exponential", "global")
                    event("block.attention.exponential", "data")
                    event("block.attention.division", "global")  # numerator encoder
                    event("block.attention.division", "global")  # internal encoder
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
                event(stage, "global")  # log encoder
                event(stage, "data")
                event(stage, "global")  # reference delivery
                event(stage, "global")  # internal encoder
                event(stage, "data")
            event("block.mlp.gelu.exponential", "global")
            event("block.mlp.gelu.exponential", "data")
            for _ in ("numerator", "denominator", "internal"):
                event("block.mlp.gelu.division", "global")
                event("block.mlp.gelu.division", "data")
        multiply("block.mlp.gelu.product", coordinates)
        linear("block.mlp.output_projection", tokens, mlp_features, features)
    layernorm("final_layernorm")
    linear("classification_head_assumed_ttfs", [0], features, list(range(config["num_labels"])))
    return totals


def fixture() -> tuple[dict, list[dict]]:
    """Construct synthetic local test results; these are not research results."""
    digest = "a" * 64
    experiment = dict(tag="synthetic_comparison_test", source_commit="b" * 40,
                      evaluator_sha256=digest, calibration_evaluator_sha256=digest,
                      gelu_evaluator_sha256=digest, selection_tolerance_correct=25,
                      validation_used_for_selection=False, models=[])
    results = []
    for key in MODEL_KEYS:
        cifar = key.startswith("cifar10")
        large = key.endswith("large")
        hidden = 1024 if large else 768 if key.endswith("base") else 384
        model = dict(model_key=key, task="cifar10" if cifar else "imagenet-1k",
                     architecture="ViT-L/16" if large else "ViT-B/16" if hidden == 768 else "ViT-S/16",
                     checkpoint_config=configuration(classes=10 if cifar else 1000,
                                                     hidden=hidden, layers=24 if large else 12),
                     expected_samples=10000 if cifar else 5000,
                     checkpoint_sha256=digest, precision="float64",
                     dataset_fingerprint="eval_fingerprint",
                     calibration_dataset_fingerprint="train_fingerprint")
        experiment["models"].append(model)
        common = dict(model_key=key, success=True, source_commit=experiment["source_commit"],
                      checkpoint_sha256=digest, log_sha256=digest, task_sha256=digest,
                      experiment_sha256=digest, evaluator_sha256=digest,
                      calibration_evaluator_sha256=digest, gelu_evaluator_sha256=digest,
                      batch_size=32, precision="float64")
        for theta_index, theta in enumerate(theta_grid()):
            candidate = dict(common, theta=theta, theta_index=theta_index, split="train",
                             expected_samples=5000, dataset_fingerprint="train_fingerprint")
            results.append(dict(candidate, run_id=f"{key}_theta_{theta_index:02d}_collect",
                                kind="theta_collect", backend="spiking", samples=5000,
                                sites=4 * model["checkpoint_config"]["num_hidden_layers"],
                                calibration_sha256=digest,
                                calibration_dataset_fingerprint="train_fingerprint"))
            correct = 4000 + min(theta_index, 6) * 100
            results.append(dict(candidate, run_id=f"{key}_theta_{theta_index:02d}_train",
                                kind="theta_train", backend="spiking", samples=5000,
                                correct=correct, accuracy=correct / 5000,
                                prediction_sha256=digest, calibration_sha256=digest))
        selected_index, selected_theta = 6, theta_grid()[6]
        final = dict(common, theta=selected_theta, theta_index=selected_index,
                     split="test" if cifar else "validation",
                     expected_samples=model["expected_samples"],
                     dataset_fingerprint="eval_fingerprint")
        for kind, backend, error in (("dense", "hf", 0), ("spiking", "spiking", 5)):
            samples = model["expected_samples"]
            correct = samples - 100 - error
            results.append(dict(final, run_id=f"{key}_{kind}_theta_{selected_index:02d}",
                                kind=kind, backend=backend,
                                samples=samples, correct=correct, accuracy=correct / samples,
                                prediction_sha256=digest, calibration_sha256=digest,
                                dataset_fingerprint="eval_fingerprint"))
    return experiment, results


class CostTests(unittest.TestCase):
    def test_enumerated_oracle(self) -> None:
        config = dict(model_type="vit", hidden_act="gelu", image_size=4, patch_size=2,
                      num_channels=1, hidden_size=4, intermediate_size=8,
                      num_attention_heads=2, num_hidden_layers=2, num_labels=3)
        estimate = estimate_vit_cost(config)
        oracle = enumerate_small_circuit(config)
        self.assertEqual(set(oracle), {row["component"] for row in estimate["breakdown"]})
        for row in estimate["breakdown"]:
            self.assertEqual(row["data_sop"], oracle[row["component"]]["data"], row["component"])
            self.assertEqual(row["global_sop"], oracle[row["component"]]["global"], row["component"])
        self.assertEqual(estimate["total_sop"], sum(sum(counts.values()) for counts in oracle.values()))

    def test_geometry_classes_and_units(self) -> None:
        small = estimate_vit_cost(configuration())
        cifar = estimate_vit_cost(configuration(classes=10))
        self.assertEqual(small["configuration"]["patches"], 196)
        self.assertEqual(small["configuration"]["tokens"], 197)
        self.assertEqual(small["total_sop"] - cifar["total_sop"], 990 * (384 + 1))
        for estimate in (small, cifar):
            self.assertAlmostEqual(estimate["energy_mj"], estimate["ops_billions"] * 0.9)
            self.assertEqual(estimate["total_sop"], estimate["data_sop"] + estimate["global_sop"])
            self.assertTrue(all(row["total_sop"] > 0 for row in estimate["breakdown"]))
        rows = {r["component"]: r for r in small["breakdown"]}
        self.assertEqual(rows["block.attention.output_projection"]["multiplicity"], 12)
        self.assertEqual(rows["final_layernorm.gamma"]["multiplicity"], 1)
        self.assertEqual(rows["block.mlp.gelu.cubic"]["global_sop_each"], 6 * 197 * 1536 + 1)

    def test_heads_affect_score_arrays_not_dot_products(self) -> None:
        config = configuration()
        original = estimate_vit_cost(config)
        config["num_attention_heads"] *= 2
        changed = estimate_vit_cost(config)
        a = {r["component"]: r for r in original["breakdown"]}
        b = {r["component"]: r for r in changed["breakdown"]}
        self.assertEqual(a["block.attention.score"]["data_sop"], b["block.attention.score"]["data_sop"])
        self.assertEqual(2 * a["block.attention.division"]["total_sop"],
                         b["block.attention.division"]["total_sop"])

    def test_invalid_configs(self) -> None:
        for key, value in (("image_size", 225), ("patch_size", True),
                           ("num_attention_heads", 7), ("num_labels", 0),
                           ("hidden_act", "relu"), ("model_type", "deit")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                estimate_vit_cost(dict(configuration(), **{key: value}))
        config = configuration(classes=10)
        config.pop("num_labels")
        config["id2label"] = {str(i): f"class_{i}" for i in range(10)}
        self.assertEqual(estimate_vit_cost(config)["configuration"]["classes"], 10)


class SummaryTests(unittest.TestCase):
    def test_complete_and_progressive(self) -> None:
        experiment, runs = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            status = build_outputs(experiment, runs[:3], folder)
            self.assertFalse(status["complete"])
            self.assertFalse((folder / "ours_rows.tex").exists())
            with self.assertRaises(ValueError):
                verify_publication_bundle(folder)
            with self.assertRaises(ValueError):
                build_outputs(experiment, runs[:3], folder, require_complete=True)
            status = build_outputs(experiment, runs, folder, require_complete=True)
            self.assertTrue(status["complete"])
            self.assertTrue(verify_publication_bundle(folder)["complete"])
            with (folder / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["snn_accuracy_percent"]), 98.95)
            self.assertIn("98.95", (folder / "ours_rows.tex").read_text())
            self.assertNotIn("ci95", (folder / "summary.csv").read_text())
            with self.assertRaises(ValueError):
                build_outputs(experiment, runs[:3], folder)
            (folder / "summary.csv").write_text("corrupted generated artifact")
            with self.assertRaises(ValueError):
                verify_publication_bundle(folder)

    def test_missing_duplicates_and_partial_results(self) -> None:
        experiment, runs = fixture()
        for variant in (runs + [runs[0]], [runs[1]], [dict(runs[0], success=False)],
                        [dict(runs[0], sites=96)]):
            with self.subTest(variant=variant[0]["kind"]), self.assertRaises(ValueError):
                validate_results(experiment, variant)
        for field, value in (("samples", 4999), ("accuracy", float("nan")),
                             ("correct", 10001), ("accuracy", 0.25),
                             ("prediction_sha256", "bad"), ("total", 123)):
            variant = deepcopy(runs)
            variant[-2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_results(experiment, variant)
        for field in ("theta", "precision", "samples", "batch_size"):
            variant = deepcopy(runs)
            variant[0].pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_results(experiment, variant)

    def test_identity_rejection(self) -> None:
        experiment, runs = fixture()
        for field, value, index in (
            ("source_commit", "c" * 40, -2), ("checkpoint_sha256", "c" * 64, -2),
            ("evaluator_sha256", "c" * 64, -2), ("calibration_sha256", "c" * 64, -1),
            ("experiment_sha256", "c" * 64, -2),
            ("calibration_evaluator_sha256", "c" * 64, 0),
            ("gelu_evaluator_sha256", "c" * 64, -1),
            ("dataset_fingerprint", "other", -2), ("batch_size", 16, -2),
            ("precision", "float32", -1), ("theta", 80, -1),
        ):
            variant = deepcopy(runs)
            variant[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_results(experiment, variant)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            build_outputs(experiment, [], folder)
            other = dict(experiment, tag="another_comparison")
            with self.assertRaises(ValueError):
                build_outputs(other, [], folder)

    def test_aliases_and_all_calibrations_required(self) -> None:
        experiment, runs = fixture()
        for run in runs:
            if run["kind"] != "theta_collect":
                run["total"] = run.pop("samples")
                run["prediction_digest"] = run.pop("prediction_sha256")
        normalized, _ = validate_results(experiment, runs)
        self.assertTrue(all("samples" in row for row in normalized))
        with self.assertRaises(ValueError):
            validate_results(experiment, [row for row in runs if row["kind"] != "theta_collect"])


def paper_fixture() -> str:
    """Minimal synthetic layout, never written to a manuscript directory."""
    lines = ["Unrelated user prose before.\n", r"\begin{table}[ht!]", "\n",
             r"\caption{Original caption with {nested} groups.}", "\n",
             r"\begin{tabular}{lllllllll}", "\n"]
    for task, architecture, steps, ann, snn in (
        ("CIFAR-10", "ViT-S", 32, "99.2", "98.7"),
        ("ImageNet-1k", "ViT-S", 64, "82.34", "81.45"),
        ("", "ViT-B", 64, "83.75", "82.71"),
        ("", "ViT-L", 64, "85.41", "83.82"),
    ):
        lines.append(f"{task} & {architecture} & "
                     r"SpikeZIP-TF$^{\dagger}$~\cite{you2024spikezip} & "
                     f"{steps} & Direct & {ann} & {snn} & -- & -- " + r"\\" + "\n")
        lines.append(r" & & \textbf{Ours} & Continuous & TTFS & -- & -- & -- & -- \\" + "\n")
    lines.extend([r"\end{tabular}", "\n", r"\label{tab:conv-acc}", "\n",
                  r"\end{table}", "\nUnrelated user prose after.\n"])
    return "".join(lines)


class PublicationTests(unittest.TestCase):
    def test_publication_is_complete_scoped_and_idempotent(self) -> None:
        experiment, results = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            build_outputs(experiment, results, folder, require_complete=True)
            original = paper_fixture()
            proposed = prepare_paper_update(original, folder)
            self.assertTrue(proposed.startswith("Unrelated user prose before.\n"))
            self.assertTrue(proposed.endswith("Unrelated user prose after.\n"))
            self.assertIn("98.95", proposed)
            self.assertIn(r"$403.2^{\ddagger}$", proposed)
            self.assertIn(r"$1270.4^{\ddagger}$", proposed)
            self.assertIn("validation 5k", proposed)
            self.assertIn("classifier itself is dense", proposed)
            self.assertEqual(proposed, prepare_paper_update(proposed, folder))
            with self.assertRaises(ValueError):
                prepare_paper_update(original.replace("ViT-B", "ViT-L"), folder)
            with self.assertRaises(ValueError):
                prepare_paper_update(original.replace("83.75", "83.76"), folder)

    def test_publication_rejects_partial_evidence(self) -> None:
        experiment, results = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            build_outputs(experiment, results[:3], folder)
            with self.assertRaises(ValueError):
                prepare_paper_update(paper_fixture(), folder)


if __name__ == "__main__":
    unittest.main()
