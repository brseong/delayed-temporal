"""Generate comparison artifacts from validated deterministic ViT evaluations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.vit_comparison_costs import estimate_vit_cost
from scripts.runtime import files as runtime_files
from scripts.runtime import identity as runtime_identity


MODEL_KEYS = (
    "cifar10_vit_small", "imagenet_vit_small", "imagenet_vit_base",
    "imagenet_vit_large",
)
KINDS = ("collect", "dense", "spiking")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"Invalid {name}")
    return value


def _require_identity(row: dict, expected: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in expected and row.get(key) != expected[key]:
            raise ValueError(f"Result identity mismatch: {key}")


def validate_results(experiment: dict, runs: list[dict]) -> tuple[list[dict], dict]:
    """Recheck completeness, identities, exact counts and frozen calibration links.

    The execution controller must first authenticate logs and their hashes. These
    checks are additional safeguards, not a replacement for parsing raw logs.
    """
    models = experiment.get("models")
    version = experiment.get("vit_calibration_policy_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unknown ViT calibration policy")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("The comparison requires exactly four model records")
    by_model = {model["model_key"]: model for model in models}
    if len(by_model) != 4 or set(by_model) != set(MODEL_KEYS):
        raise ValueError("Unexpected or duplicate comparison model")
    _digest(experiment.get("source_commit"), "source_commit", length=40)
    for model in models:
        key = model["model_key"]
        expected_task = "cifar10" if key.startswith("cifar10") else "imagenet-1k"
        expected_samples = 10000 if expected_task == "cifar10" else 5000
        if model.get("task") != expected_task:
            raise ValueError("Model and task disagree")
        if model.get("expected_samples") != expected_samples:
            raise ValueError("Unexpected evaluation sample count")
        if float(model.get("theta", 40)) != 40 or model.get("precision", "float64") != "float64":
            raise ValueError("Comparison requires theta 40 and float64")
        _digest(model.get("checkpoint_sha256"), "checkpoint_sha256")
        cost = estimate_vit_cost(model["checkpoint_config"])
        if cost["configuration"]["classes"] != (10 if expected_task == "cifar10" else 1000):
            raise ValueError("Checkpoint class count does not match task")
        expected_hidden = 1024 if key.endswith("large") else 768 if key.endswith("base") else 384
        expected_depth = 24 if key.endswith("large") else 12
        if (cost["configuration"]["hidden"], cost["configuration"]["depth"]) != (expected_hidden, expected_depth):
            raise ValueError("Checkpoint dimensions do not match comparison architecture")

    indexed: dict[tuple[str, str], dict] = {}
    identifiers: set[str] = set()
    normalized = []
    for supplied in runs:
        row = dict(supplied)
        model_key, kind = row.get("model_key"), row.get("kind")
        if model_key not in by_model or kind not in KINDS:
            raise ValueError("Only the twelve planned comparison conditions are accepted")
        if row.get("success") is not True:
            raise ValueError("Incomplete or unsuccessful result")
        if (model_key, kind) in indexed:
            raise ValueError("Duplicate model condition")
        model = by_model[model_key]
        _require_identity(row, experiment, ("source_commit",))
        if version == 2:
            _require_identity(row, experiment, ("vit_calibration_policy_version",))
        _require_identity(row, model, ("checkpoint_sha256",))
        if row.get("precision") != "float64" or row.get("theta") != 40:
            raise ValueError("Unexpected numerical evaluation condition")
        if row.get("backend") not in ({"hf"} if kind == "dense" else {"spiking"}):
            raise ValueError("Condition and evaluator backend disagree")
        run_id = row.get("run_id", f"{model_key}_{kind}")
        if not isinstance(run_id, str) or not run_id or run_id in identifiers:
            raise ValueError("Duplicate or invalid run_id")
        row["run_id"] = run_id
        identifiers.add(run_id)
        for field in ("log_sha256", "task_sha256", "experiment_sha256"):
            _digest(row.get(field), field)
        if _integer(row.get("batch_size"), "batch_size", minimum=1) not in {8, 16, 32}:
            raise ValueError("Unexpected batch size")
        if kind == "collect":
            _digest(row.get("calibration_sha256"), "calibration_sha256")
            _require_identity(row, model, ("calibration_dataset_fingerprint",))
            _require_identity(row, experiment, ("calibration_evaluator_sha256",))
            sites = row.get("sites")
            site_count = len(sites) if isinstance(sites, (dict, list)) else sites
            if experiment.get("vit_calibration_policy_version") == 2:
                from scripts.experiments.vit_comparison import calibration_site_records
                expected_sites = len(calibration_site_records(experiment, model))
                _require_identity(row, experiment, ("vit_calibration_policy_version",))
            else:
                expected_sites = 4 * int(model["checkpoint_config"]["num_hidden_layers"])
            if site_count != expected_sites:
                raise ValueError("Calibration site count disagrees with checkpoint depth")
            if row.get("samples", row.get("total")) != 5000:
                raise ValueError("Calibration must use training 5k")
        else:
            _require_identity(row, model, ("dataset_fingerprint",))
            _require_identity(row, experiment, ("evaluator_sha256",))
            samples = _integer(row.get("samples", row.get("total")), "samples", minimum=1)
            correct = _integer(row.get("correct"), "correct")
            if "total" in row and row["total"] != samples:
                raise ValueError("Conflicting sample counts")
            if samples != model["expected_samples"] or correct > samples:
                raise ValueError("Incorrect or partial evaluation sample count")
            accuracy = float(row.get("accuracy", float("nan")))
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("Accuracy must be finite and in [0, 1]")
            if not math.isclose(accuracy, correct / samples, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Accuracy and exact correct count disagree")
            prediction = row.get("prediction_sha256", row.get("prediction_digest"))
            _digest(prediction, "prediction_sha256")
            if "prediction_digest" in row and row["prediction_digest"] != prediction:
                raise ValueError("Conflicting prediction hashes")
            row.update(samples=samples, correct=correct, accuracy=accuracy,
                       prediction_sha256=prediction)
            if kind == "spiking":
                _digest(row.get("calibration_sha256"), "calibration_sha256")
                _require_identity(row, experiment, ("gelu_evaluator_sha256",))
        indexed[(model_key, kind)] = row
        normalized.append(row)

    for model_key in MODEL_KEYS:
        collect = indexed.get((model_key, "collect"))
        snn = indexed.get((model_key, "spiking"))
        ann = indexed.get((model_key, "dense"))
        if snn:
            if collect is None or snn["calibration_sha256"] != collect["calibration_sha256"]:
                raise ValueError("SNN result has no matching completed calibration")
        batches = {r["batch_size"] for r in (collect, snn, ann) if r and "batch_size" in r}
        if len(batches) > 1:
            raise ValueError("A model must use one frozen batch size")
    if len({row["experiment_sha256"] for row in normalized}) > 1:
        raise ValueError("Results belong to different experiment manifests")
    if "experiment_sha256" in experiment:
        for row in normalized:
            _require_identity(row, experiment, ("experiment_sha256",))
    normalized.sort(key=lambda r: (MODEL_KEYS.index(r["model_key"]), KINDS.index(r["kind"])))
    return normalized, indexed


def comparison_rows(experiment: dict, indexed: dict) -> tuple[list[dict], list[dict]]:
    """Join matched ANN/SNN counts without introducing stochastic intervals."""
    models = {model["model_key"]: model for model in experiment["models"]}
    summary, breakdown = [], []
    for key in MODEL_KEYS:
        model = models[key]
        cost = estimate_vit_cost(model["checkpoint_config"])
        ann, snn = indexed.get((key, "dense")), indexed.get((key, "spiking"))
        row = {
            "model_key": key, "task": model["task"], "architecture": model["architecture"],
            "complete": bool(ann and snn), "samples": model["expected_samples"],
            "theta": 40, "precision": "float64", "checkpoint_sha256": model["checkpoint_sha256"],
            "ann_correct": ann["correct"] if ann else None,
            "snn_correct": snn["correct"] if snn else None,
            "ann_accuracy_percent": 100 * ann["accuracy"] if ann else None,
            "snn_accuracy_percent": 100 * snn["accuracy"] if snn else None,
            "accuracy_difference_pp": 100 * (snn["accuracy"] - ann["accuracy"]) if ann and snn else None,
            "data_sop": cost["data_sop"], "global_sop": cost["global_sop"],
            "total_sop": cost["total_sop"], "ops_billions": cost["ops_billions"],
            "energy_mj": cost["energy_mj"], "energy_pj_per_sop": cost["energy_pj_per_sop"],
            "cost_model_version": cost["cost_model_version"],
        }
        summary.append(row)
        breakdown.extend({"model_key": key, **stage} for stage in cost["breakdown"])
    return summary, breakdown


def render_latex_rows(summary: list[dict]) -> str:
    """Render four complete Ours rows; literature values are never rewritten."""
    if len(summary) != 4 or {r["model_key"] for r in summary} != set(MODEL_KEYS):
        raise ValueError("Expected four comparison rows")
    if any(row.get("complete") is not True for row in summary):
        raise ValueError("Refusing to publish incomplete comparison rows")
    lines = ["% Generated from validated raw_runs.csv and summary.csv; do not hand edit."]
    for row in sorted(summary, key=lambda r: MODEL_KEYS.index(r["model_key"])):
        architecture = {"ViT-S/16": "ViT-S", "ViT-B/16": "ViT-B", "ViT-L/16": "ViT-L"}[row["architecture"]]
        task = "CIFAR-10" if row["task"] == "cifar10" else "ImageNet-1k"
        lines.append(
            f"% {row['model_key']}\n"
            f"{task} & {architecture} & \\textbf{{Ours}} & Continuous & TTFS & "
            f"{row['ann_accuracy_percent']:.2f} & {row['snn_accuracy_percent']:.2f} & "
            f"{row['ops_billions']:.2f} & {row['energy_mj']:.1f} \\\\"
        )
    return "\n".join(lines) + "\n"


def _csv(rows: list[dict], *, empty_fields: tuple[str, ...] = ()) -> str:
    fields = sorted({key for row in rows for key in row}) or list(empty_fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                         if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def build_outputs(experiment: dict, validated_runs: list[dict], output_dir: Path,
                  *, require_complete: bool = False) -> dict:
    """Write progressive CSVs and complete-only LaTeX with a final status marker."""
    normalized, indexed = validate_results(experiment, validated_runs)
    complete = len(indexed) == 12
    if require_complete and not complete:
        raise ValueError("Four calibrations and eight evaluations must complete")
    output_dir = Path(output_dir)
    previous_path = output_dir / "provenance.json"
    experiment_identity = runtime_identity.json_sha256(experiment)
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous.get("experiment_content_sha256") != experiment_identity:
            raise ValueError("Output directory belongs to a different experiment")
        if previous.get("complete") and not complete:
            raise ValueError("Refusing to replace complete results with partial results")
    summary, breakdown = comparison_rows(experiment, indexed)
    files = {
        "raw_runs.csv": _csv(normalized, empty_fields=("run_id", "model_key", "kind")),
        "summary.csv": _csv(summary), "sop_breakdown.csv": _csv(breakdown),
    }
    if complete:
        files["ours_rows.tex"] = render_latex_rows(summary)
    provenance = {
        "tag": experiment["tag"], "complete": complete,
        "experiment_content_sha256": experiment_identity, "experiment": experiment,
        "validated_results_content_sha256": runtime_identity.json_sha256(normalized),
        "validated_results": normalized,
        "calibrations_complete": sum(kind == "collect" for _, kind in indexed),
        "evaluations_complete": sum(kind != "collect" for _, kind in indexed),
        "files": {
            name: runtime_identity.sha256_bytes(text.encode()) for name, text in files.items()
        },
        "cost_assumptions": estimate_vit_cost(experiment["models"][0]["checkpoint_config"])["assumptions"],
        "publication_note": "ImageNet results use fixed validation 5k; CIFAR-10 uses test 10k. "
                            "ANN and SNN use identical examples and preprocessing. "
                            "Energy is a SOP-based estimate at 0.9 pJ/SOP, not measured device energy.",
    }
    # A consumer must verify every file against provenance, which is replaced last.
    for name, content in files.items():
        runtime_files.atomic_text(output_dir / name, content)
    runtime_files.atomic_text(
        previous_path,
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return provenance


def verify_publication_bundle(output_dir: Path) -> dict:
    """Check completion and every generated checksum before paper integration."""
    output_dir = Path(output_dir)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    if provenance.get("complete") is not True:
        raise ValueError("Comparison bundle is incomplete")
    if provenance.get("calibrations_complete") != 4 or provenance.get("evaluations_complete") != 8:
        raise ValueError("Comparison completion counts are incorrect")
    if set(provenance["files"]) != {"raw_runs.csv", "summary.csv", "sop_breakdown.csv", "ours_rows.tex"}:
        raise ValueError("Unexpected comparison bundle members")
    for name, digest in provenance["files"].items():
        if runtime_identity.sha256_file(output_dir / name) != digest:
            raise ValueError(f"Generated artifact checksum mismatch: {name}")
    experiment = provenance["experiment"]
    if runtime_identity.json_sha256(experiment) != provenance["experiment_content_sha256"]:
        raise ValueError("Experiment provenance checksum mismatch")
    runs, indexed = validate_results(experiment, provenance["validated_results"])
    if (len(indexed) != 12
            or runtime_identity.json_sha256(runs) != provenance["validated_results_content_sha256"]):
        raise ValueError("Validated result provenance checksum mismatch")
    summary, breakdown = comparison_rows(experiment, indexed)
    reproduced = {
        "raw_runs.csv": _csv(runs), "summary.csv": _csv(summary),
        "sop_breakdown.csv": _csv(breakdown), "ours_rows.tex": render_latex_rows(summary),
    }
    for name, content in reproduced.items():
        if content.encode() != (output_dir / name).read_bytes():
            raise ValueError(f"Artifact disagrees with validated results: {name}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--validated-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    results = json.loads(args.validated_results.read_text())
    if isinstance(results, dict):
        results = results["runs"]
    status = build_outputs(experiment, results, args.output_dir,
                           require_complete=args.require_complete)
    print(json.dumps({"complete": status["complete"],
                      "evaluations_complete": status["evaluations_complete"]}, sort_keys=True))


if __name__ == "__main__":
    main()
