"""Immutable conditions for the four-checkpoint, noise-free ViT comparison."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any

from scripts.experiments.calibrated_three_sweeps import parse_result_log
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu

LEGACY_TAG = "vit_conversion_comparison_theta40_calibrated_float64_bounds3_v1"
PREVIOUS_TAG = "vit_conversion_comparison_theta40_calibrated_float64_bounds3_v2"
TAG = "conversion_comparison_theta40_calibrated_float64_bounds3_v3"
MODEL_KEYS = ("cifar10_vit_small", "imagenet_vit_small", "imagenet_vit_base", "imagenet_vit_large")
KINDS = {"collect", "dense", "spiking", "smoke_collect", "smoke_spiking", "environment_spiking"}
PYTHON = "/opt/conda/envs/dt/bin/python"
SOURCE = "/data/delayed-temporal-worktrees/full-calibrated-comparison"
DEFAULT_ROOT = "/data/delayed-temporal/artifacts/logs/conversion_comparison/" + TAG


def require_gpu(experiment: dict[str, Any], host_label: str) -> str:
    """Allow every local A6000 only for the explicitly versioned comparison campaign."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = tuple(experiment.get("local_gpu_ids", []))
    if host_label == "local" and visible.isdecimal() and int(visible) < 4:
        if experiment.get("campaign_extra_local_gpus") != [0, 1, 2, 3]:
            raise ValueError("GPU 0 through 3 require the campaign-specific manifest override")
    return local_gpu.require_single_gpu(
        experiment["python_bin"], host_label, allowed_local_gpus=allowed,
    )


def check_source(experiment: dict[str, Any]) -> None:
    """Verify the comparison's frozen checkout and recorded executable inputs."""
    source = Path(experiment["source_root"])
    identity.verify_clean_checkout(source, experiment["source_commit"])
    files = {
        source / experiment[f"{prefix}_path"]: experiment[f"{prefix}_sha256"]
        for prefix in ("evaluator", "calibration_evaluator", "gelu_evaluator")
    }
    files.update({
        runtime_files.safe_output(source, relative): expected
        for relative, expected in experiment.get("runtime_sha256", {}).items()
    })
    identity.verify_file_identities(files)
    dependencies = experiment.get("dependency_sha256", {})
    if set(dependencies) != {"transformers", "spikingjelly"}:
        raise ValueError("Editable dependency source identities are required")
    identity.verify_package_identities({
        source / "src" / package / subtree: dependencies[package]
        for package, subtree in (("transformers", "src"), ("spikingjelly", "spikingjelly"))
    })


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def model_by_key(experiment: dict, key: str) -> dict:
    rows = [row for row in experiment["models"] if row["model_key"] == key]
    if len(rows) != 1:
        raise ValueError("Unknown or duplicate comparison model")
    return rows[0]


def calibration_policy_version(experiment: dict) -> int:
    """Recognize archived evidence separately from the current run contract."""
    version = experiment.get("vit_calibration_policy_version")
    if experiment.get("tag") == LEGACY_TAG and version is None:
        return 1
    if experiment.get("tag") in {PREVIOUS_TAG, TAG} and type(version) is int and version == 2:
        return 2
    raise ValueError("Comparison tag and calibration policy differ")


def require_current_experiment(experiment: dict) -> None:
    if calibration_policy_version(experiment) != 2:
        raise ValueError("Archived comparison evidence is read-only; new execution requires policy 2")


def calibration_site_records(experiment: dict, model: dict) -> list[dict]:
    """Read the topology captured from actual modules, retaining archived lookup."""
    depth = model["checkpoint_config"]["num_hidden_layers"]
    names = {(f"vit.encoder.layer.{index}{suffix}", name)
             for index in range(depth)
             for suffix, name in (("", "attention_residual"), ("", "output"),
                                  (".attention.attention", "attention_score"),
                                  (".intermediate", "activation_input"))}
    if calibration_policy_version(experiment) == 1:
        return [{"module_name": module, "tensor_name": name} for module, name in sorted(names)]
    names.update((f"vit.encoder.layer.{index}.attention.attention", name)
                 for index in range(depth) for name in ("query", "key", "value"))
    names.update((f"vit.encoder.layer.{index}.{name}", "centered_input")
                 for index in range(depth) for name in ("layernorm_before", "layernorm_after"))
    names.add(("vit.layernorm", "centered_input"))
    records = model.get("calibration_sites", [])
    if len(records) != len(names) or {(r["module_name"], r["tensor_name"]) for r in records} != names:
        raise ValueError("Stored calibration sites differ from the complete model topology")
    for record in records:
        policy = "signed_symmetric_ceiling" if record["tensor_name"] == "attention_score" else "signed_symmetric"
        if record.get("range_policy") != policy:
            raise ValueError("Stored calibration site policy differs")
        endpoints = (record.get("fixed_min"), record.get("fixed_max"))
        if policy == "signed_symmetric_ceiling":
            low, high = endpoints
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not math.isfinite(high) or high <= 0 or low != -high:
                raise ValueError("Invalid stored attention score ceiling")
        elif endpoints != (None, None):
            raise ValueError("Unexpected calibration endpoint limit")
    return records


def validate_experiment(experiment: dict) -> None:
    version = calibration_policy_version(experiment)
    for key, value in {"theta": 40.0, "precision": "float64",
                       "output_bounds_version": 3, "tau_s": 1.0,
                       "local_gpu_ids": list(range(8)), "evaluation_count": 8,
                       "campaign_extra_local_gpus": [0, 1, 2, 3],
                       "calibration_count": 4, "tracking": "disabled"}.items():
        if experiment.get(key) != value:
            raise ValueError(f"Experiment contract mismatch: {key}")
    rows = experiment.get("models", [])
    if len(rows) != 4 or {row.get("model_key") for row in rows} != set(MODEL_KEYS):
        raise ValueError("Exactly four comparison checkpoints are required")
    if not re.fullmatch(r"[a-f0-9]{40}", experiment.get("source_commit", "")):
        raise ValueError("A frozen full source commit is required")
    for name in ("evaluator_sha256", "calibration_evaluator_sha256", "gelu_evaluator_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", experiment.get(name, "")):
            raise ValueError("Missing evaluator hash")
    dependencies = experiment.get("dependency_sha256", {})
    if set(dependencies) != {"transformers", "spikingjelly"} or any(
            not re.fullmatch(r"[a-f0-9]{64}", value) for value in dependencies.values()):
        raise ValueError("Missing dependency hash")
    runtime = experiment.get("runtime_sha256", {})
    if not {"scripts/experiments/run_vit_comparison.py", "scripts/experiments/vit_comparison.py"}.issubset(runtime):
        raise ValueError("Missing runtime identity")
    if any(Path(path).is_absolute() or ".." in Path(path).parts or not re.fullmatch(r"[a-f0-9]{64}", digest)
           for path, digest in runtime.items()):
        raise ValueError("Missing runtime identity")
    packages = experiment.get("package_versions", {})
    if set(packages) != {"torch", "torchvision", "transformers", "spikingjelly", "numpy", "datasets",
                         "safetensors", "timm", "tokenizers", "scipy"} or any(
            not isinstance(value, str) or not value for value in packages.values()):
        raise ValueError("Missing dependency package versions")
    for row in rows:
        cifar = row["model_key"] == "cifar10_vit_small"
        if row.get("task") != ("cifar10" if cifar else "imagenet-1k"):
            raise ValueError("Model dataset mismatch")
        if row.get("expected_samples") != (10000 if cifar else 5000):
            raise ValueError("Comparison sample count differs")
        if row.get("evaluation_split") != ("test" if cifar else "validation"):
            raise ValueError("Comparison split differs")
        if row.get("evaluation_quick_test") is not (not cifar):
            raise ValueError("Comparison sample count differs")
        for field in ("dataset_fingerprint", "calibration_dataset_fingerprint"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Dataset fingerprint is missing")
        if not re.fullmatch(r"[a-f0-9]{64}", row.get("preprocessing_sha256", "")):
            raise ValueError("Missing preprocessing hash")
        config = row["checkpoint_config"]
        if len(config["id2label"]) != (10 if cifar else 1000):
            raise ValueError("Checkpoint class count differs")
        expected_depth = 24 if row["model_key"] == "imagenet_vit_large" else 12
        if config["num_hidden_layers"] != expected_depth:
            raise ValueError("Unexpected checkpoint depth")
        if version == 2:
            epsilon = config.get("layer_norm_eps")
            if isinstance(epsilon, bool) or not isinstance(epsilon, (float, int)) or not math.isfinite(epsilon) or epsilon < 0:
                raise ValueError("Checkpoint LayerNorm epsilon is missing or invalid")
            calibration_site_records(experiment, row)
        width, heads, intermediate = ((1024, 16, 4096) if expected_depth == 24 else
                                      (768, 12, 3072) if row["model_key"] == "imagenet_vit_base" else
                                      (384, 6, 1536))
        geometry = {"hidden_size": width, "num_attention_heads": heads, "intermediate_size": intermediate,
                    "image_size": 224, "patch_size": 16, "num_channels": 3}
        if any(config.get(key) != value for key, value in geometry.items()):
            raise ValueError("Invalid checkpoint geometry")
        for field in ("checkpoint", "dataset", "calibration_dataset"):
            if not Path(row[field + "_path"]).is_absolute():
                raise ValueError("Asset paths must be absolute")
            if not re.fullmatch(r"[a-f0-9]{64}", row.get(field + "_sha256", "")):
                raise ValueError("Missing asset hash")


def make_task(experiment: dict, model_key: str, kind: str, batch_size: int,
              *, calibration_sha256: str = "", host_label: str | None = None) -> dict:
    if kind not in KINDS or batch_size not in (32, 16, 8):
        raise ValueError("Unapproved task or batch size")
    model = model_by_key(experiment, model_key)
    smoke = kind in {"smoke_collect", "smoke_spiking", "environment_spiking"}
    collect = kind in {"collect", "smoke_collect"}
    dense = kind == "dense"
    run_id = f"{model_key}_{kind}" + (f"_bs{batch_size}" if smoke else "")
    if kind == "environment_spiking":
        if model_key != "imagenet_vit_base" or host_label != "ubai":
            raise ValueError("Environment replay is the UBAI ViT-B comparison")
        run_id += "_ubai"
    calibration_file = (f"smoke/{model_key}/bs{batch_size}/calibration.json" if smoke
                        else f"calibration/{model_key}.json")
    task = {"run_id": run_id, "model_key": model_key, "kind": kind,
            "backend": "hf" if dense else "spiking", "batch_size": batch_size,
            "theta": 40.0, "precision": "float64", "seed": None,
            "split": "train" if collect else model["evaluation_split"],
            "expected_samples": 2 * batch_size if smoke else 5000 if collect else model["expected_samples"],
            "calibration_samples": 2 * batch_size if smoke else 5000,
            "calibration_file": "" if dense else calibration_file,
            "calibration_sha256": calibration_sha256,
            "log_file": f"logs/{run_id}.log", "result_file": f"results/{run_id}.json",
            "dataset_fingerprint": model["calibration_dataset_fingerprint" if collect else "dataset_fingerprint"],
            "calibration_dataset_fingerprint": model["calibration_dataset_fingerprint"],
            "checkpoint_sha256": model["checkpoint_sha256"], "gpu_family": "rtxa6000",
            "preprocessing_sha256": model["preprocessing_sha256"],
            "time_noise_std_frac": 0.0, "time_noise_std_abs": 0.0,
            "deadline_margin_std": 0.0, "deadline_margin_abs": 0.0}
    for key in ("source_commit", "evaluator_sha256", "calibration_evaluator_sha256", "gelu_evaluator_sha256"):
        task[key] = experiment[key]
    if calibration_policy_version(experiment) == 2:
        task["vit_calibration_policy_version"] = 2
    if host_label is not None:
        task["host_label"] = host_label
    return task


def validate_task(task: dict, experiment: dict) -> None:
    expected = make_task(experiment, task["model_key"], task["kind"], task["batch_size"],
                         calibration_sha256=task["calibration_sha256"], host_label=task.get("host_label"))
    if task != expected:
        raise ValueError("Task differs from its frozen comparison contract")
    if task["kind"] not in {"collect", "smoke_collect", "dense"}:
        if not re.fullmatch(r"[a-f0-9]{64}", task["calibration_sha256"]):
            raise ValueError("Evaluation requires an assigned calibration hash")


def validate_table(path: Path, task: dict, experiment: dict) -> dict:
    model = model_by_key(experiment, task["model_key"])
    table = read_json(path)
    metadata = table["metadata"]
    layers = table["layers"]
    records = calibration_site_records(experiment, model)
    expected_sites = len(records)
    if len(layers) != expected_sites or len({(r["module_name"], r["tensor_name"]) for r in layers}) != expected_sites:
        raise ValueError("Incomplete or duplicate model calibration sites")
    expected_names = {(r["module_name"], r["tensor_name"]) for r in records}
    if {(r["module_name"], r["tensor_name"]) for r in layers} != expected_names:
        raise ValueError("Calibration model sites differ")
    for key, value in {"theta": 40.0, "dtype": "float64", "dataset_split": "train",
                       "model_id": model["checkpoint_path"], "tau_s": 1.0, "tau_m": 1.0,
                       "clip_margin": 1e-5, "dataset_id": model["task"]}.items():
        if metadata.get(key) != value:
            raise ValueError(f"Calibration metadata mismatch: {key}")
    options = dict(metadata["model_options"])
    expected = {"source_commit": experiment["source_commit"], "output_bounds_version": 3,
                "checkpoint_sha256": model["checkpoint_sha256"],
                "gelu_evaluator_sha256": experiment["gelu_evaluator_sha256"],
                "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
                "calibration_source_fingerprint": model["calibration_dataset_fingerprint"],
                "calibration_dataset_id": model["task"],
                "calibration_image_key": "img" if model["task"] == "cifar10" else "image",
                "calibration_purpose": "final" if task["calibration_samples"] == 5000 else "smoke",
                "vit_evaluator_sha256": experiment["evaluator_sha256"],
                "calibration_wrapper_sha256": experiment["calibration_evaluator_sha256"],
                "attention_implementation": "spiking_sdpa", "spiking_mlp_exact_gelu": False,
                "spiking_ln_mul": True, "spiking_ln_log": True, "spiking_ln_expdiff": True,
                "use_spiking_layernorm": True, "use_spiking_mlp": True}
    if calibration_policy_version(experiment) == 2:
        expected.update(vit_calibration_policy_version=2,
                        layer_norm_eps=model["checkpoint_config"]["layer_norm_eps"],
                        layer_norm_clip_margin=1e-5)
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"Calibration implementation mismatch: {key}")
    preprocessing = json.loads(metadata["preprocessing"])
    if options.get("calibration_dataset_fingerprint") != preprocessing.get("subset_fingerprint"):
        raise ValueError("Calibration subset metadata is inconsistent")
    if preprocessing.get("subset_samples") != task["calibration_samples"] or preprocessing.get("subset_seed") != 0:
        raise ValueError("Calibration subset mismatch")
    if task["calibration_samples"] == 5000 and preprocessing.get("subset_fingerprint") != model["calibration_dataset_fingerprint"]:
        raise ValueError("Final calibration training fingerprint differs")
    for layer in layers:
        low, high = layer["bounds"]["min"], layer["bounds"]["max"]
        if not all(math.isfinite(v) for v in (low, high)) or low >= high:
            raise ValueError("Nonfinite or unordered calibration bounds")
        if calibration_policy_version(experiment) == 2:
            spec = next(r for r in records if (r["module_name"], r["tensor_name"]) == (layer["module_name"], layer["tensor_name"]))
            if any(layer.get(field) != spec[field] for field in ("range_policy", "fixed_min", "fixed_max")):
                raise ValueError("Calibration record policy differs from the frozen model sites")
            if low != -high or (layer["tensor_name"] == "centered_input" and high <= 1e-5):
                raise ValueError("Invalid symmetric calibration encoder range")
            if spec["fixed_max"] is not None and high > spec["fixed_max"]:
                raise ValueError("Calibration score exceeds its numerical ceiling")
        if (layer["lower_quantile"], layer["upper_quantile"], layer["margin_fraction"]) != (0.0, 1.0, 0.05):
            raise ValueError("Calibration endpoint policy differs")
        histogram = layer.get("histogram", {})
        bins = histogram.get("bin_counts", [])
        count = layer.get("num_values", 0)
        tails = [histogram.get(key, -1) for key in ("underflows", "overflows")]
        if (type(count) is not int or count <= 0 or len(bins) != 2048
                or histogram.get("num_values") != count
                or any(type(value) is not int or value < 0 for value in [*bins, *tails])
                or sum(bins) + sum(tails) != count):
            raise ValueError("Calibration histogram is incomplete")
    return table


def validate_log_header(task: dict, text: str) -> None:
    headers = [line.removeprefix("Comparison task — ") for line in text.splitlines()
               if line.startswith("Comparison task — ")]
    if len(headers) != 1 or json.loads(headers[0]) != {
            "task_sha256": identity.json_sha256(task), "batch_size": task["batch_size"]}:
        raise ValueError("Comparison task identity differs")


def parsed_result(task: dict, root: Path) -> dict:
    result = json.loads(json.dumps(parse_result_log(task, root)))
    text = runtime_files.safe_output(root, task["log_file"]).read_text()
    validate_log_header(task, text)
    if task["backend"] == "spiking" and not result["clamp_sites"]:
        raise ValueError("Spiking evaluation lacks named clamp counts")
    if text.splitlines().count(f"Model backend: {task['backend']}") != 1:
        raise ValueError("Backend is absent or duplicated")
    if task["backend"] == "spiking":
        for line in ("Spiking LayerNorm: True, Spiking Attention: True",
                     "  LN stages — mul: True, log: True, expdiff: True", "Spiking MLP: True"):
            if text.splitlines().count(line) != 1:
                raise ValueError("Spiking stage configuration differs")
    result.update(log_file=task["log_file"], total=result["samples"], prediction_digest=result["prediction_sha256"])
    return result


def validate_result(task: dict, result: dict, experiment: dict, root: Path) -> None:
    validate_task(task, experiment)
    if any(result.get(key) != value for key, value in task.items() if key != "calibration_sha256"):
        raise ValueError("Result differs from the assigned task")
    if result.get("success") is not True or result.get("task_sha256") != identity.json_sha256(task):
        raise ValueError("Missing successful task identity")
    if result.get("experiment_sha256") != identity.json_sha256(experiment):
        raise ValueError("Result experiment differs")
    if result.get("host_label") not in {"local", "ubai"}:
        raise ValueError("Missing execution environment")
    if not math.isfinite(result.get("elapsed_seconds", -1)) or result["elapsed_seconds"] < 0:
        raise ValueError("Invalid run duration")
    log = runtime_files.safe_output(root, task["log_file"])
    if result.get("log_sha256") != identity.sha256_file(log):
        raise ValueError("Result log changed")
    if task["kind"] != "dense":
        table = runtime_files.safe_output(root, task["calibration_file"])
        if result.get("calibration_sha256") != identity.sha256_file(table):
            raise ValueError("Calibration artifact changed")
        if task["kind"] not in {"collect", "smoke_collect"} and result["calibration_sha256"] != task["calibration_sha256"]:
            raise ValueError("Assigned calibration differs")
        validate_table(table, task, experiment)
    if task["kind"] in {"collect", "smoke_collect"}:
        expected = len(calibration_site_records(experiment, model_by_key(experiment, task["model_key"])))
        if result.get("sites") != expected:
            raise ValueError("Incomplete calibration collection")
        if result.get("samples") != task["calibration_samples"] or result.get("total") != task["calibration_samples"]:
            raise ValueError("Calibration subset mismatch")
        text = log.read_text()
        validate_log_header(task, text)
        progress = [json.loads(line.removeprefix("Calibration progress — "))
                    for line in text.splitlines() if line.startswith("Calibration progress — ")]
        batches = 2 * math.ceil(task["calibration_samples"] / task["batch_size"])
        expected_progress = {"pass": 2, "completed_batches": batches, "total_batches": batches,
                             "completed_samples": 2 * task["calibration_samples"]}
        if not progress or any(progress[-1].get(key) != value for key, value in expected_progress.items()):
            raise ValueError("Incomplete calibration log")
        if "Traceback (most recent call last)" in text or text.splitlines().count(
            f"Calibration identity — mode: collect, sha256: {result['calibration_sha256']}") != 1:
            raise ValueError("Incomplete calibration log")
    else:
        parsed = parsed_result(task, root)
        if any(result.get(key) != value for key, value in parsed.items()):
            raise ValueError("Result fields differ from validated raw log")


def evaluator_command(experiment: dict, task: dict, root: Path) -> list[str]:
    model = model_by_key(experiment, task["model_key"])
    source = Path(experiment["source_root"])
    dense = task["kind"] == "dense"
    collect = task["kind"] in {"collect", "smoke_collect"}
    smoke = task["kind"] in {"smoke_collect", "smoke_spiking", "environment_spiking"}
    command = [experiment["python_bin"], "-u", str(source / experiment["evaluator_path" if dense else "calibration_evaluator_path"])]
    if not dense:
        command += ["--source-root", str(source), "--calibration-dataset-path", model["calibration_dataset_path"],
                    "--calibration-dataset-fingerprint", model["calibration_dataset_fingerprint"],
                    "--gelu-cubic-implementation", "phi_nl_psi_ed", "--gelu-cubic-floor", "1e-5"]
        if smoke:
            command += ["--calibration-smoke-samples", str(task["calibration_samples"])]
    command += ["--experiment_name", task["run_id"], "--device", "cuda", "--model_backend", task["backend"],
                "--model_id", model["checkpoint_path"], "--dataset_id", model["task"],
                "--evaluation-dataset-path", model["dataset_path"], "--evaluation-split", model["evaluation_split"],
                "--batch_size", str(task["batch_size"]), "--theta", "40", "--precision", "float64",
                "--source-commit", experiment["source_commit"], "--checkpoint-sha256", model["checkpoint_sha256"],
                "--no-tensorboard", "--report-clamp-stats", "--spiking-layernorm", "--spiking-ln-mul",
                "--spiking-ln-log", "--spiking-ln-expdiff", "--spiking-attention", "--spiking-mlp",
                "--no-spiking-mlp-exact-gelu", "--calibration-mode", "none" if dense else "collect" if collect else "validate",
                "--no-gaussian-time-noise", "--time-noise-seed", "0", "--time-noise-std-frac", "0",
                "--time-noise-mean", "0", "--time-noise-deadline-margin-std", "0",
                "--no-mismatch-enabled", "--mismatch-theta-std", "0", "--mismatch-seed", "0",
                "--weight-noise-std", "0", "--bias-noise-std", "0"]
    if not dense:
        command += [
            "--calibration-path",
            str(runtime_files.safe_output(root, task["calibration_file"])),
            "--calibration-samples", str(task["calibration_samples"]), "--calibration-seed", "0",
            "--calibration-bins", "2048", "--calibration-lower-quantile", "0",
            "--calibration-upper-quantile", "1", "--calibration-margin-fraction", "0.05",
        ]
    if model.get("evaluation_quick_test", model["task"] == "imagenet-1k"):
        command += ["--quick-test"]
    if smoke and not collect:
        command += ["--max_eval_batches", "2"]
    return command
