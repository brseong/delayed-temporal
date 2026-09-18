"""Verify comparison assignments, result evidence and memory admission without GPUs."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments import run_vit_comparison as runner
from scripts.experiments import vit_comparison as contract
from scripts.runtime import identity
from scripts.runtime import local_gpu


def reject(action, errors=(ValueError, KeyError, FileNotFoundError, RuntimeError)) -> None:
    try:
        action()
    except errors:
        return
    raise AssertionError("Invalid comparison evidence was accepted")


def put_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def topology_fixture(depth: int, *, version: int = 2) -> list[dict]:
    suffixes = [("", "attention_residual"), ("", "output"),
                (".attention.attention", "attention_score"), (".intermediate", "activation_input")]
    if version == 2:
        suffixes += [(".attention.attention", name) for name in ("query", "key", "value")]
        suffixes += [("." + name, "centered_input") for name in ("layernorm_before", "layernorm_after")]
    names = [(f"vit.encoder.layer.{index}{suffix}", name) for index in range(depth) for suffix, name in suffixes]
    if version == 2:
        names.append(("vit.layernorm", "centered_input"))
    ceiling = 0.5 * (-math.log(sys.float_info.min) - math.log(197) - 2.0) if version == 2 else 40.0
    return [{"module_name": module, "tensor_name": name,
             "range_policy": "signed_symmetric_ceiling" if name == "attention_score" else "signed_symmetric",
             "fixed_min": -ceiling if name == "attention_score" else None,
             "fixed_max": ceiling if name == "attention_score" else None} for module, name in names]


def experiment_fixture(root: Path) -> dict:
    models = []
    for index, key in enumerate(contract.MODEL_KEYS):
        cifar, large = index == 0, index == 3
        models.append({
            "model_key": key, "task": "cifar10" if cifar else "imagenet-1k",
            "architecture": "ViT-L/16" if large else "ViT-B/16" if index == 2 else "ViT-S/16",
            "checkpoint_id": key, "checkpoint_path": str(root / "assets" / key),
            "checkpoint_sha256": str(index + 1) * 64,
            "preprocessing_sha256": "a" * 64,
            "image_preprocessing_config": "" if cifar else str(root / "source/scripts/configs/vit_timm_preprocessing.json"),
            "checkpoint_config": {
                "id2label": {str(i): str(i) for i in range(10 if cifar else 1000)},
                "num_hidden_layers": 24 if large else 12,
                "hidden_size": 1024 if large else 768 if index == 2 else 384,
                "num_attention_heads": 16 if large else 12 if index == 2 else 6,
                "intermediate_size": 4096 if large else 3072 if index == 2 else 1536,
                "image_size": 224, "patch_size": 16, "num_channels": 3,
                "layer_norm_eps": 1e-12,
            },
            "calibration_sites": topology_fixture(24 if large else 12),
            "calibration_dataset_path": str(root / "assets" / ("cifar_train" if cifar else "imagenet_train")),
            "calibration_dataset_fingerprint": "training-data",
            "calibration_dataset_sha256": "5" * 64,
            "dataset_path": str(root / "assets" / ("cifar_test" if cifar else "imagenet_validation")),
            "dataset_fingerprint": "evaluation-data", "dataset_sha256": "6" * 64,
            "expected_samples": 10000 if cifar else 5000,
            "evaluation_split": "test" if cifar else "validation", "evaluation_quick_test": not cifar,
        })
    experiment = {
        "tag": contract.TAG, "theta_candidates": list(contract.theta_grid()),
        "precision": "float64", "output_bounds_version": 3,
        "vit_calibration_policy_version": 2,
        "tau_s": 1.0, "tracking": "disabled", "local_gpu_ids": [4, 5, 6, 7],
        "campaign_extra_local_gpus": [],
        "evaluation_count": 4 * (len(contract.theta_grid()) + 2),
        "calibration_count": 4 * len(contract.theta_grid()),
        "selection_tolerance_correct": 25,
        "selection_population": "training_seed0_5000",
        "validation_used_for_selection": False, "models": models,
        "source_commit": "a" * 40, "source_root": str(root / "source"),
        "python_bin": sys.executable, "runtime_root": str(root / "runtime"),
        "evaluator_path": "scripts/evaluation/error_analysis_vit.py", "evaluator_sha256": "7" * 64,
        "calibration_evaluator_path": "scripts/analysis/evaluate_calibrated_vit.py", "calibration_evaluator_sha256": "8" * 64,
        "gelu_evaluator_path": "scripts/analysis/gelu_cubic_phi_nl_vit.py", "gelu_evaluator_sha256": "9" * 64,
        "dependency_sha256": {"transformers": "b" * 64, "spikingjelly": "c" * 64},
        "runtime_sha256": {runner.SCRIPT: "d" * 64, "scripts/experiments/vit_comparison.py": "e" * 64},
        "package_versions": {name: "1.0" for name in (
            "torch", "torchvision", "transformers", "spikingjelly", "numpy", "datasets",
            "safetensors", "timm", "tokenizers", "scipy")},
        "assets_manifest_sha256": "f" * 64, "asset_checks": {},
    }
    return experiment


def table_fixture(experiment: dict, task: dict) -> dict:
    row = contract.model_by_key(experiment, task["model_key"])
    smoke = task["calibration_samples"] != 5000
    fingerprint = "smoke-prefix-data" if smoke else row["calibration_dataset_fingerprint"]
    options = {
        "source_commit": experiment["source_commit"], "output_bounds_version": 3,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "gelu_evaluator_sha256": experiment["gelu_evaluator_sha256"],
        "vit_evaluator_sha256": experiment["evaluator_sha256"],
        "calibration_wrapper_sha256": experiment["calibration_evaluator_sha256"],
        "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
        "calibration_dataset_fingerprint": fingerprint,
        "calibration_source_fingerprint": row["calibration_dataset_fingerprint"],
        "calibration_dataset_id": row["task"], "calibration_image_key": "img" if row["task"] == "cifar10" else "image",
        "calibration_purpose": "smoke" if smoke else "final",
        "attention_implementation": "spiking_sdpa", "hidden_act": "gelu", "gelu_output_min": -0.170041,
        "spiking_ln_mul": True, "spiking_ln_log": True, "spiking_ln_expdiff": True,
        "use_spiking_layernorm": True, "use_spiking_mlp": True, "spiking_mlp_exact_gelu": False,
    }
    version = contract.calibration_policy_version(experiment)
    if version == 2:
        options.update(vit_calibration_policy_version=2, layer_norm_eps=1e-12,
                       layer_norm_clip_margin=1e-5)
    layers = []
    for spec in topology_fixture(row["checkpoint_config"]["num_hidden_layers"], version=version):
        layers.append({
                **spec,
                "bounds": {"min": -1.0, "max": 1.0}, "observed_min": -0.9, "observed_max": 0.9,
                "num_values": 100, "lower_quantile": 0.0, "upper_quantile": 1.0, "margin_fraction": 0.05,
                "histogram": {"bounds": {"min": -0.9, "max": 0.9}, "num_values": 100,
                              "underflows": 0, "overflows": 0, "bin_counts": [100] + [0] * 2047},
            })
    preprocessing = {"subset_samples": task["calibration_samples"], "subset_seed": 0,
                     "subset_fingerprint": fingerprint, "subset_selection": "seeded_training_permutation_prefix"}
    if row["task"] == "imagenet-1k":
        preprocessing.update(preprocessing_backend="timm", preprocessing_config_sha256=row["preprocessing_sha256"])
    return {"format_version": 1, "layers": layers, "metadata": {
        "theta": task["theta"], "dtype": "float64", "dataset_split": "train", "dataset_id": row["task"],
        "model_id": row["checkpoint_path"], "model_family": "vit", "input_shape": [3, 224, 224],
        "tau_s": 1.0, "tau_m": 1.0, "clip_margin": 1e-5, "max_sequence_length": None,
        "model_options": sorted(options.items()),
        "preprocessing": json.dumps(preprocessing),
    }}


def log_fixture(experiment: dict, task: dict) -> str:
    row = contract.model_by_key(experiment, task["model_key"])
    samples = task["expected_samples"]
    correct = (3000 + min(task["theta_index"], 6) * 100
               if task["kind"] == "theta_train" else samples * 4 // 5)
    text = (
        "Slurm identity — job: fixture, gpu_family: rtxa6000\n"
        "Comparison task — " + json.dumps({
            "task_sha256": identity.json_sha256(task), "batch_size": task["batch_size"],
        }) + "\n"
        "GPU model: NVIDIA RTX A6000\n"
        f"Model backend: {task['backend']}\n"
        f"Artifact identity — source_commit: {task['source_commit']}, checkpoint_sha256: {task['checkpoint_sha256']}\n"
        f"Gaussian time noise — enabled: False, std_frac: 0.0, identity_window: {2 * task['theta']}, "
        "std_abs: 0.0, mean_abs: 0.0, seed: 0, identity_deadline_ulp: 1e-12, "
        "std_to_identity_ulp: 0.0, deadline_margin_std: 0.0, deadline_margin_abs: 0.0\n"
        "Static threshold mismatch — enabled: False, theta_std: 0.0, seed: 0\n"
        f"Evaluation metadata — model: {row['checkpoint_path']}, dataset: {row['task']}, split: {task['split']}, "
        f"samples: {samples}, theta: {task['theta']}, precision: float64, source: disk:{row['dataset_path']}, "
        f"fingerprint: {task['dataset_fingerprint']}\n"
        f"Correct: {correct}\nEvaluated samples: {samples}\nPrediction SHA256: {'f' * 64}\nAccuracy: {correct / samples}\n"
    )
    if task["backend"] == "spiking":
        text += (
            "GELU cubic implementation: phi_nl_psi_ed\nGELU cubic magnitude floor: 1e-05\n"
            f"Calibration identity — mode: validate, sha256: {task['calibration_sha256']}\n"
            "Spiking LayerNorm: True, Spiking Attention: True\n"
            "  LN stages — mul: True, log: True, expdiff: True\nSpiking MLP: True\n"
            "Clamp[layernorm/test] values=100, underflows=1 (rate=0.01), overflows=2 (rate=0.02)\n"
        )
    return text


def completed_fixture(root: Path, experiment: dict, task: dict) -> dict:
    collect = task["kind"] in {"theta_collect", "smoke_collect"}
    if task["calibration_file"]:
        table_path = root / task["calibration_file"]
        put_json(table_path, table_fixture(experiment, task))
        table_hash = identity.sha256_file(table_path)
    else:
        table_hash = ""
    if not collect and task["kind"] != "dense":
        task = contract.make_task(experiment, task["model_key"], task["kind"], task["batch_size"],
                                  theta_index=task["theta_index"],
                                  calibration_sha256=table_hash, host_label=task.get("host_label"))
    put_json(root / "tasks" / (task["run_id"] + ".json"), task)
    log = root / task["log_file"]
    log.parent.mkdir(parents=True, exist_ok=True)
    if collect:
        log.write_text(
            "Comparison task — " + json.dumps({
                "task_sha256": identity.json_sha256(task), "batch_size": task["batch_size"],
            }) + "\n"
            "Calibration progress — " + json.dumps({"completed_batches": 2 * math.ceil(task["calibration_samples"] / task["batch_size"]),
                "total_batches": 2 * math.ceil(task["calibration_samples"] / task["batch_size"]),
                "completed_samples": 2 * task["calibration_samples"], "pass": 2, "elapsed_seconds": 1.0}) + "\n"
            f"Calibration identity — mode: collect, sha256: {table_hash}\n")
    else:
        log.write_text(log_fixture(experiment, task))
    result = {
        **task, "success": True, "task_sha256": identity.json_sha256(task),
        "experiment_sha256": identity.json_sha256(experiment),
        "host_label": task.get("host_label", "local"),
        "elapsed_seconds": 1.0, "log_sha256": identity.sha256_file(log),
        "calibration_sha256": table_hash,
    }
    if collect:
        result["sites"] = len(contract.calibration_site_records(experiment, contract.model_by_key(experiment, task["model_key"])))
        result["samples"] = result["total"] = task["calibration_samples"]
    else:
        result.update(contract.parsed_result(task, root))
    put_json(root / task["result_file"], result)
    return json.loads(json.dumps(result))


def verify_pipeline_outputs(root: Path) -> None:
    experiment = experiment_fixture(root)
    put_json(root / "experiment.json", experiment)
    invoked = []
    def execute(output, exp, task, host):
        invoked.append((task["model_key"], task["kind"]))
        return completed_fixture(output, exp, task)
    with patch.object(runner, "admission", return_value={"batch_size": 32}), \
            patch.object(runner, "run_task", side_effect=execute), patch.object(runner, "event"):
        for key in contract.MODEL_KEYS:
            runner.pipeline(root, experiment, key, "local")
    expected = []
    for key in contract.MODEL_KEYS:
        for _ in contract.theta_grid():
            expected.extend([(key, "theta_collect"), (key, "theta_train")])
        expected.extend([(key, "dense"), (key, "spiking")])
    assert invoked == expected
    # Constructed results intentionally use the worker's exact schema. A reducer
    # requiring fields which the worker does not emit must fail this integration.
    value = runner.summarize(root, experiment, require_complete=True)
    assert value["complete"]
    assert value["calibrations_complete"] == 4 * len(contract.theta_grid())
    assert value["evaluations_complete"] == 4 * (len(contract.theta_grid()) + 2)
    from scripts.analysis.summarize_vit_comparison import verify_publication_bundle
    verify_publication_bundle(root / "outputs")


def verify_worker_integration(root: Path) -> None:
    """Exercise the real worker's persisted schema with a simulated evaluator child."""
    experiment = experiment_fixture(root)
    source = Path(experiment["source_root"])
    source.mkdir()
    # The worker must reject /tmp even in tests, so only its scoped scratch path
    # uses the maintained disk-backed runtime tree.
    disk_root = ROOT / "artifacts/runtime"
    disk_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comparison-worker-test-", dir=disk_root) as runtime:
        experiment["runtime_root"] = runtime
        kinds = [("smoke_collect", None), ("smoke_spiking", None)]
        kinds += [("theta_collect", index) for index in range(len(contract.theta_grid()))]
        kinds += [("theta_train", index) for index in range(len(contract.theta_grid()))]
        kinds += [("dense", 6), ("spiking", 6)]
        for kind, theta_index in kinds:
            table_name = ("smoke/cifar10_vit_small/bs32/calibration.json" if kind.startswith("smoke")
                          else f"calibration/cifar10_vit_small/theta_{theta_index:02d}.json")
            digest = identity.sha256_file(root / table_name) if kind in {"theta_train", "spiking", "smoke_spiking"} else ""
            task = contract.make_task(experiment, "cifar10_vit_small", kind, 32,
                                      theta_index=theta_index, calibration_sha256=digest)
            class Child:
                def __init__(self, command, *, stdout, **kwargs):
                    assert command == contract.evaluator_command(experiment, task, root)
                    assert kwargs["start_new_session"] is True
                    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
                    if kind in {"theta_collect", "smoke_collect"}:
                        path = root / task["calibration_file"]
                        put_json(path, table_fixture(experiment, task))
                        total_batches = 2 * math.ceil(task["calibration_samples"] / task["batch_size"])
                        stdout.write("Calibration progress — " + json.dumps({
                            "completed_batches": total_batches, "total_batches": total_batches,
                            "completed_samples": 2 * task["calibration_samples"], "pass": 2,
                            "elapsed_seconds": 0.01}) + "\n")
                        stdout.write(
                            f"Calibration identity — mode: collect, sha256: {identity.sha256_file(path)}\n"
                        )
                    else:
                        text = "\n".join(line for line in log_fixture(experiment, task).splitlines()
                                         if not line.startswith(("Slurm identity — ", "Comparison task — ")))
                        stdout.write(text + "\n")
                    stdout.flush()
                def wait(self, **_):
                    return 0
                def poll(self):
                    return 0
            with patch.object(runner, "check_source"), patch.object(runner, "check_assets"), \
                    patch.object(runner, "event"), patch.object(runner.subprocess, "Popen", Child), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
                result = runner.run_task(root, experiment, task, "local")
                reloaded = runner.completed(root, experiment, task)
                assert reloaded == result
                assert runner.run_task(root, experiment, task, "local") == result
        value = runner.summarize(root, experiment)
        assert value["calibrations_complete"] == len(contract.theta_grid())
        assert value["evaluations_complete"] == len(contract.theta_grid()) + 2


def verify_contract(root: Path) -> None:
    experiment = experiment_fixture(root)
    contract.validate_experiment(experiment)
    tasks = [contract.make_task(experiment, key, kind, 32, theta_index=6,
                                calibration_sha256="b" * 64 if kind in {"theta_train", "spiking"} else "")
             for key in contract.MODEL_KEYS for kind in ("theta_collect", "theta_train", "dense", "spiking")]
    assert len({task["run_id"] for task in tasks}) == 16
    for task in tasks:
        contract.validate_task(task, experiment)
        for field, value in (("source_commit", "b" * 40), ("checkpoint_sha256", "a" * 64),
                             ("dataset_fingerprint", "other"), ("time_noise_std_frac", 1e-5),
                             ("deadline_margin_std", 4.0), ("gelu_evaluator_sha256", "0" * 64)):
            reject(lambda: contract.validate_task({**task, field: value}, experiment))
        reject(lambda: contract.validate_task({**task, "preprocessing_sha256": "0" * 64}, experiment))
    for batch in (0, 1, 4, 64, 128):
        reject(lambda: contract.make_task(experiment, contract.MODEL_KEYS[0], "theta_collect", batch, theta_index=0))
    for field, value in (("local_gpu_ids", list(range(8))), ("theta_candidates", [40]), ("evaluation_count", 4)):
        reject(lambda: contract.validate_experiment({**experiment, field: value}))
    for field in ("dependency_sha256", "runtime_sha256", "package_versions"):
        reject(lambda: contract.validate_experiment({**experiment, field: {}}))
    for field, value in (("dataset_fingerprint", ""), ("calibration_dataset_fingerprint", ""),
                         ("evaluation_quick_test", True)):
        changed = copy.deepcopy(experiment)
        changed["models"][0][field] = value
        reject(lambda: contract.validate_experiment(changed))


def verify_commands(root: Path) -> None:
    experiment = experiment_fixture(root)
    for key in contract.MODEL_KEYS:
        for kind in ("theta_collect", "theta_train", "dense", "spiking", "smoke_collect", "smoke_spiking"):
            theta_index = 6 if kind in {"theta_collect", "theta_train", "dense", "spiking"} else None
            task = contract.make_task(experiment, key, kind, 16, theta_index=theta_index,
                                      calibration_sha256="b" * 64)
            args = contract.evaluator_command(experiment, task, root)
            at = lambda name: args[args.index(name) + 1]
            assert at("--batch_size") == "16" and float(at("--theta")) == task["theta"] and at("--precision") == "float64"
            assert "--no-tensorboard" in args and "--no-gaussian-time-noise" in args and "--no-mismatch-enabled" in args
            for flag in ("--time-noise-std-frac", "--time-noise-mean", "--time-noise-deadline-margin-std", "--mismatch-theta-std", "--weight-noise-std", "--bias-noise-std"):
                assert float(at(flag)) == 0
            assert ("--quick-test" in args) == (key != "cifar10_vit_small" and task["split"] != "train")
            if kind.startswith("smoke"):
                assert at("--calibration-smoke-samples") == at("--calibration-samples") == "32"
                assert "smoke/" in at("--calibration-path")
                if kind == "smoke_spiking":
                    assert at("--max_eval_batches") == "2"
            else:
                assert "--max_eval_batches" not in args and "--calibration-smoke-samples" not in args


def verify_tables(root: Path) -> None:
    experiment = experiment_fixture(root)
    for key in contract.MODEL_KEYS:
        for kind in ("theta_collect", "smoke_collect"):
            task = contract.make_task(experiment, key, kind, 32,
                                      theta_index=0 if kind == "theta_collect" else None)
            table = table_fixture(experiment, task)
            path = root / task["calibration_file"]
            put_json(path, table)
            contract.validate_table(path, task, experiment)
            for field, value in (("source_commit", "z" * 40), ("gelu_evaluator_sha256", "0" * 64),
                                 ("vit_evaluator_sha256", "0" * 64), ("calibration_wrapper_sha256", "0" * 64),
                                 ("calibration_source_fingerprint", "other"), ("calibration_purpose", "wrong"),
                                 ("attention_implementation", "eager"), ("output_bounds_version", 2)):
                changed = copy.deepcopy(table)
                options = dict(changed["metadata"]["model_options"])
                options[field] = value
                changed["metadata"]["model_options"] = sorted(options.items())
                put_json(path, changed)
                reject(lambda: contract.validate_table(path, task, experiment))
            for mutation in (lambda t: t["layers"].pop(), lambda t: t["layers"].__setitem__(0, t["layers"][1]),
                             lambda t: t["layers"][0].__setitem__("module_name", "another.module"),
                             lambda t: t["layers"][0]["histogram"].__setitem__("bin_counts", [100, 0]),
                             lambda t: t["metadata"].__setitem__("dataset_id", "another-dataset")):
                changed = copy.deepcopy(table)
                mutation(changed)
                put_json(path, changed)
                reject(lambda: contract.validate_table(path, task, experiment))
            put_json(path, table)


def verify_completed_logs(root: Path) -> None:
    experiment = experiment_fixture(root)
    for key in contract.MODEL_KEYS:
        for kind in ("theta_collect", "theta_train", "dense", "spiking", "smoke_collect", "smoke_spiking"):
            theta_index = 6 if kind in {"theta_collect", "theta_train", "dense", "spiking"} else None
            result = completed_fixture(root, experiment, contract.make_task(
                experiment, key, kind, 32, theta_index=theta_index))
            task = runner.read_json(root / "tasks" / (result["run_id"] + ".json"))
            contract.validate_result(task, result, experiment, root)
            assert runner.completed(root, experiment, task) == result
            for field, value in (("success", False), ("source_commit", "b" * 40), ("task_sha256", "f" * 64),
                                 ("experiment_sha256", "f" * 64), ("log_sha256", "f" * 64)):
                reject(lambda: contract.validate_result(task, {**result, field: value}, experiment, root))
            if kind not in {"theta_collect", "smoke_collect"}:
                log = root / task["log_file"]
                original = log.read_text()
                for changed in (original.replace("Prediction SHA256:", "Partial prediction:"), original + "Correct: 0\n",
                                original.replace("Comparison task — ", "Wrong task — "), original + "Traceback (most recent call last)\n"):
                    log.write_text(changed)
                    reject(lambda: contract.parsed_result(task, root))
                log.write_text(original)


def verify_admission(root: Path) -> None:
    experiment, key = experiment_fixture(root), "imagenet_vit_base"
    calls = []
    def execute(output, exp, task, host):
        calls.append((task["kind"], task["batch_size"]))
        if task["batch_size"] == 32:
            path = output / task["log_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
            raise RuntimeError("evaluator exited")
        return completed_fixture(output, exp, task)
    with patch.object(runner, "run_task", side_effect=execute), patch.object(runner, "event"):
        selected = runner.admission(root, experiment, key, "local")
        assert selected["batch_size"] == 16
        assert calls == [("smoke_collect", 32), ("smoke_collect", 16), ("smoke_spiking", 16)]
        assert runner.admission(root, experiment, key, "local") == selected
        admission_path = root / "admissions" / (key + ".json")
        for field, value in (("result_sha256", {}), ("batch_size", 8), ("model_key", "imagenet_vit_small"),
                             ("calibration_sha256", "0" * 64), ("prediction_sha256", "0" * 64)):
            put_json(admission_path, {**selected, field: value})
            reject(lambda: runner.admission(root, experiment, key, "local"))
        put_json(admission_path, selected)
    bad_root = root / "technical_failure"
    bad_root.mkdir()
    with patch.object(runner, "run_task", side_effect=RuntimeError("bad input")) as call:
        reject(lambda: runner.admission(bad_root, experiment, key, "local"))
        assert call.call_count == 1
    oom_root = root / "all_oom"
    oom_root.mkdir()
    def oom(output, exp, task, host):
        path = output / task["log_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CUDA out of memory\n")
        raise RuntimeError("evaluator exited")
    with patch.object(runner, "run_task", side_effect=oom) as call, patch.object(runner, "event"):
        reject(lambda: runner.admission(oom_root, experiment, key, "local"))
        assert [item.args[2]["batch_size"] for item in call.call_args_list] == [32, 16, 8]


def verify_source_and_gpu(root: Path) -> None:
    experiment = experiment_fixture(root)
    with patch("subprocess.check_output", side_effect=["b" * 40 + "\n", ""]):
        reject(lambda: contract.check_source(experiment))
    with patch("subprocess.check_output", side_effect=[experiment["source_commit"] + "\n", " M changed.py\n"]):
        reject(lambda: contract.check_source(experiment))
    for prefix in ("evaluator", "calibration_evaluator", "gelu_evaluator"):
        path = Path(experiment["source_root"]) / experiment[prefix + "_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + prefix)
        experiment[prefix + "_sha256"] = identity.sha256_file(path)
    for name in experiment["runtime_sha256"]:
        path = Path(experiment["source_root"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture runtime " + name)
        experiment["runtime_sha256"][name] = identity.sha256_file(path)
    def dependency_identity(path):
        return experiment["dependency_sha256"]["transformers" if path.name == "src" else "spikingjelly"], []
    with patch("subprocess.check_output", side_effect=lambda args, **kwargs: experiment["source_commit"] + "\n" if args[-1] == "HEAD" else ""), \
            patch("scripts.runtime.identity.package_source_identity", side_effect=dependency_identity):
        contract.check_source(experiment)
        for prefix in ("evaluator", "calibration_evaluator", "gelu_evaluator"):
            reject(lambda: contract.check_source({**experiment, prefix + "_sha256": "0" * 64}))
        changed = copy.deepcopy(experiment)
        changed["runtime_sha256"][runner.SCRIPT] = "0" * 64
        reject(lambda: contract.check_source(changed))
        changed = copy.deepcopy(experiment)
        changed["dependency_sha256"]["transformers"] = "0" * 64
        reject(lambda: contract.check_source(changed))
    good = json.dumps({"count": 1, "model": "NVIDIA RTX A6000"})
    for gpu in range(4, 8):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}), patch("subprocess.check_output", return_value=good):
            assert contract.require_gpu(experiment, "local") == "NVIDIA RTX A6000"
    for gpu in ("", "0", "1", "2", "3", "4,5", "8", "-1"):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": gpu}), patch("subprocess.check_output") as query:
            reject(lambda: contract.require_gpu(experiment, "local"))
            query.assert_not_called()
    without_override = {**experiment, "campaign_extra_local_gpus": []}
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), patch("subprocess.check_output") as query:
        reject(lambda: contract.require_gpu(without_override, "local"))
        query.assert_not_called()
    assert not local_gpu.gpu_available({"memory_used_mib": 20000, "utilization_gpu_percent": 0, "pids": []})
    assert not local_gpu.gpu_available({"memory_used_mib": 0, "utilization_gpu_percent": 100, "pids": []})
    assert local_gpu.gpu_available({"memory_used_mib": 0, "utilization_gpu_percent": 0, "pids": [12345]})
    for changed in ({"count": 2, "model": "NVIDIA RTX A6000"}, {"count": 1, "model": "NVIDIA RTX 3090"}):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}), patch("subprocess.check_output", return_value=json.dumps(changed)):
            reject(lambda: contract.require_gpu(experiment, "local"))


def verify_policy2_and_preparation(root: Path) -> None:
    """Keep archived results readable while preparation never launches inference."""
    experiment = experiment_fixture(root)
    contract.validate_experiment(experiment)
    for model in experiment["models"]:
        expected = 217 if model["model_key"].endswith("large") else 109
        assert len(contract.calibration_site_records(experiment, model)) == expected
        broken = copy.deepcopy(experiment)
        contract.model_by_key(broken, model["model_key"])["calibration_sites"].pop()
        reject(lambda: contract.validate_experiment(broken))
    for value in (None, 1, 3, True):
        reject(lambda value=value: contract.validate_experiment({**experiment, "vit_calibration_policy_version": value}))
    path = root / "smoke/imagenet_vit_small/bs32/calibration.json"
    new_task = contract.make_task(experiment, "imagenet_vit_small", "smoke_collect", 32)
    table = table_fixture(experiment, new_task)
    for key, value in (("vit_calibration_policy_version", 1), ("layer_norm_eps", 1e-5),
                       ("layer_norm_clip_margin", 1e-6)):
        changed = copy.deepcopy(table)
        changed["metadata"]["model_options"] = sorted({**dict(changed["metadata"]["model_options"]), key: value}.items())
        put_json(path, changed)
        reject(lambda: contract.validate_table(path, new_task, experiment))
    with patch.object(runner, "check_source"), patch.object(runner, "event"), \
            patch.object(runner, "run_task") as execute, patch.object(runner, "admission") as admit:
        value = runner.prepare_execution(root, experiment)
        execute.assert_not_called()
        admit.assert_not_called()
        assert value["launch_performed"] is False and len(value["models"]) == 4
        for row in value["models"]:
            assert set(row["commands_by_gpu"]) == {str(gpu) for gpu in range(4, 8)}
            for gpu, command in row["commands_by_gpu"].items():
                assert command == ["env", "CUDA_VISIBLE_DEVICES=" + gpu, *row["command"]]
        assert all(row["batch_size"] is None and row["requires_full_calibration"] for row in value["models"])
        assert len(list((root / "prepared").glob("*.json"))) == 1
        assert runner.prepare_execution(root, experiment) == value
    with patch.object(runner, "admission", return_value={"batch_size": 32}), \
            patch.object(runner, "pipeline") as launch, patch.object(runner, "check_source"), \
            patch.object(runner.fcntl, "flock"), \
            patch.object(runner, "require_gpu"), patch.object(runner.local_gpu, "gpu_activity", return_value={4: {}}), \
            patch.object(runner.local_gpu, "gpu_available", return_value=True), \
            patch.object(runner, "package_versions", return_value=experiment["package_versions"]), \
            patch.object(runner.platform, "python_version", return_value="3.12.13"), \
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
        put_json(root / "experiment.json", experiment)
        runner.worker(root, "admit", "imagenet_vit_small", "local")
        launch.assert_not_called()


def main() -> None:
    checks = (verify_contract, verify_commands, verify_tables, verify_completed_logs,
              verify_admission, verify_source_and_gpu, verify_pipeline_outputs, verify_worker_integration,
              verify_policy2_and_preparation)
    failures = []
    with tempfile.TemporaryDirectory(prefix="vit-comparison-runner-tests-") as temporary:
        for check in checks:
            root = Path(temporary) / check.__name__
            root.mkdir()
            try:
                check(root)
                print(f"PASS {check.__name__}")
            except Exception as error:
                failures.append((check.__name__, repr(error)))
                print(f"FAIL {check.__name__}: {error!r}")
    if failures:
        raise AssertionError(failures)
    print(f"Comparison runner verification passed ({len(checks)} groups).")


if __name__ == "__main__":
    main()
