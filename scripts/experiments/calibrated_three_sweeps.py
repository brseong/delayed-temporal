"""Immutable conditions and validation for three calibrated ViT sweeps."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

TAG = "vit_base_calibrated_theta_rt_ratio_float64_bounds3_v1"
RATIOS = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
KINDS = {"collect", "theta_train", "theta_validation", "theta_replay", "dense",
         "noise", "smoke_clean", "smoke_noise"}


def theta_grid() -> tuple[float, ...]:
    return tuple(10.0 * 2.0 ** (i / 2.0) for i in range(9))


def rt_grid() -> tuple[float, ...]:
    return tuple(1e-5 * 10.0 ** (i / 8.0) for i in range(9))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable identity changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError(f"Immutable identity changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def safe_output(root: Path, relative: str) -> Path:
    path = Path(relative)
    target = (root / path).resolve()
    if path.is_absolute() or not target.is_relative_to(root.resolve()) or ".." in path.parts:
        raise ValueError("Task output must remain inside its experiment directory")
    return target


def make_task(experiment: dict[str, Any], kind: str, *, theta_index: int = 4,
              seed: int | None = None, rt: float = 0.0, ratio: float = 0.0,
              calibration_sha256: str = "", expected_samples: int = 5000,
              host_label: str | None = None) -> dict[str, Any]:
    if kind not in KINDS or type(theta_index) is not int or not 0 <= theta_index < 9:
        raise ValueError("Unknown task kind or theta candidate")
    theta = theta_grid()[theta_index]
    noisy = kind in {"noise", "smoke_noise"}
    if noisy:
        if seed not in (0, 1, 2) or rt not in rt_grid() or ratio not in RATIOS:
            raise ValueError("Noise condition is outside the approved grid")
        if ratio != 4.0 and rt != rt_grid()[0]:
            raise ValueError("Only the two approved one-dimensional noise sweeps are allowed")
        suffix = f"rt_{rt_grid().index(rt):02d}" if ratio == 4.0 else f"ratio_{RATIOS.index(ratio):02d}"
        run_id = f"{kind}_{suffix}_seed{seed}"
    else:
        if seed is not None or rt != 0.0 or ratio != 0.0:
            raise ValueError("Deterministic tasks cannot contain stochastic settings")
        run_id = {"collect": f"theta_{theta_index:02d}_collect",
                  "theta_train": f"theta_{theta_index:02d}_train",
                  "theta_validation": f"theta_{theta_index:02d}_validation",
                  "theta_replay": f"theta_{theta_index:02d}_replay",
                  "dense": "dense_reference", "smoke_clean": "smoke_clean"}[kind]
    if kind.startswith("smoke"):
        if expected_samples != 160:
            raise ValueError("Environment checks require five complete batches")
        if host_label not in {"local", "ubai"}:
            raise ValueError("Environment checks require local or ubai assignment")
        run_id += f"_{host_label}"
    elif host_label is not None:
        raise ValueError("Fixed environment labels are only used for environment checks")
    elif expected_samples != 5000:
        raise ValueError("All scientific evaluations require exactly 5000 samples")
    training = kind in {"collect", "theta_train", "theta_replay"}
    sigma = 2.0 * theta * rt
    task = {
        "run_id": run_id, "kind": kind, "theta": theta, "theta_index": theta_index,
        "seed": seed, "backend": "hf" if kind == "dense" else "spiking",
        "split": "train" if training else "validation", "expected_samples": expected_samples,
        "time_noise_std_frac": rt, "time_noise_std_abs": sigma,
        "deadline_margin_std": ratio, "deadline_margin_abs": ratio * sigma,
        "calibration_file": "" if kind == "dense" else f"calibration/theta_{theta_index:02d}.json",
        "calibration_sha256": calibration_sha256,
        "log_file": f"logs/{run_id}.log", "result_file": f"results/{run_id}.json",
        "dataset_fingerprint": experiment["calibration_dataset_fingerprint" if training else "dataset_fingerprint"],
        "precision": "float64", "gpu_family": "rtxa6000",
    }
    for key in ("source_commit", "checkpoint_sha256", "evaluator_sha256", "calibration_evaluator_sha256"):
        task[key] = experiment[key]
    if host_label is not None:
        task["host_label"] = host_label
    return task


def make_tasks(experiment: dict[str, Any], phase: str, *, selected_theta: float | None = None,
               seed: int | None = None, calibration_sha256: str = "",
               host_label: str | None = None) -> list[dict[str, Any]]:
    if phase == "theta":
        return [make_task(experiment, kind, theta_index=i) for i in range(9)
                for kind in ("collect", "theta_train", "theta_validation")] + [make_task(experiment, "dense")]
    index = theta_grid().index(40.0 if selected_theta is None else selected_theta)
    if phase == "noise":
        conditions = [(rt, 4.0) for rt in rt_grid()] + [(rt_grid()[0], k) for k in RATIOS if k != 4.0]
        return [make_task(experiment, "noise", theta_index=index, seed=seed, rt=rt, ratio=k,
                          calibration_sha256=calibration_sha256) for rt, k in conditions]
    if phase in {"replay", "smoke_clean", "smoke_noise"}:
        kind = "theta_replay" if phase == "replay" else phase
        return [make_task(experiment, kind, theta_index=index,
                          seed=0 if kind == "smoke_noise" else None,
                          rt=rt_grid()[0] if kind == "smoke_noise" else 0.0,
                          ratio=4.0 if kind == "smoke_noise" else 0.0,
                          expected_samples=5000 if kind == "theta_replay" else 160,
                          calibration_sha256=calibration_sha256, host_label=host_label)]
    raise ValueError(f"Unknown experiment phase: {phase}")


def validate_task(task: dict[str, Any], experiment: dict[str, Any]) -> None:
    expected = make_task(experiment, task["kind"], theta_index=task["theta_index"],
                         seed=task["seed"], rt=task["time_noise_std_frac"],
                         ratio=task["deadline_margin_std"], calibration_sha256=task["calibration_sha256"],
                         expected_samples=task["expected_samples"], host_label=task.get("host_label"))
    for key, value in expected.items():
        if task.get(key) != value:
            raise ValueError(f"Task identity mismatch: {key}")
    if task["kind"] not in {"collect", "dense"} and not re.fullmatch(r"[0-9a-f]{64}", task["calibration_sha256"]):
        raise ValueError("A frozen calibration table hash is required before assigning evaluation")


def validate_experiment(experiment: dict[str, Any]) -> None:
    for key, value in {"tag": TAG, "output_bounds_version": 3, "precision": "float64",
                       "batch_size": 32, "seeds": [0, 1, 2]}.items():
        if experiment.get(key) != value:
            raise ValueError(f"Experiment setting mismatch: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", experiment.get("source_commit", "")):
        raise ValueError("A complete frozen source commit is required")
    for name in ("checkpoint_sha256", "calibration_dataset_sha256", "dataset_sha256",
                 "evaluator_sha256", "calibration_evaluator_sha256", "gelu_evaluator_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", experiment.get(name, "")):
            raise ValueError(f"Missing experiment content hash: {name}")
    for name in ("source_root", "python_bin", "checkpoint_path", "calibration_dataset_path", "dataset_path"):
        if not Path(experiment[name]).is_absolute():
            raise ValueError(f"Experiment path must be absolute: {name}")
    for name in ("dataset_fingerprint", "calibration_dataset_fingerprint"):
        if not experiment.get(name):
            raise ValueError(f"Dataset identity is required: {name}")


def validate_table(path: Path, task: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    table = json.loads(path.read_text())
    layers = table["layers"]
    if len(layers) != 48 or len({(layer["module_name"], layer["tensor_name"]) for layer in layers}) != 48:
        raise ValueError("Calibration requires 48 unique sites")
    metadata = table["metadata"]
    preprocessing = json.loads(metadata["preprocessing"])
    for key, expected in {"subset_samples": 5000, "subset_seed": 0,
                          "subset_fingerprint": experiment["calibration_dataset_fingerprint"]}.items():
        if preprocessing.get(key) != expected:
            raise ValueError(f"Calibration training subset identity mismatch: {key}")
    for name, expected in {"theta": task["theta"], "dtype": "float64", "dataset_split": "train",
                           "clip_margin": 1e-5, "tau_s": 1.0, "tau_m": 1.0,
                           "model_id": experiment["checkpoint_path"]}.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Calibration metadata mismatch: {name}")
    options = dict(metadata["model_options"])
    required = {"output_bounds_version": 3, "source_commit": experiment["source_commit"],
                "checkpoint_sha256": experiment["checkpoint_sha256"],
                "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
                "gelu_evaluator_sha256": experiment["gelu_evaluator_sha256"],
                "calibration_dataset_fingerprint": experiment["calibration_dataset_fingerprint"],
                "spiking_ln_mul": True, "spiking_ln_log": True, "spiking_ln_expdiff": True,
                "use_spiking_layernorm": True, "use_spiking_mlp": True}
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("Calibration source, bounds policy, or operator configuration mismatch")
    for layer in layers:
        bounds = layer["bounds"]
        if not all(math.isfinite(bounds[key]) for key in ("min", "max")) or bounds["min"] >= bounds["max"]:
            raise ValueError("Calibration bounds must be finite and ordered")
        if (layer["lower_quantile"], layer["upper_quantile"], layer["margin_fraction"]) != (0.0, 1.0, 0.05):
            raise ValueError("Calibration must use min/max with five percent range expansion")
    return table


def parse_result_log(task: dict[str, Any], root: Path) -> dict[str, Any]:
    from scripts.analysis.summarize_sigma_margin_sweep import ManifestRun, parse_run_log
    noisy = task["kind"] in {"noise", "smoke_noise"}
    keys = ("run_id", "backend", "theta", "time_noise_std_frac", "time_noise_std_abs",
            "deadline_margin_std", "deadline_margin_abs", "seed", "split", "expected_samples",
            "dataset_fingerprint", "precision", "source_commit", "checkpoint_sha256", "gpu_family", "log_file")
    spec = ManifestRun(**{key: task[key] for key in keys},
                       stage="sigma_margin" if noisy else "baseline", row={})
    parsed = asdict(parse_run_log(spec, root))
    text = safe_output(root, task["log_file"]).read_text()
    if task["backend"] == "spiking":
        for line in ("GELU cubic implementation: phi_nl_psi_ed", "GELU cubic magnitude floor: 1e-05",
                     f"Calibration identity — mode: validate, sha256: {task['calibration_sha256']}"):
            if text.splitlines().count(line) != 1:
                raise ValueError(f"Missing or duplicate evaluator identity: {line}")
    clamp_pattern = re.compile(r"^Clamp\[(?P<site>[^]]+)] values=(?P<values>\d+), underflows=(?P<underflows>\d+) .*?overflows=(?P<overflows>\d+)", re.MULTILINE)
    parsed["clamp_sites"] = [{key: match.group(key) if key == "site" else int(match.group(key))
                              for key in ("site", "values", "underflows", "overflows")}
                             for match in clamp_pattern.finditer(text)]
    if len({site["site"] for site in parsed["clamp_sites"]}) != len(parsed["clamp_sites"]):
        raise ValueError("Duplicate clamp counts")
    for site in parsed["clamp_sites"]:
        if site["underflows"] + site["overflows"] > site["values"]:
            raise ValueError("Invalid clamp counts")
    return parsed


def validate_result(task: dict[str, Any], result: dict[str, Any], experiment: dict[str, Any],
                    output_root: Path | None = None) -> None:
    validate_task(task, experiment)
    for key in ("run_id", "kind", "theta", "theta_index", "seed", "backend", "split",
                "source_commit", "checkpoint_sha256", "dataset_fingerprint", "precision",
                "time_noise_std_frac", "time_noise_std_abs", "deadline_margin_std", "deadline_margin_abs",
                "evaluator_sha256", "calibration_evaluator_sha256"):
        if result.get(key) != task[key]:
            raise ValueError(f"Result identity mismatch: {key}")
    if result.get("success") is not True or result.get("task_sha256") != task_sha256(task):
        raise ValueError("Incomplete result or task hash mismatch")
    if result.get("experiment_sha256") != task_sha256(experiment):
        raise ValueError("Result experiment identity mismatch")
    if result.get("host_label") not in {"local", "ubai"}:
        raise ValueError("Result execution environment is required")
    if task.get("host_label", result["host_label"]) != result["host_label"]:
        raise ValueError("Result execution environment differs from its assignment")
    if not math.isfinite(result.get("elapsed_seconds", -1)) or result["elapsed_seconds"] < 0:
        raise ValueError("Invalid elapsed time")
    if task["kind"] == "collect":
        if result.get("sites") != 48:
            raise ValueError("Incomplete calibration result")
    else:
        if result.get("samples") != task["expected_samples"] or type(result.get("correct")) is not int:
            raise ValueError("Incomplete sample counts")
        if not 0 <= result["correct"] <= result["samples"]:
            raise ValueError("Correct count is outside valid range")
        if not math.isfinite(result.get("accuracy", math.nan)) or result["accuracy"] != result["correct"] / result["samples"]:
            raise ValueError("Accuracy does not match correct count")
        if not re.fullmatch(r"[0-9a-f]{64}", result.get("prediction_sha256", "")):
            raise ValueError("Prediction digest is required")
        if task["kind"] != "dense" and result.get("calibration_sha256") != task["calibration_sha256"]:
            raise ValueError("Result calibration identity mismatch")
    if output_root is not None:
        log = safe_output(output_root, task["log_file"])
        if sha256_file(log) != result.get("log_sha256"):
            raise ValueError("Completed log identity changed")
        if task["kind"] != "dense":
            table = safe_output(output_root, task["calibration_file"])
            if sha256_file(table) != result.get("calibration_sha256"):
                raise ValueError("Frozen calibration table changed")
            validate_table(table, task, experiment)
        if task["kind"] == "collect":
            marker = f"Calibration identity — mode: collect, sha256: {result['calibration_sha256']}"
            text = log.read_text()
            if text.splitlines().count(marker) != 1 or "Traceback (most recent call last)" in text:
                raise ValueError("Collection log is incomplete")
        else:
            parsed = parse_result_log(task, output_root)
            # Absolute log paths differ after transfer; all scientific fields must match.
            for key, value in parsed.items():
                if key not in {"log_file", "sites"} and result.get(key) != value:
                    raise ValueError(f"Result differs from its completed log: {key}")
            if list(result.get("sites", [])) != list(parsed["sites"]):
                raise ValueError("Result site counts differ from its completed log")


class RangeInsufficient(ValueError):
    """The predeclared theta candidates do not establish a usable selection."""


def _theta_results(results: list[dict[str, Any]], kind: str) -> dict[int, dict[str, Any]]:
    rows = [r for r in results if r["kind"] == kind]
    indexed = {row["theta_index"]: row for row in rows}
    if len(rows) != 9 or set(indexed) != set(range(9)):
        raise ValueError("Exactly nine complete theta candidate results are required")
    identity_fields = ("source_commit", "checkpoint_sha256", "dataset_fingerprint", "precision",
                       "evaluator_sha256", "calibration_evaluator_sha256")
    if any(len({row.get(key) for row in rows}) != 1 or not rows[0].get(key) for key in identity_fields):
        raise ValueError("Theta candidate result identities differ")
    for index, row in indexed.items():
        if row.get("success") is not True or row["samples"] != 5000 or row["theta"] != theta_grid()[index]:
            raise ValueError("Invalid theta candidate result")
        if type(row["correct"]) is not int or not 0 <= row["correct"] <= 5000:
            raise ValueError("Invalid theta correct count")
        if row.get("accuracy") != row["correct"] / 5000 or not re.fullmatch(r"[0-9a-f]{64}", row.get("prediction_sha256", "")):
            raise ValueError("Invalid theta accuracy or prediction digest")
    return indexed


def select_theta(training_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _theta_results(training_results, "theta_train")
    highest = max(row["correct"] for row in rows.values())
    index = min(i for i, row in rows.items() if row["correct"] >= highest - 25)
    if index == 0:
        raise RangeInsufficient("Smallest theta candidate is selected; lower range must be reconsidered")
    if rows[8]["correct"] - rows[7]["correct"] > 5:
        raise RangeInsufficient("Largest theta candidate is still improving by more than 0.1 percentage points")
    return {"status": "selected", "theta": theta_grid()[index], "theta_index": index,
            "training_correct": rows[index]["correct"], "training_samples": 5000,
            "maximum_training_correct": highest, "training_result_sha256": task_sha256(rows[index]),
            "source_commit": rows[index]["source_commit"],
            "checkpoint_sha256": rows[index]["checkpoint_sha256"],
            "calibration_sha256": rows[index]["calibration_sha256"]}


def confirm_selection(selection: dict[str, Any], validation_results: list[dict[str, Any]],
                      replay_result: dict[str, Any], training_results: list[dict[str, Any]]) -> dict[str, Any]:
    expected = select_theta(training_results)
    if selection != expected:
        raise ValueError("Selection does not match complete training results")
    validation = _theta_results(validation_results, "theta_validation")
    training = _theta_results(training_results, "theta_train")[selection["theta_index"]]
    index = selection["theta_index"]
    for key in ("theta", "theta_index", "correct", "samples", "prediction_sha256", "source_commit",
                "checkpoint_sha256", "calibration_sha256", "dataset_fingerprint", "evaluator_sha256",
                "calibration_evaluator_sha256", "precision"):
        if replay_result.get(key) != training[key]:
            raise ValueError(f"Opposite-environment replay differs: {key}")
    if replay_result.get("kind") != "theta_replay" or replay_result.get("success") is not True:
        raise ValueError("A complete selected-theta replay is required")
    if {replay_result.get("host_label"), training.get("host_label")} != {"local", "ubai"}:
        raise ValueError("Replay must use the opposite execution environment")
    neighbors = [i for i in (index - 1, index, index + 1) if i in validation]
    if validation[index]["correct"] < max(validation[i]["correct"] for i in neighbors) - 25:
        raise ValueError("Validation selection stability failed")
    return {**selection, "status": "confirmed", "validation_correct": validation[index]["correct"],
            "validation_samples": 5000, "validation_neighbor_indices": neighbors,
            "replay_result_sha256": task_sha256(replay_result),
            "validation_result_sha256": task_sha256(validation[index]), "full_validation": False}
