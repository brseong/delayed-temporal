"""Verify calibrated sweep conditions and result identities without evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.calibrated_three_sweeps import (
    TAG, RATIOS, confirm_selection, make_task, make_tasks, parse_result_log,
    rt_grid, select_theta, theta_grid, validate_experiment,
    validate_result, validate_table, validate_task,
)
from scripts.experiments.run_calibrated_three_sweep_task import check_source, evaluator_command, require_gpu
from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import identity


def must_reject(callback) -> None:
    try:
        callback()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError("Invalid condition was accepted")


def experiment_fixture() -> dict:
    return {"tag": TAG, "output_bounds_version": 3, "precision": "float64", "batch_size": 32,
            "seeds": [0, 1, 2], "source_root": str(ROOT), "python_bin": "/opt/conda/envs/dt/bin/python",
            "source_commit": "a" * 40, "checkpoint_path": "/fixture/checkpoint", "checkpoint_sha256": "b" * 64,
            "calibration_dataset_path": "/fixture/train", "calibration_dataset_fingerprint": "train-fingerprint",
            "calibration_dataset_sha256": "c" * 64, "dataset_path": "/fixture/validation",
            "dataset_fingerprint": "validation-fingerprint", "dataset_sha256": "d" * 64,
            "evaluator_path": "scripts/evaluation/error_analysis_vit.py", "evaluator_sha256": "e" * 64,
            "calibration_evaluator_path": "scripts/analysis/evaluate_calibrated_vit.py",
            "calibration_evaluator_sha256": "f" * 64, "gelu_evaluator_sha256": "1" * 64,
            "gelu_evaluator_path": "scripts/analysis/gelu_cubic_phi_nl_vit.py"}


def table_fixture(experiment: dict, theta: float = 40.0) -> dict:
    return {"layers": [{"module_name": f"layer.{i // 4}", "tensor_name": f"site.{i % 4}",
                         "bounds": {"min": -1.0, "max": 1.0}, "lower_quantile": 0.0,
                         "upper_quantile": 1.0, "margin_fraction": 0.05} for i in range(48)],
            "metadata": {"theta": theta, "dtype": "float64", "dataset_split": "train",
                         "clip_margin": 1e-5, "tau_s": 1.0, "tau_m": 1.0,
                         "preprocessing": json.dumps({"subset_samples": 5000, "subset_seed": 0,
                                                      "subset_fingerprint": experiment["calibration_dataset_fingerprint"]}),
                         "model_id": experiment["checkpoint_path"], "model_options": list({
                             "output_bounds_version": 3, "source_commit": experiment["source_commit"],
                             "checkpoint_sha256": experiment["checkpoint_sha256"],
                             "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
                             "gelu_evaluator_sha256": experiment["gelu_evaluator_sha256"],
                             "calibration_dataset_fingerprint": experiment["calibration_dataset_fingerprint"],
                             "spiking_ln_mul": True, "spiking_ln_log": True, "spiking_ln_expdiff": True,
                             "use_spiking_layernorm": True, "use_spiking_mlp": True}.items())}}


def result_fixture(experiment: dict, task: dict, *, correct: int = 4250, host: str = "local") -> dict:
    return {**task, "success": True, "task_sha256": identity.json_sha256(task),
            "experiment_sha256": identity.json_sha256(experiment), "host_label": host,
            "elapsed_seconds": 1.0, "correct": correct, "samples": task["expected_samples"],
            "accuracy": correct / task["expected_samples"], "prediction_sha256": "9" * 64}


def verify_conditions(experiment: dict) -> None:
    validate_experiment(experiment)
    assert theta_grid()[0] == 10 and theta_grid()[4] == 40 and theta_grid()[-1] == 160
    assert rt_grid()[0] == 1e-5 and rt_grid()[-1] == 1e-4 and len(RATIOS) == 9
    theta_tasks = make_tasks(experiment, "theta")
    assert len(theta_tasks) == 28 and sum(t["kind"] == "collect" for t in theta_tasks) == 9
    noise = [t for seed in range(3) for t in make_tasks(experiment, "noise", selected_theta=40.0,
                                                      seed=seed, calibration_sha256="2" * 64)]
    assert len(noise) == len({t["run_id"] for t in noise}) == 51
    assert len(theta_tasks) + len(noise) + 1 - 9 == 71
    for task in noise:
        validate_task(task, experiment)
        assert task["time_noise_std_abs"] == 80 * task["time_noise_std_frac"]
        assert task["deadline_margin_abs"] == task["time_noise_std_abs"] * task["deadline_margin_std"]
    assert sum(t["time_noise_std_frac"] == 1e-5 and t["deadline_margin_std"] == 4 for t in noise) == 3
    for field, value in (("theta", 41), ("seed", 9), ("source_commit", "0" * 40),
                         ("dataset_fingerprint", "other"), ("time_noise_std_abs", 1.0),
                         ("checkpoint_sha256", "0" * 64), ("calibration_sha256", "")):
        must_reject(lambda field=field, value=value: validate_task({**noise[0], field: value}, experiment))
    for version in (2, 4):
        must_reject(lambda version=version: validate_experiment({**experiment, "output_bounds_version": version}))
    for host in ("local", "ubai"):
        smoke = make_tasks(experiment, "smoke_noise", host_label=host, calibration_sha256="2" * 64)[0]
        validate_task(smoke, experiment)
        assert smoke["run_id"].endswith(host) and smoke["expected_samples"] == 160
    train = make_task(experiment, "theta_train", calibration_sha256="2" * 64)
    validate = make_task(experiment, "theta_validation", calibration_sha256="2" * 64)
    assert "--quick-test" not in evaluator_command(experiment, train, Path("/fixture/output"))
    command = evaluator_command(experiment, validate, Path("/fixture/output"))
    assert "--quick-test" in command and "--no-tensorboard" in command
    assert all(flag in command for flag in ("--spiking-ln-mul", "--spiking-ln-log", "--spiking-ln-expdiff"))


def verify_selection(experiment: dict) -> None:
    scores = (4000, 4100, 4200, 4225, 4250, 4251, 4250, 4252, 4253)
    train = [result_fixture(experiment, make_task(experiment, "theta_train", theta_index=i,
                                                calibration_sha256="2" * 64), correct=value)
             for i, value in enumerate(scores)]
    selected = select_theta(train)
    assert selected["theta_index"] == 4
    train[3].update(correct=4228, accuracy=4228 / 5000)
    assert select_theta(train)["theta_index"] == 3  # Inclusive 0.5 percentage-point boundary.
    train[3].update(correct=4227, accuracy=4227 / 5000)
    assert select_theta(train)["theta_index"] == 4
    for bad in (train[:-1], train + [train[0]]):
        must_reject(lambda bad=bad: select_theta(bad))
    lower = copy.deepcopy(train)
    lower[0].update(correct=4253, accuracy=4253 / 5000)
    must_reject(lambda: select_theta(lower))
    upper = copy.deepcopy(train)
    upper[8].update(correct=4258, accuracy=4258 / 5000)
    must_reject(lambda: select_theta(upper))
    # Exactly five additional correct predictions are allowed; six are not.
    upper[8].update(correct=4257, accuracy=4257 / 5000)
    select_theta(upper)
    selected = select_theta(train)
    validation = [result_fixture(experiment, make_task(experiment, "theta_validation", theta_index=i,
                                                       calibration_sha256="2" * 64), correct=4250)
                  for i in range(9)]
    replay = result_fixture(experiment, make_task(experiment, "theta_replay", theta_index=4,
                                                 calibration_sha256="2" * 64), correct=train[4]["correct"], host="ubai")
    confirmed = confirm_selection(selected, validation, replay, train)
    assert confirmed["status"] == "confirmed" and confirmed["full_validation"] is False
    for field, value in (("host_label", "local"), ("prediction_sha256", "8" * 64), ("correct", 0)):
        must_reject(lambda field=field, value=value: confirm_selection(selected, validation, {**replay, field: value}, train))
    validation[4].update(correct=4224, accuracy=4224 / 5000)
    must_reject(lambda: confirm_selection(selected, validation, replay, train))
    validation[4].update(correct=4225, accuracy=4225 / 5000)
    confirm_selection(selected, validation, replay, train)


def write_log(path: Path, task: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noisy = task["kind"] == "noise"
    sigma = task["time_noise_std_abs"]
    path.write_text(
        "Slurm identity — job: fixture, gpu_family: rtxa6000\nGPU model: NVIDIA RTX A6000\n"
        f"Artifact identity — source_commit: {task['source_commit']}, checkpoint_sha256: {task['checkpoint_sha256']}\n"
        f"Gaussian time noise — enabled: {noisy}, std_frac: {task['time_noise_std_frac']}, identity_window: {2*task['theta']}, "
        f"std_abs: {sigma}, mean_abs: 0.0, seed: {task['seed'] or 0}, identity_deadline_ulp: 1e-12, "
        f"std_to_identity_ulp: {sigma/1e-12}, deadline_margin_std: {task['deadline_margin_std']}, deadline_margin_abs: {task['deadline_margin_abs']}\n"
        "Static threshold mismatch — enabled: False, theta_std: 0.0, seed: 0\n"
        f"Evaluation metadata — model: fixture, dataset: imagenet-1k, split: {task['split']}, samples: 5000, "
        f"theta: {task['theta']}, precision: float64, source: disk:/fixture, fingerprint: {task['dataset_fingerprint']}\n"
        f"Correct: 4250\nEvaluated samples: 5000\nPrediction SHA256: {'9'*64}\nAccuracy: 0.85\n"
        "GELU cubic implementation: phi_nl_psi_ed\nGELU cubic magnitude floor: 1e-05\n"
        f"Calibration identity — mode: validate, sha256: {task['calibration_sha256']}\n"
        "Gaussian[layernorm.log] events=100, misses=10 (rate=0.1), deadline_events=20 (rate=0.2), "
        "deadline_ulp_min=1e-12, deadline_ulp_max=2e-12, std_to_ulp_min=10, std_to_ulp_max=20, "
        "outputs=50, underflows=1 (rate=0.02), overflows=2 (rate=0.04)\n"
        "Clamp[layernorm/test] values=100, underflows=1 (rate=0.01), overflows=2 (rate=0.02)\n")


def verify_logs(experiment: dict, root: Path) -> None:
    table = root / "calibration/theta_04.json"
    table.parent.mkdir()
    table.write_text(json.dumps(table_fixture(experiment)))
    task = make_tasks(
        experiment, "noise", selected_theta=40.0, seed=0,
        calibration_sha256=identity.sha256_file(table),
    )[0]
    validate_table(table, task, experiment)
    for field, value in (("output_bounds_version", 2), ("gelu_evaluator_sha256", "9" * 64),
                         ("source_commit", "b" * 40)):
        changed = table_fixture(experiment)
        options = dict(changed["metadata"]["model_options"])
        options[field] = value
        changed["metadata"]["model_options"] = list(options.items())
        table.write_text(json.dumps(changed))
        must_reject(lambda: validate_table(table, task, experiment))
    table.write_text(json.dumps(table_fixture(experiment)))
    log = root / task["log_file"]
    write_log(log, task)
    parsed = parse_result_log(task, root)
    assert parsed["events"] == 100 and parsed["misses"] == 10 and parsed["underflows"] == 1
    result = {
        **result_fixture(experiment, task), **parsed,
        "log_sha256": identity.sha256_file(log),
    }
    validate_result(task, result, experiment, root)
    for field, value in (("task_sha256", "wrong"), ("correct", 0), ("accuracy", float("nan")),
                         ("samples", 4999), ("experiment_sha256", "wrong"), ("events", 200)):
        must_reject(lambda field=field, value=value: validate_result(task, {**result, field: value}, experiment, root))
    original = log.read_text()
    for content in (original.replace("Correct: 4250\n", ""), original + "Correct: 4250\n",
                    original.replace("fingerprint: validation-fingerprint", "fingerprint: wrong"),
                    original + "Traceback (most recent call last)\n"):
        log.write_text(content)
        must_reject(lambda: parse_result_log(task, root))
    log.write_text(original)
    runtime_files.immutable_json(root / "task.json", task)
    runtime_files.immutable_json(root / "task.json", task)
    must_reject(lambda: runtime_files.immutable_json(root / "task.json", {**task, "theta": 80}))


def verify_runtime(experiment: dict) -> None:
    with patch("subprocess.check_output", side_effect=["b" * 40 + "\n", ""]):
        must_reject(lambda: check_source(experiment))
    with patch("subprocess.check_output", side_effect=["a" * 40 + "\n", " M file.py\n"]):
        must_reject(lambda: check_source(experiment))
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
        must_reject(lambda: require_gpu(experiment, "local"))
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4,5"}):
        must_reject(lambda: require_gpu(experiment, "local"))
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}), patch("subprocess.check_output", return_value='{"count":1,"model":"NVIDIA RTX A6000"}'):
        assert require_gpu(experiment, "local") == "NVIDIA RTX A6000"


# @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Campaign]]
def main() -> None:
    experiment = experiment_fixture()
    verify_conditions(experiment)
    verify_selection(experiment)
    runtime_root = ROOT / "artifacts/runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-three-sweep-", dir=runtime_root) as temporary:
        verify_logs(experiment, Path(temporary))
    verify_runtime(experiment)
    print("Calibrated three-sweep contract checks passed (4 groups).")


if __name__ == "__main__":
    main()
