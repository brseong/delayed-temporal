"""Verify the local GPU adapter without running a model or touching GPU jobs."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import identity as runtime_identity

WRAPPER = ROOT / "scripts/experiments/run_calibrated_three_sweep_local_task.py"
spec = importlib.util.spec_from_file_location("_local_worker_adapter_test", WRAPPER)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def must_reject(callback, exceptions=(ValueError, KeyError, FileNotFoundError)) -> None:
    try:
        callback()
    except exceptions:
        return
    raise AssertionError("An invalid worker configuration was accepted")


def verify_control_identity(root: Path) -> None:
    control = root / "control"
    controller = control / "scripts/experiments/run_calibrated_three_sweeps.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("# controller fixture\n")
    output = root / "experiment"
    records = output / "controllers"
    records.mkdir(parents=True)
    head = "a" * 40
    experiment = {"source_commit": "b" * 40}
    identity = {"source_commit": head, "source_root": str(control),
                "evaluator_source_commit": experiment["source_commit"],
                "controller_sha256": runtime_identity.sha256_file(controller),
                "local_worker_sha256": runtime_identity.sha256_file(WRAPPER),
                "local_gpu_ids": [4, 5, 6, 7],
                "supported_local_gpu_ids": list(range(8))}
    identity_path = records / f"{head}.json"
    identity_path.write_text(json.dumps(identity))
    with patch.object(adapter, "CONTROL_ROOT", control), patch.object(adapter, "clean_head", return_value=head):
        assert adapter.check_control_identity(output, experiment) == identity
        for key, value in (("source_commit", "c" * 40), ("source_root", str(root / "other")),
                           ("evaluator_source_commit", "c" * 40), ("controller_sha256", "e" * 64),
                           ("local_worker_sha256", "f" * 64), ("local_gpu_ids", list(range(8))),
                           ("supported_local_gpu_ids", [4, 5, 6, 7])):
            identity_path.write_text(json.dumps({**identity, key: value}))
            must_reject(lambda: adapter.check_control_identity(output, experiment))
        identity_path.write_text(json.dumps(identity))
        must_reject(lambda: adapter.check_control_identity(root / "missing-output", experiment))
    with patch("subprocess.check_output", side_effect=[head + "\n", ""]):
        assert adapter.clean_head(control) == head
    for source_head, dirty in ((head, " M changed.py\n"), ("invalid", "")):
        with patch("subprocess.check_output", side_effect=[source_head + "\n", dirty]):
            must_reject(lambda: adapter.clean_head(control))


def verify_frozen_import(root: Path) -> None:
    source = root / "frozen"
    worker_path, common_path = source / adapter.WORKER_PATH, source / adapter.COMMON_PATH
    worker_path.parent.mkdir(parents=True)
    worker_path.write_text("MARKER = 'frozen'\ndef require_gpu(experiment, host_label): return 'unchanged'\ndef main(): return MARKER\n")
    common_path.write_text("BOUND_POLICY = 3\n")
    experiment = {"source_commit": "b" * 40, "source_root": str(source),
                  "runtime_sha256": {
                      adapter.WORKER_PATH: runtime_identity.sha256_file(worker_path),
                      adapter.COMMON_PATH: runtime_identity.sha256_file(common_path),
                  }}
    original_path = list(sys.path)
    try:
        with patch.object(adapter, "clean_head", return_value=experiment["source_commit"]):
            worker = adapter.load_frozen_worker(experiment)
            assert worker.MARKER == "frozen" and worker.main() == "frozen"
            assert Path(worker.__file__).resolve() == worker_path
            assert sys.path[0] == str(source)
            for relative in (adapter.WORKER_PATH, adapter.COMMON_PATH):
                wrong = {**experiment, "runtime_sha256": {**experiment["runtime_sha256"], relative: "f" * 64}}
                with patch("importlib.util.spec_from_file_location", side_effect=AssertionError("Changed source must not be imported")):
                    must_reject(lambda: adapter.load_frozen_worker(wrong))
            with patch.dict(sys.modules, {"scripts.experiments.calibrated_three_sweeps": SimpleNamespace(__file__="/other/checkout/common.py")}):
                must_reject(lambda: adapter.load_frozen_worker(experiment))
        with patch.object(adapter, "clean_head", return_value="c" * 40):
            must_reject(lambda: adapter.load_frozen_worker(experiment))
    finally:
        sys.path[:] = original_path


def verify_local_gpu_admission() -> None:
    experiment = {"python_bin": "/fixture/python"}
    good = json.dumps({"count": 1, "model": "NVIDIA RTX A6000"})
    for gpu in (4, 5, 6, 7):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": str(gpu)}), patch("subprocess.check_output", return_value=good) as probe:
            assert adapter.require_local_gpu(experiment, "local") == "NVIDIA RTX A6000"
            assert probe.call_args.args[0][:2] == ["/fixture/python", "-c"]
    for gpu in range(8):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": str(gpu)}), patch("subprocess.check_output", return_value=good):
            assert adapter.require_local_gpu(experiment, "local", allow_temporary=True) == "NVIDIA RTX A6000"
    for gpu in range(4):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": str(gpu)}), patch("subprocess.check_output") as probe:
            must_reject(lambda: adapter.require_local_gpu(experiment, "local"))
            probe.assert_not_called()
    for visible in ("", "-1", "8", "0,1", "all", "none", "GPU-uuid", " 0"):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": visible}), patch("subprocess.check_output") as probe:
            must_reject(lambda: adapter.require_local_gpu(experiment, "local"))
            must_reject(lambda: adapter.require_local_gpu(experiment, "local", allow_temporary=True))
            probe.assert_not_called()
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), patch("subprocess.check_output") as probe:
        must_reject(lambda: adapter.require_local_gpu(experiment, "ubai"))
        probe.assert_not_called()
    for response in ({"count": 0, "model": ""}, {"count": 2, "model": "NVIDIA RTX A6000"},
                     {"count": 1, "model": "NVIDIA RTX 3090"}):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}), patch("subprocess.check_output", return_value=json.dumps(response)):
            must_reject(lambda: adapter.require_local_gpu(experiment, "local"))


def verify_temporary_campaign_override(root: Path) -> None:
    output = root / adapter.TEMPORARY_TAG
    output.mkdir()
    experiment = {"tag": adapter.TEMPORARY_TAG, "source_commit": adapter.TEMPORARY_SOURCE_COMMIT,
                  "source_root": "/frozen", "python_bin": "/fixture/python"}
    adapter.validate_temporary_campaign(experiment, output)
    must_reject(lambda: adapter.validate_temporary_campaign({**experiment, "tag": "another-campaign"}, output))
    must_reject(lambda: adapter.validate_temporary_campaign({**experiment, "source_commit": "b" * 40}, output))
    must_reject(lambda: adapter.validate_temporary_campaign(experiment, root / "another-output"))
    experiment_path, task_path = output / "experiment.json", output / "task.json"
    experiment_path.write_text(json.dumps(experiment))
    task = {"seed": 2, "run_id": "noise_rt_00_seed2"}
    task_path.write_text(json.dumps(task))
    argv = [str(WRAPPER), "--experiment", str(experiment_path), "--task", str(task_path),
            "--output-root", str(output), "--host-label", "local", "--temporary-local-gpus"]
    original_require, original_execute, original_command = object(), object(), object()
    worker = SimpleNamespace(require_gpu=original_require, execute=original_execute,
                             evaluator_command=original_command)
    expected_result = {"seed": 2, "accuracy": .85}

    def frozen_main():
        assert sys.argv == argv[:-1]
        assert "--temporary-local-gpus" not in sys.argv
        assert worker.execute is original_execute and worker.evaluator_command is original_command
        assert worker.require_gpu(experiment, "local") == "NVIDIA RTX A6000"
        assert json.loads(task_path.read_text()) == task
        return expected_result

    worker.main = frozen_main
    with patch.object(sys, "argv", argv), patch.object(adapter, "check_control_identity", return_value={}), patch.object(adapter, "load_frozen_worker", return_value=worker), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), patch("subprocess.check_output", return_value='{"count":1,"model":"NVIDIA RTX A6000"}'):
        assert adapter.main() is expected_result
        assert sys.argv is argv
    assert worker.require_gpu is original_require


def verify_exact_delegation(root: Path) -> None:
    output = root / "delegation"
    output.mkdir()
    experiment_path, task_path = output / "experiment.json", output / "task.json"
    experiment = {"source_commit": "b" * 40, "source_root": "/frozen"}
    task = {"run_id": "noise_rt_00_seed1", "seed": 1, "theta": 40,
            "time_noise_std_frac": 1e-5, "deadline_margin_std": 4}
    experiment_path.write_text(json.dumps(experiment))
    task_path.write_text(json.dumps(task))
    before = [path.read_bytes() for path in (experiment_path, task_path)]
    argv = [str(WRAPPER), "--experiment", str(experiment_path), "--task", str(task_path),
            "--output-root", str(output), "--host-label", "local"]
    original_require, original_execute, original_command = object(), object(), object()
    expected_result = {"seed": 1, "correct": 4250, "accuracy": .85, "source_commit": experiment["source_commit"]}
    worker = SimpleNamespace(require_gpu=original_require, execute=original_execute,
                             evaluator_command=original_command)
    original_path = list(sys.path)

    def frozen_main():
        assert sys.argv is argv
        assert worker.require_gpu is adapter.require_local_gpu
        assert worker.execute is original_execute and worker.evaluator_command is original_command
        assert json.loads(task_path.read_text()) == task
        assert json.loads(experiment_path.read_text()) == experiment
        return expected_result

    def frozen_loader(_experiment):
        assert _experiment == experiment
        sys.path.insert(0, "/fixture/frozen-first")
        return worker

    worker.main = frozen_main
    with patch.object(sys, "argv", argv), patch.object(adapter, "check_control_identity", return_value={}) as control, patch.object(adapter, "load_frozen_worker", side_effect=frozen_loader):
        assert adapter.main() is expected_result
        control.assert_called_once_with(output, experiment)
    assert worker.require_gpu is original_require
    assert worker.execute is original_execute and worker.evaluator_command is original_command
    assert sys.path == original_path
    assert [path.read_bytes() for path in (experiment_path, task_path)] == before
    with patch.object(sys, "argv", argv), patch.object(adapter, "check_control_identity", return_value={}), patch.object(adapter, "load_frozen_worker", return_value=worker), patch.object(worker, "main", side_effect=RuntimeError("fixture failure")):
        must_reject(adapter.main, (RuntimeError,))
    assert worker.require_gpu is original_require and sys.path == original_path
    with patch.object(sys, "argv", argv[:-1] + ["ubai"]), redirect_stderr(io.StringIO()):
        must_reject(adapter.main, (SystemExit,))


# @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Local Worker]]
def main() -> None:
    runtime = ROOT / "artifacts/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-local-worker-", dir=runtime) as directory:
        root = Path(directory)
        verify_control_identity(root)
        verify_frozen_import(root)
        verify_local_gpu_admission()
        verify_temporary_campaign_override(root)
        verify_exact_delegation(root)
    print("Local frozen-worker adapter checks passed (5 groups).")


if __name__ == "__main__":
    main()
