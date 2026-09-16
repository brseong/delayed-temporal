"""CPU checks for the current comparison's temporary additional GPU execution."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import run_vit_comparison_extra_gpus as extra
from scripts.experiments import run_vit_comparison as runner
from scripts.experiments import vit_comparison as contract
from scripts.experiments import vit_comparison_controller as controller
from scripts.runtime import identity as runtime_identity
from scripts.verification.verify_vit_comparison_runner import (
    completed_fixture, experiment_fixture, log_fixture, put_json,
)


class ExtraGpuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="comparison-extra-test-")
        self.root = Path(self.temporary.name)
        self.experiment = experiment_fixture(self.root)
        self.key = "imagenet_vit_large"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_gpu_policy_is_unchanged(self) -> None:
        self.assertEqual(controller.LOCAL_GPUS, (4, 5, 6, 7))
        contract.validate_experiment(self.experiment)
        changed = copy.deepcopy(self.experiment)
        changed["local_gpu_ids"] = [1, 2, 3, 4, 5, 6, 7]
        with self.assertRaises(ValueError):
            contract.validate_experiment(changed)
        for gpu in ("0", "1", "2", "3"):
            with self.subTest(gpu=gpu), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": gpu}):
                with self.assertRaises(ValueError):
                    runner.require_gpu(self.experiment, "local")

    def test_nonblocking_run_and_gpu_locks(self) -> None:
        path = self.root / "one.lock"
        with extra.lock(path):
            with self.assertRaises(BlockingIOError):
                with extra.lock(path):
                    self.fail("A duplicate lock was accepted")
        with extra.lock(path):
            pass

    def test_completed_dense_reuse_rechecks_evidence(self) -> None:
        task = contract.make_task(self.experiment, self.key, "dense", 32)
        result = completed_fixture(self.root, self.experiment, task)
        (self.root / "tasks" / (task["run_id"] + ".json")).write_text(
            json.dumps(task, indent=2, sort_keys=True, allow_nan=False) + "\n")
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Unexpected evaluator")):
            self.assertEqual(runner.run_task(self.root, self.experiment, task, "local"), result)
        path = self.root / task["log_file"]
        path.write_text(path.read_text() + "\nchanged")
        with self.assertRaises(ValueError):
            runner.run_task(self.root, self.experiment, task, "local")

    def test_busy_dense_lock_does_not_replace_partial_log(self) -> None:
        task = contract.make_task(self.experiment, self.key, "dense", 32)
        path = self.root / task["log_file"]
        path.parent.mkdir(parents=True)
        path.write_text("partial evaluator output\n")
        locks = self.root / "locks"
        locks.mkdir()
        with extra.lock(locks / (task["run_id"] + ".lock")):
            with self.assertRaises(BlockingIOError):
                runner.run_task(self.root, self.experiment, task, "local")
        self.assertEqual(path.read_text(), "partial evaluator output\n")
        self.assertFalse((self.root / task["result_file"]).exists())

    def test_primary_retry_preserves_unfinished_extra_output(self) -> None:
        task = contract.make_task(self.experiment, self.key, "dense", 32)
        path = self.root / task["log_file"]
        path.parent.mkdir(parents=True)
        original = "interrupted additional dense evaluation\n"
        path.write_text(original)
        disk = ROOT / "artifacts" / "runtime"
        with tempfile.TemporaryDirectory(prefix="comparison-extra-retry-test-", dir=disk) as runtime:
            self.experiment["runtime_root"] = runtime
            class Child:
                def __init__(child, command, *, stdout, **kwargs):
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4")
                    self.assertTrue(kwargs["start_new_session"])
                    text = "\n".join(line for line in log_fixture(self.experiment, task).splitlines()
                                     if not line.startswith(("Slurm identity — ", "Comparison task — ")))
                    stdout.write(text + "\n")
                    stdout.flush()
                def wait(child, **kwargs):
                    return 0
                def poll(child):
                    return 0
            with patch.object(runner, "check_source"), patch.object(runner, "check_assets"), \
                    patch.object(runner, "event"), patch.object(runner.subprocess, "Popen", Child), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
                result = runner.run_task(self.root, self.experiment, task, "local")
            self.assertTrue(result["success"])
            preserved = list((self.root / "rejected").glob("*/*"))
            self.assertEqual([candidate.read_text() for candidate in preserved], [original])



    def admission_evidence(self) -> dict:
        collected = completed_fixture(self.root, self.experiment, contract.make_task(
            self.experiment, self.key, "smoke_collect", 32))
        evaluated = completed_fixture(self.root, self.experiment, contract.make_task(
            self.experiment, self.key, "smoke_spiking", 32,
            calibration_sha256=collected["calibration_sha256"]))
        selected = {field: evaluated[field] for field in (
            "model_key", "batch_size", "correct", "samples", "prediction_sha256", "calibration_sha256")}
        selected["experiment_sha256"] = runtime_identity.json_sha256(self.experiment)
        selected["result_sha256"] = {
            value["run_id"]: runtime_identity.sha256_file(self.root / value["result_file"])
            for value in (collected, evaluated)}
        put_json(self.root / "admissions" / (self.key + ".json"), selected)
        return selected

    def window_evidence(self, remaining: int = 40) -> tuple[dict, dict, Path]:
        self.admission_evidence()
        identity = {"pid": 123456789, "start_ticks": 987, "command": ["fixture-worker"], "state": "R"}
        assignments = {
            "experiment_sha256": runtime_identity.json_sha256(self.experiment),
            "models": {self.key: {"owner": "local", "status": "running"}},
            "local": {self.key: {**identity, "mode": "pipeline", "status": "running"}},
        }
        put_json(self.root / "assignments.json", assignments)
        task = contract.make_task(self.experiment, self.key, "collect", 32)
        put_json(self.root / "tasks" / (task["run_id"] + ".json"), task)
        path = self.root / task["log_file"]
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "Comparison task — " + json.dumps({
                "task_sha256": runtime_identity.json_sha256(task), "batch_size": 32,
            }) + "\n" +
            "Calibration progress — " + json.dumps({"completed_batches": 314 - remaining,
                "total_batches": 314, "completed_samples": 8000, "pass": 2}) + "\n")
        return identity, task, path

    def test_exact_temporary_scope(self) -> None:
        experiment = copy.deepcopy(self.experiment)
        experiment["tag"] = extra.TAG
        experiment.pop("vit_calibration_policy_version")
        for model in experiment["models"]:
            model.pop("calibration_sites", None)
        experiment.update(source_commit=extra.SOURCE_COMMIT, source_root=str(extra.SOURCE))
        path = self.root / "experiment.json"
        put_json(path, experiment)
        with patch.object(extra, "DEFAULT_ROOT", self.root), \
                patch.object(extra, "EXPERIMENT_FILE_SHA256", runtime_identity.sha256_file(path)):
            self.assertEqual(extra.validate_scope(self.root, 1, self.key, True), experiment)
            for gpu in (0, 4, 5, 6, 7, -1, True, "1"):
                with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                    extra.validate_scope(self.root, gpu, self.key, True)
            for gpu in (1, 2, 3):
                self.assertEqual(extra.validate_scope(self.root, gpu, self.key, True), experiment)
            for root, model, enabled in ((self.root / "other", self.key, True),
                                         (self.root, "other", True), (self.root, self.key, False)):
                with self.assertRaises(ValueError):
                    extra.validate_scope(root, 1, model, enabled)
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                extra.validate_scope(self.root, 1, self.key, True)
        for field, value in (("tag", "other"), ("source_commit", "a" * 40),
                              ("source_root", "/different/source"), ("local_gpu_ids", [1, 2, 3])):
            put_json(path, {**experiment, field: value})
            with patch.object(extra, "DEFAULT_ROOT", self.root), \
                    patch.object(
                        extra, "EXPERIMENT_FILE_SHA256", runtime_identity.sha256_file(path)
                    ), self.assertRaises(ValueError):
                extra.validate_scope(self.root, 1, self.key, True)

    def test_mutable_imports_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extra.load_frozen_runner(self.experiment)

    def test_admission_required_and_fully_verified(self) -> None:
        with patch.object(runner, "run_task", side_effect=AssertionError("Unexpected admission evaluation")):
            with self.assertRaises(ValueError):
                extra.prepare_dense(self.root, self.experiment, self.key, runner)
            selected = self.admission_evidence()
            task = extra.prepare_dense(self.root, self.experiment, self.key, runner)
            self.assertEqual(task["kind"], "dense")
            self.assertEqual(task["batch_size"], 32)
            self.assertEqual(task["calibration_file"], "")
            path = self.root / "admissions" / (self.key + ".json")
            put_json(path, {**selected, "batch_size": 64})
            with self.assertRaises(ValueError):
                extra.prepare_dense(self.root, self.experiment, self.key, runner)
            put_json(path, {**selected, "result_sha256": {}})
            with self.assertRaises(ValueError):
                extra.prepare_dense(self.root, self.experiment, self.key, runner)

    def test_progress_parser_rejects_invalid_and_ignores_partial_tail(self) -> None:
        text = "Calibration progress — " + json.dumps(
            {"completed_batches": 274, "total_batches": 314, "pass": 2}) + "\n"
        self.assertEqual(extra.calibration_progress(text)["remaining_batches"], 40)
        self.assertEqual(extra.calibration_progress(text + 'Calibration progress — {"completed_batches":'),
                         extra.calibration_progress(text))
        for payload in ({}, {"completed_batches": True, "total_batches": 314, "pass": 2},
                        {"completed_batches": -1, "total_batches": 314, "pass": 1},
                        {"completed_batches": 315, "total_batches": 314, "pass": 2},
                        {"completed_batches": 0, "total_batches": 0, "pass": 1},
                        {"completed_batches": 1, "total_batches": 314, "pass": 3}):
            with self.assertRaises(ValueError):
                extra.calibration_progress("Calibration progress — " + json.dumps(payload))
        with self.assertRaises(ValueError):
            extra.calibration_progress("unrelated or incomplete log")

    def test_window_requires_live_owned_collection_and_40_batches(self) -> None:
        identity, task, log = self.window_evidence()
        with patch.object(extra, "process_identity", return_value=identity), \
                patch.object(extra, "lock_held", return_value=True):
            self.assertEqual(extra.require_window(self.root, self.experiment, self.key, runner)["remaining_batches"], 40)
            with self.assertRaises(ValueError):
                extra.require_window(self.root, self.experiment, self.key, runner, minimum_remaining=41)
            self.assertEqual(extra.require_window(self.root, self.experiment, self.key, runner, minimum_remaining=0)["main_pid"],
                             identity["pid"])
            original = log.read_text()
            log.write_text(original.replace('"total_batches": 314', '"total_batches": 315'))
            with self.assertRaises(ValueError):
                extra.require_window(self.root, self.experiment, self.key, runner)
            log.write_text(original + original)
            with self.assertRaises(ValueError):
                extra.require_window(self.root, self.experiment, self.key, runner)
            log.write_text(original)
            put_json(self.root / task["result_file"], {})
            with self.assertRaises(ValueError):
                extra.require_window(self.root, self.experiment, self.key, runner)
        (self.root / task["result_file"]).unlink()
        with patch.object(extra, "process_identity", return_value={**identity, "start_ticks": 988}), \
                patch.object(extra, "lock_held", return_value=True), self.assertRaises(ValueError):
            extra.require_window(self.root, self.experiment, self.key, runner)
        with patch.object(extra, "process_identity", return_value=identity), \
                patch.object(extra, "lock_held", return_value=False), self.assertRaises(ValueError):
            extra.require_window(self.root, self.experiment, self.key, runner)
        assignments = json.loads((self.root / "assignments.json").read_text())
        assignments["models"][self.key]["owner"] = "ubai"
        put_json(self.root / "assignments.json", assignments)
        with self.assertRaises(ValueError):
            extra.require_window(self.root, self.experiment, self.key, runner)

    def test_watchdog_stops_only_itself_at_20_batches(self) -> None:
        stop = Mock()
        stop.wait.side_effect = [False, False]
        stop.is_set.return_value = False
        signal_self = Mock()
        reason = []
        with patch.object(extra, "require_window", side_effect=[
                {"remaining_batches": 21}, {"remaining_batches": 20}]), patch.object(extra.os, "getpid", return_value=456789):
            extra.watchdog(self.root, self.experiment, self.key, runner, stop,
                           signal_self=signal_self, interval=0, reason=reason)
        signal_self.assert_called_once_with(456789, signal.SIGTERM)
        self.assertEqual(len(reason), 1)
        stop = Mock()
        stop.wait.return_value = False
        stop.is_set.return_value = False
        signal_self.reset_mock()
        with patch.object(extra, "require_window", side_effect=ValueError("Assignment changed")), \
                patch.object(extra.os, "getpid", return_value=456789):
            extra.watchdog(self.root, self.experiment, self.key, runner, stop,
                           signal_self=signal_self, interval=0)
        signal_self.assert_called_once_with(456789, signal.SIGTERM)

    def test_watchdog_does_not_signal_after_completion(self) -> None:
        stop = threading.Event()
        stop.set()
        signal_self = Mock()
        extra.watchdog(self.root, self.experiment, self.key, runner, stop,
                       signal_self=signal_self, interval=0)
        signal_self.assert_not_called()

    def test_gpu_probe_rejects_occupancy_multiple_devices_and_wrong_family(self) -> None:
        sample = {"memory_used_mib": 271, "utilization_gpu_percent": 0}
        with patch.object(extra.local_gpu, "gpu_activity", return_value={1: sample}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                patch.object(extra.subprocess, "check_output", return_value=json.dumps({"count": 1, "model": "NVIDIA RTX A6000"})):
            self.assertEqual(extra.gpu_probe(self.experiment, 1)["count"], 1)
            sample["memory_used_mib"] = 10000
            with self.assertRaises(RuntimeError):
                extra.gpu_probe(self.experiment, 1)
            sample["memory_used_mib"] = 271
            sample["utilization_gpu_percent"] = 99
            with self.assertRaises(RuntimeError):
                extra.gpu_probe(self.experiment, 1)
            sample["utilization_gpu_percent"] = 0
            for result in ({"count": 2, "model": "NVIDIA RTX A6000"},
                           {"count": 1, "model": "NVIDIA RTX 3090"}, {"count": 0, "model": ""}):
                with patch.object(extra.subprocess, "check_output", return_value=json.dumps(result)), self.assertRaises(ValueError):
                    extra.gpu_probe(self.experiment, 1)
            for visible in ("1,2", "0", "4", ""):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), self.assertRaises(ValueError):
                    extra.gpu_probe(self.experiment, 1)

    def test_execute_reuses_result_before_gpu_or_window_checks(self) -> None:
        self.admission_evidence()
        task = contract.make_task(self.experiment, self.key, "dense", 32)
        completed_fixture(self.root, self.experiment, task)
        with patch.object(extra, "validate_scope", return_value=self.experiment), \
                patch.object(extra, "load_frozen_runner", return_value=runner), \
                patch.object(extra, "require_window", side_effect=AssertionError("Unexpected window")), \
                patch.object(extra, "gpu_probe", side_effect=AssertionError("Unexpected GPU probe")):
            self.assertEqual(extra.execute(self.root, 1, self.key, True)["status"], "reused")

    def test_execute_defers_existing_dense_without_touching_primary(self) -> None:
        self.admission_evidence()
        with patch.object(extra, "validate_scope", return_value=self.experiment), \
                patch.object(extra, "load_frozen_runner", return_value=runner), \
                patch.object(extra, "GPU_LOCK_ROOT", self.root / "gpu-locks"), \
                patch.object(extra, "lock_held", return_value=True), \
                patch.object(extra, "require_window", side_effect=AssertionError("Unexpected window")), \
                patch.object(extra.os, "kill", side_effect=AssertionError("Unexpected signal")):
            self.assertEqual(extra.execute(self.root, 1, self.key, True)["status"], "deferred")

    def full_execute(self, interrupted: bool) -> dict:
        disk = Path("/data/delayed-temporal/artifacts/runtime")
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with tempfile.TemporaryDirectory(prefix="comparison-extra-execute-test-", dir=disk) as runtime:
            self.experiment["runtime_root"] = runtime
            self.admission_evidence()
            put_json(self.root / "assignments.json", {"local": {}})
            progress = {"remaining_batches": 100, "main_pid": 123456789, "main_start_ticks": 987}
            called = []
            def evaluate(root, experiment, task, host):
                called.append(task["run_id"])
                self.assertEqual(host, "local")
                self.assertEqual(task["kind"], "dense")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
                self.assertEqual(os.environ["WANDB_MODE"], "disabled")
                self.assertTrue(Path(os.environ["TMPDIR"]).is_relative_to(Path(runtime)))
                if interrupted:
                    handler = signal.getsignal(signal.SIGTERM)
                    handler(signal.SIGTERM, None)
                    self.fail("The installed interruption handler did not interrupt")
                return completed_fixture(root, experiment, task)
            with patch.object(extra, "validate_scope", return_value=self.experiment), \
                    patch.object(extra, "load_frozen_runner", return_value=runner), \
                    patch.object(extra, "GPU_LOCK_ROOT", self.root / "gpu-locks"), \
                    patch.object(extra, "TAG", Path(runtime).name), \
                    patch.object(extra, "require_window", return_value=progress), \
                    patch.object(extra, "gpu_probe", return_value={"count": 1, "model": "NVIDIA RTX A6000"}), \
                    patch.object(extra, "process_identity", return_value={"start_ticks": 456}), \
                    patch.object(extra.os, "sched_getaffinity", return_value=set(range(32))), \
                    patch.object(extra.os, "sched_setaffinity") as affinity, \
                    patch.object(extra.os, "kill", side_effect=AssertionError("Unexpected actual signal")), \
                    patch.object(extra.subprocess, "check_output", return_value="ext2/ext3\n"), \
                    patch.object(runner, "run_task", side_effect=evaluate), \
                    patch.dict(os.environ), redirect_stdout(io.StringIO()):
                result = extra.execute(self.root, 1, self.key, True)
            affinity.assert_called_once_with(0, {16, 17, 18, 19})
            self.assertEqual(called, [self.key + "_dense"])
        for sig, handler in handlers.items():
            self.assertIs(signal.getsignal(sig), handler)
        events = [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in events], ["temporary_gpu_started", "temporary_gpu_finished"])
        authorization = events[0]["authorization"]
        self.assertEqual(authorization["root"], str(self.root))
        self.assertEqual(authorization["gpu"], 1)
        self.assertTrue(authorization["temporary_local_gpus"])
        self.assertEqual(events[0]["grant_id"], events[1]["grant_id"])
        records = list((self.root / "temporary_local_gpus" / "end").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text())["status"], result["status"])
        with extra.lock(self.root / "gpu-locks" / "gpu-1.lock"):
            pass
        with extra.lock(self.root / "temporary_local_gpus" / (self.key + ".lock")):
            pass
        return result

    def test_execute_complete_logs_real_events_and_restores_handlers(self) -> None:
        self.assertEqual(self.full_execute(False)["status"], "complete")
        self.assertTrue((self.root / "results" / (self.key + "_dense.json")).exists())

    def test_execute_interrupted_logs_real_events_and_defers_safely(self) -> None:
        result = self.full_execute(True)
        self.assertEqual(result["status"], "deferred")
        self.assertIn("interrupted", result["reason"])
        self.assertFalse((self.root / "results" / (self.key + "_dense.json")).exists())


if __name__ == "__main__":
    unittest.main()
