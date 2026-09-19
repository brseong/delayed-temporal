"""CPU checks for comparison ownership, quotas, reassignment and scoped transfer."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.experiments import vit_comparison_controller as module
from scripts.experiments.vit_comparison import make_task
from scripts.verification.verify_vit_comparison_runner import (
    completed_fixture, experiment_fixture, put_json,
)


def controller(root: Path | None = None) -> module.Controller:
    result = module.Controller.__new__(module.Controller)
    result.root = root or Path("/unused-comparison-test-root")
    result.state = {
        "models": {key: {"owner": "ubai" if key in module.REMOTE_MODELS else "local",
                          "status": "ready"} for key in module.MODEL_KEYS},
        "local": {}, "remote": {"staged": True, "prep": {}, "pair": {
            "job_id": "1234", "name": "vc-test-pair-1", "status": "submitted",
            "submitted_at": 100.0,
        }},
    }
    result.remote_root = module.REMOTE_BASE + "/delayed-temporal-experiments/" + module.TAG
    result.remote = Mock(side_effect=AssertionError("Unmocked remote call"))
    result.save = Mock()
    result.event = Mock()
    result.accounting = Mock(return_value=[])
    result.model_complete = Mock(return_value=False)
    result.children = {}
    return result


def queue_row(**updates) -> dict:
    return {"job_id": "1234", "name": "vc-test-pair-1", "state": "PENDING", "gpus": 2, **updates}


def slurm_info(**updates) -> str:
    fields = dict(JobId="1234", JobName="vc-test-pair-1", UserId="sizz1997(12345)",
                  Account="uos", JobState="PENDING", Priority="100", Reason="Resources")
    fields.update(updates)
    return " ".join(f"{key}={value}" for key, value in fields.items())


def deployment_fixture() -> tuple[dict, dict]:
    canonical = "/data/delayed-temporal/artifacts/assets/theta-selection-v1"
    digests = {name: str(index) * 64 for index, name in enumerate(
        ("base", "dataset", "train", "small", "transformers", "spikingjelly"), start=1)}
    paths = {"base": canonical + "/checkpoints/base", "dataset": canonical + "/datasets/validation",
             "train": canonical + "/datasets/train", "small": "/data/nas/vit_small_patch16_224"}
    models = []
    for key, checkpoint in (("imagenet_vit_small", "small"), ("imagenet_vit_base", "base")):
        models.append(dict(model_key=key, checkpoint_path=paths[checkpoint],
                           checkpoint_sha256=digests[checkpoint], dataset_path=paths["dataset"],
                           dataset_sha256=digests["dataset"], calibration_dataset_path=paths["train"],
                           calibration_dataset_sha256=digests["train"]))
    experiment = dict(source_root="/data/delayed-temporal-worktrees/vit-conversion-comparison",
                      source_commit="a" * 40, models=models,
                      dependency_sha256={key: digests[key] for key in ("transformers", "spikingjelly")})
    old = dict(assets_root=canonical, assets=[dict(path=paths[key], aggregate_sha256=digests[key])
                                          for key in ("base", "dataset", "train")],
               dependency_sources=[dict(name=key, path=canonical + "/dependencies/" + key,
                                        aggregate_sha256=digests[key])
                                   for key in ("transformers", "spikingjelly")],
               runtime_tools=[], runtime=dict(
                   host_assets_root=module.REMOTE_BASE + "/delayed-temporal-assets/theta-selection-v1",
                   env_archive="/remote/runtime.tar.gz", env_archive_sha256="a" * 64,
                   container_image="/remote/ubuntu.sqsh", container_image_sha256="b" * 64,
                   env_unpacked_bytes=96 * 1024 ** 3))
    return experiment, old


class QuotaAndProcessTests(unittest.TestCase):
    def test_submission_running_and_gpu_quotas(self) -> None:
        self.assertTrue(module.quota_available([], 2))
        self.assertTrue(module.quota_available([queue_row(gpus=10)], 2))
        self.assertFalse(module.quota_available([queue_row(gpus=11)], 2))
        self.assertTrue(module.quota_available([queue_row(gpus=12)], 0))
        self.assertFalse(module.quota_available([queue_row(gpus=0)] * 20, 0))
        self.assertFalse(module.quota_available([queue_row(state="RUNNING", gpus=0)] * 10, 0))
        self.assertTrue(module.quota_available([queue_row(state="RUNNING", gpus=0)] * 9 +
                                               [queue_row(state="PENDING", gpus=0)] * 10, 2))
        for request in (1, 3, -1):
            with self.assertRaises(ValueError):
                module.quota_available([], request)

    def test_pid_identity_and_reuse_rejection(self) -> None:
        actual = module.process_identity(os.getpid())
        self.assertIsNotNone(actual)
        self.assertTrue(module.matches_process(actual))
        self.assertFalse(module.matches_process(dict(actual, start_ticks=actual["start_ticks"] + 1)))
        self.assertFalse(module.matches_process(dict(actual, command=["not-the-same-command"])))
        self.assertIsNone(module.process_identity(999999999))
        with patch.object(module, "process_identity", return_value=dict(actual, state="Z")):
            self.assertFalse(module.matches_process(actual))

    def test_recover_interrupted_spawn_with_exact_launch_token(self) -> None:
        record = {"command": ["python", "worker.py", "--model", "imagenet_vit_base"],
                  "launch_id": "only-our-launch"}
        identity = dict(pid=555, start_ticks=100, command=record["command"], state="S")
        with patch.object(module.Path, "iterdir", return_value=iter([Path("/proc/555")])), \
             patch.object(module, "process_identity", return_value=identity), \
             patch.object(module.Path, "read_bytes", return_value=b"VIT_COMPARISON_LAUNCH_ID=only-our-launch\0"):
            self.assertEqual(module.recover_starting(record), identity)
        with patch.object(module.Path, "iterdir", return_value=iter([Path("/proc/555")])), \
             patch.object(module, "process_identity", return_value=identity), \
             patch.object(module.Path, "read_bytes", return_value=b"VIT_COMPARISON_LAUNCH_ID=another-launch\0"):
            with self.assertRaises(RuntimeError):
                module.recover_starting(record)

    def test_gpu_scope_and_duplicate_assignment(self) -> None:
        subject = controller()
        for gpu in (0, 1, 2, 3, 8):
            with self.assertRaises(ValueError):
                subject.launch("imagenet_vit_large", "pipeline", gpu)
        subject.state["local"]["cifar10_vit_small"] = {"gpu": 4}
        with self.assertRaises(ValueError):
            subject.launch("imagenet_vit_large", "pipeline", 4)
        with self.assertRaises(ValueError):
            subject.launch("imagenet_vit_base", "pipeline", 5)

    def test_free_gpus_intersects_occupancy_and_assignments(self) -> None:
        subject = controller()
        subject.state["local"] = {"cifar10_vit_small": {"gpu": 4}}
        with patch.object(module.local_gpu, "gpu_activity", return_value={4: True, 5: False, 6: True, 7: True}) as sample, \
             patch.object(module.local_gpu, "gpu_available", side_effect=lambda value: value):
            self.assertEqual(subject.free_gpus(), [6, 7])
            sample.assert_called_once_with(gpu_ids=(4, 5, 6, 7))


class HandoffTests(unittest.TestCase):
    def test_hold_cancel_and_terminal_proof_order(self) -> None:
        subject = controller()
        observed = []

        def remote(arguments: list[str]) -> str:
            observed.append((arguments, deepcopy(subject.state["remote"]["pair"])))
            if arguments[:3] == ["scontrol", "show", "job"]:
                if sum(args[:2] == ["scontrol", "hold"] for args, _ in observed):
                    return slurm_info(Priority="0", Reason="JobHeldUser")
                return slurm_info()
            return ""

        subject.remote = Mock(side_effect=remote)
        with patch.object(module.time, "time", return_value=200):
            subject.tick_handoff([queue_row()], [4, 5])
        commands = [item[0] for item in observed]
        self.assertEqual(commands[1], ["scontrol", "hold", "1234"])
        self.assertEqual(observed[1][1]["status"], "cancelling")
        self.assertEqual(commands[-1], ["scancel", "--state=PENDING", "--user=sizz1997",
                                        "--name=vc-test-pair-1", "1234"])
        self.assertTrue(all(subject.state["models"][key]["owner"] == "ubai" for key in module.REMOTE_MODELS))
        subject.accounting.return_value = [{"state": "CANCELLED", "start": "Unknown", "elapsed_raw": 0}]
        subject.tick_handoff([], [4, 5])
        self.assertTrue(all(subject.state["models"][key]["owner"] == "local" for key in module.REMOTE_MODELS))
        self.assertEqual(subject.state["remote"]["pair"]["status"], "cancelled_for_local")

    def test_handoff_requires_wait_pending_and_two_distinct_allowed_gpus(self) -> None:
        for queue, free, now in (([queue_row()], [4], 200), ([queue_row()], [4, 5], 159),
                                 ([queue_row(state="RUNNING")], [4, 5], 200)):
            subject = controller()
            with patch.object(module.time, "time", return_value=now):
                subject.tick_handoff(queue, free)
            subject.remote.assert_not_called()
        for free in ([4, 4], [0, 1], [4, 8]):
            subject = controller()
            with patch.object(module.time, "time", return_value=200), self.assertRaises(ValueError):
                subject.tick_handoff([queue_row()], free)

    def test_foreign_identity_prevents_mutation(self) -> None:
        for field, value in (("JobId", "1235"), ("JobName", "someone-else"),
                             ("UserId", "other(123)"), ("Account", "other")):
            subject = controller()
            subject.remote = Mock(return_value=slurm_info(**{field: value}))
            with patch.object(module.time, "time", return_value=200), self.assertRaises(ValueError):
                subject.tick_handoff([queue_row()], [4, 5])
            self.assertEqual(subject.remote.call_count, 1)
            self.assertEqual(subject.state["remote"]["pair"]["status"], "submitted")

    def test_existing_admin_or_user_holds_are_not_touched(self) -> None:
        for reason in ("JobHeldAdmin", "JobHeldUser"):
            subject = controller()
            subject.remote = Mock(return_value=slurm_info(Priority="0", Reason=reason))
            with patch.object(module.time, "time", return_value=200):
                subject.tick_handoff([queue_row()], [4, 5])
            self.assertEqual(subject.remote.call_count, 1)
            self.assertFalse(subject.state["remote"]["pair"].get("hold_requested", False))

    def test_start_race_after_hold_never_cancels_running_job(self) -> None:
        subject = controller()
        subject.remote = Mock(side_effect=[slurm_info(), "", slurm_info(JobState="RUNNING", Priority="0", Reason="None"), ""])
        with patch.object(module.time, "time", return_value=200):
            subject.tick_handoff([queue_row()], [4, 5])
        self.assertFalse(any(call.args[0][0] == "scancel" for call in subject.remote.call_args_list))
        self.assertIn(["scontrol", "release", "1234"], [call.args[0] for call in subject.remote.call_args_list])
        self.assertEqual(subject.state["remote"]["pair"]["status"], "submitted")
        self.assertFalse(subject.state["remote"]["pair"]["hold_requested"])
        self.assertTrue(all(subject.state["models"][key]["owner"] == "ubai" for key in module.REMOTE_MODELS))

    def test_terminal_accounting_is_required_and_started_jobs_stay_remote(self) -> None:
        for records in ([], [{"state": "RUNNING", "start": "Unknown", "elapsed_raw": 0}],
                        [{"state": "CANCELLED", "start": "2026-09-15T12:00:00", "elapsed_raw": 0}],
                        [{"state": "CANCELLED", "start": "Unknown", "elapsed_raw": 1}],
                        [{"state": "FAILED", "start": "Unknown", "elapsed_raw": 0}]):
            subject = controller()
            subject.state["remote"]["pair"]["status"] = "cancelling"
            subject.accounting.return_value = records
            subject.tick_handoff([], [4, 5])
            self.assertTrue(all(subject.state["models"][key]["owner"] == "ubai" for key in module.REMOTE_MODELS))
        self.assertFalse(module.never_started([]))
        for value in ("Unknown", "none", "(null)"):
            self.assertTrue(module.never_started([dict(state="CANCELLED", start=value, elapsed_raw=0)]))

    def test_accounting_validates_owner_name_and_account(self) -> None:
        subject = controller()
        del subject.accounting
        job = subject.state["remote"]["pair"]
        good = "1234|vc-test-pair-1|sizz1997|uos|CANCELLED by 123|Unknown|0|\n"
        subject.remote = Mock(return_value=good)
        self.assertTrue(module.never_started(subject.accounting(job)))
        for old, new in (("1234|", "1235|"), ("vc-test-pair-1", "foreign"),
                         ("sizz1997", "other"), ("|uos|", "|other|"), ("|0|", "|bad|")):
            subject.remote = Mock(return_value=good.replace(old, new))
            with self.assertRaises(ValueError):
                subject.accounting(job)


class DeploymentAndTransferTests(unittest.TestCase):
    def test_reused_assets_and_only_missing_small_checkpoint(self) -> None:
        experiment, old = deployment_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment.json").write_text(json.dumps(experiment))
            deployment = module.deployment_for(experiment, root, old)
            self.assertEqual(len(deployment["assets"]), 4)
            new = [item for item in deployment["assets"] if "vit-conversion-comparison-v1" in item["host_path"]]
            self.assertEqual(len(new), 1)
            self.assertEqual(new[0]["path"], experiment["models"][0]["checkpoint_path"])
            self.assertEqual(deployment["source_commit"], experiment["source_commit"])
            self.assertIn(experiment["source_commit"], deployment["host_source_root"])
            self.assertEqual(deployment["runtime"]["minimum_scratch_bytes"], 16 * 1024 ** 3)
            self.assertEqual({item["package"] for item in deployment["dependency_sources"]},
                             {"transformers", "spikingjelly"})
            bad = deepcopy(experiment)
            bad["dependency_sha256"]["transformers"] = "f" * 64
            with self.assertRaises(ValueError):
                module.deployment_for(bad, root, old)
            bad = deepcopy(experiment)
            bad["models"][1]["dataset_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                module.deployment_for(bad, root, old)
            bad = deepcopy(experiment)
            bad["models"][1]["checkpoint_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                module.deployment_for(bad, root, old)

    def test_transfer_is_explicit_and_rejects_parent_or_absolute_paths(self) -> None:
        subject = controller()
        for name in ("../secret", "/absolute", "logs/file\nother"):
            with self.assertRaises(ValueError):
                subject.transfer([name])
        with patch.object(module.subprocess, "run", return_value=None) as execute:
            subject.transfer(["results/imagenet_vit_small_dense.json"], pull=True)
            arguments = execute.call_args.args[0]
            self.assertIn("--files-from=-", arguments)
            self.assertNotIn("--delete", arguments)
            self.assertEqual(execute.call_args.kwargs["input"], "results/imagenet_vit_small_dense.json\n")

    def test_sync_never_fetches_local_owned_model_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ubai").mkdir()
            subject = controller(root)
            subject.experiment = experiment_fixture(root)
            subject.state["models"]["imagenet_vit_small"]["owner"] = "local"
            put_json(root / "admissions/imagenet_vit_base.json", {"batch_size": 32})
            subject.remote_exists = Mock(return_value=False)
            subject.transfer = Mock()
            subject.sync_remote()
            names = subject.transfer.call_args.args[0]
            self.assertTrue(any("imagenet_vit_base_theta_00_collect" in name for name in names))
            self.assertFalse(any("imagenet_vit_small" in name for name in names))
            self.assertFalse(any("cifar10" in name or "vit_large" in name for name in names))
            self.assertNotIn("assignments.json", names)
            self.assertNotIn("experiment.json", names)

    def test_sync_refuses_immutable_result_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ubai").mkdir()
            subject = controller(root)
            subject.experiment = experiment_fixture(root)
            put_json(root / "admissions/imagenet_vit_base.json", {"batch_size": 32})
            selection = {"theta_index": 6, "calibration_sha256": "a" * 64}
            subject.remote_exists = Mock(return_value=True)
            subject.remote = Mock(return_value=json.dumps(selection))
            task = make_task(subject.experiment, "imagenet_vit_base", "dense", 32,
                             theta_index=6, host_label="ubai")
            completed_fixture(root, subject.experiment, task)
            path = root / task["result_file"]
            original = path.read_bytes()

            def transfer(names: list[str], *, pull: bool, target: Path) -> None:
                self.assertTrue(pull)
                result = completed_fixture(target, subject.experiment, task)
                result["elapsed_seconds"] = 2.0
                put_json(target / task["result_file"], result)

            subject.transfer = Mock(side_effect=transfer)
            with self.assertRaises(ValueError):
                subject.sync_remote()
            self.assertEqual(path.read_bytes(), original)

    def test_sync_promotes_only_consistent_task_log_result_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ubai").mkdir()
            subject = controller(root)
            subject.experiment = experiment_fixture(root)
            put_json(root / "admissions/imagenet_vit_base.json", {"batch_size": 32})
            selection = {"theta_index": 6, "calibration_sha256": "a" * 64}
            subject.remote_exists = Mock(return_value=True)
            subject.remote = Mock(return_value=json.dumps(selection))
            task = make_task(subject.experiment, "imagenet_vit_base", "dense", 32,
                             theta_index=6, host_label="ubai")

            def transfer(names: list[str], *, pull: bool, target: Path) -> None:
                completed_fixture(target, subject.experiment, task)

            subject.transfer = Mock(side_effect=transfer)
            subject.sync_remote()
            result = module.read_json(root / task["result_file"])
            module.validate_result(task, result, subject.experiment, root)
            subject.sync_remote()
            self.assertEqual(module.read_json(root / task["result_file"]), result)

    def test_sync_defers_result_when_copied_log_is_still_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ubai").mkdir()
            subject = controller(root)
            subject.experiment = experiment_fixture(root)
            put_json(root / "admissions/imagenet_vit_base.json", {"batch_size": 32})
            selection = {"theta_index": 6, "calibration_sha256": "a" * 64}
            subject.remote_exists = Mock(return_value=True)
            subject.remote = Mock(return_value=json.dumps(selection))
            task = make_task(subject.experiment, "imagenet_vit_base", "dense", 32,
                             theta_index=6, host_label="ubai")

            def transfer(names: list[str], *, pull: bool, target: Path) -> None:
                completed_fixture(target, subject.experiment, task)
                (target / task["log_file"]).write_text("Partial live log snapshot\n")

            subject.transfer = Mock(side_effect=transfer)
            subject.sync_remote()
            self.assertFalse((root / task["result_file"]).exists())

    def test_sync_spiking_result_requires_accepted_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ubai").mkdir()
            subject = controller(root)
            subject.experiment = experiment_fixture(root)
            put_json(root / "admissions/imagenet_vit_base.json", {"batch_size": 32})
            subject.remote_exists = Mock(return_value=False)
            task = make_task(subject.experiment, "imagenet_vit_base", "spiking", 32,
                             theta_index=6, calibration_sha256="a" * 64, host_label="ubai")

            def transfer(names: list[str], *, pull: bool, target: Path) -> None:
                completed_fixture(target, subject.experiment, task)

            subject.transfer = Mock(side_effect=transfer)
            subject.sync_remote()
            self.assertFalse((root / task["result_file"]).exists())
            self.assertFalse((root / task["calibration_file"]).exists())

    def test_controller_rejects_source_or_manifest_change_on_resume(self) -> None:
        experiment, _ = deployment_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / module.TAG
            root.mkdir()
            (root / "experiment.json").write_text(json.dumps(experiment))
            with patch.object(module, "validate_experiment"), \
                 patch.object(module, "check_source") as check, \
                 patch.object(module.socket, "gethostname", return_value="baekryun-cuda129"):
                module.Controller(root)
                check.assert_called_once_with(experiment)
                altered = dict(experiment, source_commit="b" * 40)
                (root / "experiment.json").write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    module.Controller(root)
            with patch.object(module, "validate_experiment"), \
                 patch.object(module, "check_source", side_effect=ValueError("source changed")):
                with self.assertRaises(ValueError):
                    module.Controller(root)


if __name__ == "__main__":
    unittest.main()
