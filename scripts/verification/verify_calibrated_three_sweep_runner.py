"""Verify calibrated sweep scheduling without GPU execution or remote calls."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.calibrated_three_sweeps import make_task, make_tasks, task_sha256
from scripts.experiments.run_calibrated_three_sweeps import (
    Controller, LOCAL_GPUS, NeedsAttention, assert_seed_barrier, controller_identity, default_host,
    gpu_available, pair_tasks_compatible, parse_gpu_activity, parse_gpu_occupancy,
    parse_queue, quota_available, seed_range_reason,
)
from scripts.verification.verify_calibrated_three_sweep_contract import experiment_fixture, result_fixture


def must_reject(callback, exception=(ValueError, NeedsAttention)) -> None:
    try:
        callback()
    except exception:
        return
    raise AssertionError("Invalid scheduling state was accepted")


def noise_tasks(experiment: dict, seed: int = 0) -> list[dict]:
    return make_tasks(experiment, "noise", selected_theta=40.0, seed=seed, calibration_sha256="2" * 64)


def verify_default_distribution_and_order(experiment: dict) -> None:
    candidates = [make_task(experiment, "collect", theta_index=i) for i in range(9)]
    assert Counter(default_host(task, i) for i, task in enumerate(candidates)) == {"local": 3, "ubai": 6}
    tasks = noise_tasks(experiment)
    assert Counter(default_host(task, i) for i, task in enumerate(tasks)) == {"local": 6, "ubai": 11}
    assert len(tasks) == 17
    assert sum(t["time_noise_std_frac"] == 1e-5 and t["deadline_margin_std"] == 4 for t in tasks) == 1
    seed0 = [result_fixture(experiment, task) for task in tasks]
    seed1 = [result_fixture(experiment, task) for task in noise_tasks(experiment, 1)]
    assert_seed_barrier(0, [], experiment, 40.0, "2" * 64)
    assert_seed_barrier(1, seed0, experiment, 40.0, "2" * 64)
    assert_seed_barrier(2, seed0 + seed1, experiment, 40.0, "2" * 64)
    for incomplete in (seed0[:-1], seed0[:9], seed1):
        must_reject(lambda incomplete=incomplete: assert_seed_barrier(1, incomplete, experiment, 40.0, "2" * 64))
    must_reject(lambda: assert_seed_barrier(2, seed0 + seed1[:-1], experiment, 40.0, "2" * 64))
    must_reject(lambda: assert_seed_barrier(1, seed0 + [seed0[0]], experiment, 40.0, "2" * 64))
    for host in ("local", "ubai"):
        smoke = make_tasks(experiment, "smoke_clean", host_label=host, calibration_sha256="2" * 64)[0]
        assert default_host(smoke, 0) == host


def verify_occupancy_and_quota(experiment: dict) -> None:
    devices = "\n".join(f"{i}, GPU-{i}" for i in range(8))
    # Host PIDs remain in diagnostics even when /proc in this container lacks them.
    with patch.object(Path, "exists", return_value=False):
        occupied = parse_gpu_occupancy(devices, "GPU-0, 999999990\nGPU-4, 999999991\nGPU-7, 999999992")
    assert set(occupied) == {4, 5, 6, 7}
    assert occupied[4] == {999999991} and occupied[7] == {999999992}
    assert occupied[5] == occupied[6] == set()
    must_reject(lambda: parse_gpu_occupancy("0,GPU-0", ""))
    must_reject(lambda: parse_gpu_occupancy(devices, "GPU-9,123"))
    must_reject(lambda: parse_gpu_occupancy(devices, "GPU-4,N/A"))
    dummy = Controller.__new__(Controller)
    for gpu in (0, 1, 2, 3, 8, -1):
        with patch("subprocess.Popen", side_effect=AssertionError("No evaluator may start")):
            must_reject(lambda gpu=gpu: dummy.start_local(noise_tasks(experiment)[0], gpu))
    queue = parse_queue("1|RUNNING|foreign|gpu:rtxa6000:2\n2|PENDING|c3-test-one|gpu:1\n3|RUNNING|cpu|N/A")
    assert [row["gpus"] for row in queue] == [2, 1, 0]
    assert quota_available([], "c3-test-") == 10
    rows = lambda count, state="RUNNING", own=False, gpus=1: [
        {"job_id": str(i), "state": state, "name": ("c3-test-" if own else "other-") + str(i), "gpus": gpus}
        for i in range(count)]
    assert quota_available(rows(8, own=True), "c3-test-") == 2
    assert quota_available(rows(10), "c3-test-") == 0
    assert quota_available(rows(20, state="PENDING", gpus=0), "c3-test-") == 0
    assert quota_available(rows(3, gpus=4), "c3-test-") == 0
    assert quota_available(rows(2, gpus=4), "c3-test-") == 4
    for queue in (rows(2), rows(4, state="PENDING"), rows(5, own=True), rows(9, gpus=0)):
        capacity = quota_available(queue, "c3-test-")
        assert 0 <= capacity <= 10
        assert len(queue) + capacity <= 20
        assert sum(row["state"] not in {"PENDING", "CONFIGURING"} for row in queue) + capacity <= 10
        assert sum(row["gpus"] for row in queue) + capacity <= 12


def verify_paired_quota_and_compatibility(experiment: dict) -> None:
    assert quota_available([], "c3-test-", gpus_per_job=2) == 6
    rows = lambda count, gpus=1: [
        {"job_id": str(i), "state": "RUNNING", "name": f"c3-test-{i}", "gpus": gpus}
        for i in range(count)]
    assert quota_available(rows(8), "c3-test-", gpus_per_job=2) == 2
    assert quota_available(rows(6, 2), "c3-test-", gpus_per_job=2) == 0
    assert quota_available(rows(10), "c3-test-", gpus_per_job=2) == 0
    assert quota_available(rows(4, 2), "c3-test-", gpus_per_job=2) == 2
    for queue in (rows(4, 2), rows(8), rows(5), rows(3, 3)):
        for gpus in (1, 2):
            jobs = quota_available(queue, "c3-test-", gpus_per_job=gpus)
            assert len(queue) + jobs <= 10
            assert sum(row["gpus"] for row in queue) + jobs * gpus <= 12
    for invalid in (0, -1, 3):
        must_reject(lambda invalid=invalid: quota_available([], "c3-test-", gpus_per_job=invalid))
    tasks = noise_tasks(experiment)
    assert pair_tasks_compatible(tasks[:2])
    assert not pair_tasks_compatible(tasks[:1])
    assert not pair_tasks_compatible(tasks[:3])
    assert not pair_tasks_compatible([tasks[0], tasks[0]])
    assert not pair_tasks_compatible([tasks[0], noise_tasks(experiment, 1)[1]])
    theta_train = make_task(experiment, "theta_train", theta_index=0, calibration_sha256="2" * 64)
    theta_validation = make_task(experiment, "theta_validation", theta_index=0, calibration_sha256="2" * 64)
    assert pair_tasks_compatible([theta_train, theta_validation])
    assert pair_tasks_compatible([theta_validation, make_task(experiment, "dense")])
    assert not pair_tasks_compatible([tasks[0], theta_train])
    assert not pair_tasks_compatible([make_task(experiment, "collect", theta_index=0), theta_train])
    smoke_clean = make_tasks(experiment, "smoke_clean", host_label="ubai", calibration_sha256="2" * 64)[0]
    smoke_noise = make_tasks(experiment, "smoke_noise", host_label="ubai", calibration_sha256="2" * 64)[0]
    assert pair_tasks_compatible([smoke_clean, smoke_noise])
    local_smoke = make_tasks(experiment, "smoke_clean", host_label="local", calibration_sha256="2" * 64)[0]
    assert not pair_tasks_compatible([local_smoke, smoke_noise])


def activity_fixture(memory: float = 262, utilization: float = 0) -> dict:
    return {gpu: {"gpu_uuid": f"GPU-{gpu}", "memory_used_mib": memory,
                  "utilization_gpu_percent": utilization, "pids": [999999900 + gpu]}
            for gpu in LOCAL_GPUS}


def verify_activity_admission() -> None:
    devices = "\n".join(f"{i},GPU-{i},262,0" for i in range(8))
    applications = "GPU-0,999999990\nGPU-4,999999994\nGPU-4,999999995\nGPU-7,999999997"
    samples = parse_gpu_activity(devices, applications)
    assert set(samples) == {4, 5, 6, 7}
    assert samples[4]["pids"] == [999999994, 999999995]
    assert samples[4]["memory_used_mib"] == 262
    assert samples[4]["utilization_gpu_percent"] == 0
    assert gpu_available(samples[4])  # PID presence alone no longer blocks admission.
    sample = activity_fixture()[4]
    assert gpu_available({**sample, "memory_used_mib": 1024, "utilization_gpu_percent": 5})
    assert gpu_available({**sample, "memory_used_mib": 0, "utilization_gpu_percent": 0})
    assert not gpu_available({**sample, "memory_used_mib": 1025})
    assert not gpu_available({**sample, "utilization_gpu_percent": 6})
    assert not gpu_available({**sample, "memory_used_mib": 1024.01})
    assert not gpu_available({**sample, "utilization_gpu_percent": 5.01})
    for key in ("memory_used_mib", "utilization_gpu_percent"):
        for value in (float("nan"), float("inf"), -1):
            assert not gpu_available({**sample, key: value})
    assert not gpu_available({**sample, "utilization_gpu_percent": 101})
    assert not gpu_available({})
    assert not gpu_available({"memory_used_mib": 262})
    for bad_devices in ("\n".join(devices.splitlines()[:7]),
                        devices + "\n4,GPU-4,262,0",
                        devices + "\n8,GPU-4,262,0",
                        devices.replace("4,GPU-4,262,0", "4,GPU-4,N/A,0"),
                        devices.replace("4,GPU-4,262,0", "4,GPU-4,262,N/A"),
                        devices.replace("4,GPU-4,262,0", "4,GPU-4,nan,0")):
        must_reject(lambda bad_devices=bad_devices: parse_gpu_activity(bad_devices, applications))
    must_reject(lambda: parse_gpu_activity(devices, "GPU-9,1"))
    must_reject(lambda: parse_gpu_activity(devices, "GPU-4,N/A"))


def verify_separate_controller_identity(experiment: dict) -> None:
    helpers = (
        "scripts/experiments/calibrated_three_sweeps.py",
        "scripts/experiments/run_calibrated_three_sweep_task.py",
        "scripts/analysis/summarize_calibrated_three_sweeps.py",
        "scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py",
    )
    identity_experiment = {**experiment, "runtime_sha256": {path: "c" * 64 for path in helpers}}

    def content_hash(path):
        return "d" * 64 if path.name == "run_calibrated_three_sweeps.py" else "c" * 64

    with patch("subprocess.check_output", side_effect=["b" * 40 + "\n", ""]), patch(
            "scripts.experiments.run_calibrated_three_sweeps.sha256_file", side_effect=content_hash):
        identity = controller_identity(identity_experiment)
    assert identity["source_commit"] == "b" * 40
    assert identity["evaluator_source_commit"] == experiment["source_commit"]
    assert identity["controller_sha256"] == "d" * 64
    assert identity["gpu_admission_policy"] == {"max_memory_used_mib": 1024, "max_utilization_gpu_percent": 5}
    with patch("subprocess.check_output", side_effect=["b" * 40 + "\n", " M file.py\n"]):
        must_reject(lambda: controller_identity(identity_experiment))
    for helper in helpers:
        changed = {**identity_experiment, "runtime_sha256": {**identity_experiment["runtime_sha256"], helper: "e" * 64}}
        with patch("subprocess.check_output", side_effect=["b" * 40 + "\n", ""]), patch(
                "scripts.experiments.run_calibrated_three_sweeps.sha256_file", side_effect=content_hash):
            must_reject(lambda: controller_identity(changed))


def verify_range_checks(experiment: dict) -> None:
    tasks = noise_tasks(experiment)
    rows = [result_fixture(experiment, task, correct=4200) for task in tasks]
    assert seed_range_reason(rows, .85) is not None
    for row in rows:
        row.update(correct=50, accuracy=.01)
    assert seed_range_reason(rows, .85) is not None
    for row in rows:
        row.update(correct=2000, accuracy=.4)
    assert seed_range_reason(rows, .85) is None
    rows[0].update(correct=5000, accuracy=1.0)
    assert seed_range_reason(rows, .85) is None  # A nonmonotonic result does not remove a condition.
    must_reject(lambda: seed_range_reason(rows[1:], .85))
    must_reject(lambda: seed_range_reason([row for row in rows if row["deadline_margin_std"] != 4], .85))


class FakeProcess:
    def poll(self) -> int:
        return 0


class FakeController(Controller):
    """Use real scheduling and polling with only launch and transport replaced."""

    def __init__(self, root: Path, experiment: dict):
        self.root = root
        self.root.mkdir(parents=True)
        self.experiment = experiment
        self.source = ROOT
        self.prefix = "c3-test-"
        self.state_path = root / "assignments.json"
        self.state = {"tasks": {}, "experiment_sha256": task_sha256(experiment)}
        self.children = {}
        self.gpu_locks = {}
        self.poll_seconds = 0
        self.max_attempts = 3
        self.outputs = {}
        self.starts = []
        self.pair_starts = []
        self.events = []
        self.transfers = []
        self.failures = {}
        self.queue_text = ""
        self.summary_calls = 0

    def event(self, kind: str, **fields) -> None:
        self.events.append({"kind": kind, **fields})

    def remote(self, arguments, **kwargs) -> str:
        if arguments[0] == "squeue":
            return self.queue_text
        if arguments[0] == "sacct":
            return getattr(self, "accounting_state", "COMPLETED") + "|\n"
        raise AssertionError(f"Unexpected remote action: {arguments}")

    def transfer(self, files, *, pull: bool) -> None:
        self.transfers.append((files, pull))

    def check_preparation(self) -> bool:
        return True

    def completed(self, task: dict) -> dict | None:
        return self.outputs.get(task["run_id"])

    def report(self, **kwargs) -> dict:
        self.summary_calls += 1
        return {}

    def launch(self, task: dict, host: str, gpu: int | None = None) -> None:
        row = self.state["tasks"][task["run_id"]]
        assert row["status"] == "pending"
        row.update(status="running", host=host, attempt=row["attempt"] + 1)
        self.starts.append((task["run_id"], host, gpu))
        if host == "local":
            row.update(gpu=gpu, pid=100 + len(self.starts))
            self.children[task["run_id"]] = FakeProcess()
        else:
            row["job_id"] = str(100 + len(self.starts))
        if row["attempt"] > self.failures.get(task["run_id"], 0):
            self.outputs[task["run_id"]] = result_fixture(self.experiment, task, host=host)

    def start_local(self, task: dict, gpu: int) -> bool:
        assert gpu in LOCAL_GPUS
        self.launch(task, "local", gpu)
        return True

    def start_remote(self, task: dict, queue: list[dict]) -> None:
        self.launch(task, "ubai")
        self.state["tasks"][task["run_id"]]["slurm_name"] = self.prefix + task["run_id"]

    def start_remote_pair(self, tasks: list[dict], queue: list[dict]) -> None:
        assert pair_tasks_compatible(tasks)
        self.pair_starts.append(tuple(task["run_id"] for task in tasks))
        shared_job_id = str(1000 + len(self.pair_starts))
        shared_name = self.prefix + f"pair-{len(self.pair_starts)}"
        for task in tasks:
            self.launch(task, "ubai")
            self.state["tasks"][task["run_id"]].update(job_id=shared_job_id, slurm_name=shared_name)


def verify_resume_retries_and_reassignment(experiment: dict, root: Path) -> None:
    free = activity_fixture()
    tasks = noise_tasks(experiment)
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", return_value=free), patch("time.sleep"):
        controller = FakeController(root / "resume", experiment)
        controller.outputs[tasks[0]["run_id"]] = result_fixture(experiment, tasks[0])
        results = controller.run_tasks("noise-seed-0", tasks)
        assert len(results) == 17 and len(controller.starts) == 16
        assert tasks[0]["run_id"] not in {row[0] for row in controller.starts}
        starts = list(controller.starts)
        controller.run_tasks("noise-seed-0", tasks)
        assert controller.starts == starts  # A complete phase starts nothing on resume.
        missing = FakeController(root / "missing-complete", experiment)
        missing.prepare_task(tasks[0], 0)
        missing.state["tasks"][tasks[0]["run_id"]]["status"] = "complete"
        must_reject(lambda: missing.poll_task(tasks[0], []))
        must_reject(lambda: missing.run_tasks("missing-output", tasks[:1]))
        assert missing.starts == []
        retry = FakeController(root / "retry", experiment)
        retry.failures[tasks[0]["run_id"]] = 1
        retry.run_tasks("retry-phase", tasks[:1])
        assert len(retry.starts) == 2 and retry.state["tasks"][tasks[0]["run_id"]]["attempt"] == 2
        assert sum(event["kind"] == "task_failed" for event in retry.events) == 1
        exhausted = FakeController(root / "exhausted", experiment)
        exhausted.failures[tasks[0]["run_id"]] = 10
        must_reject(lambda: exhausted.run_tasks("failed-phase", tasks[:1]))
        assert len(exhausted.starts) == 3
        remote_full = FakeController(root / "remote-full", experiment)
        remote_full.queue_text = "\n".join(f"{i}|PENDING|c3-test-reserved-{i}|gpu:1" for i in range(10))
        # Ordinal zero is local, ordinal one is UBAI; completed zero leaves only the latter.
        remote_full.outputs[tasks[0]["run_id"]] = result_fixture(experiment, tasks[0])
        remote_full.run_tasks("move-local", tasks[:2])
        assert remote_full.starts == [(tasks[1]["run_id"], "local", 4)]
    busy = activity_fixture(memory=1025)
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", return_value=busy), patch("time.sleep"):
        local_full = FakeController(root / "local-full", experiment)
        local_full.run_tasks("move-remote", tasks[:1])
        assert local_full.starts == [(tasks[0]["run_id"], "ubai", None)]
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", side_effect=ValueError("Invalid GPU measurements")), patch("time.sleep"):
        unavailable = FakeController(root / "no-measurements", experiment)
        unavailable.run_tasks("move-after-invalid-measurements", tasks[:1])
        assert unavailable.starts == [(tasks[0]["run_id"], "ubai", None)]
        assert "Invalid GPU measurements" in unavailable.state["local_wait_reason"]


def verify_owned_worker_reservation(experiment: dict, root: Path) -> None:
    tasks = noise_tasks(experiment)
    controller = FakeController(root / "reserved", experiment)
    controller.state["tasks"]["existing-worker"] = {
        "status": "running", "host": "local", "gpu": 4, "attempt": 1}
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", return_value=activity_fixture()), patch("time.sleep"):
        controller.run_tasks("reserved-device", tasks[:1], force_hosts={tasks[0]["run_id"]: "local"})
    assert controller.starts == [(tasks[0]["run_id"], "local", 5)]


def verify_pair_schedule_and_retry(experiment: dict, root: Path) -> None:
    tasks = noise_tasks(experiment)
    controller = FakeController(root / "pair-schedule", experiment)
    for ordinal, task in enumerate(tasks):
        controller.prepare_task(task, ordinal, "ubai")
    queue = []
    submitted = controller.schedule_remote(tasks, queue)
    assert len(submitted) == len(controller.starts) == 12
    assert len(controller.pair_starts) == len(queue) == 6
    assert sum(job["gpus"] for job in queue) == 12
    assert len({task["run_id"] for task in submitted}) == 12
    odd = FakeController(root / "pair-odd", experiment)
    for ordinal, task in enumerate(tasks[:3]):
        odd.prepare_task(task, ordinal, "ubai")
    odd_queue = []
    assert len(odd.schedule_remote(tasks[:3], odd_queue)) == 3
    assert len(odd.pair_starts) == 1 and sorted(row["gpus"] for row in odd_queue) == [1, 2]
    filtered = FakeController(root / "pair-pending-only", experiment)
    for ordinal, task in enumerate(tasks[:4]):
        filtered.prepare_task(task, ordinal, "local" if ordinal == 0 else "ubai")
    filtered.state["tasks"][tasks[1]["run_id"]]["status"] = "running"
    filtered.state["tasks"][tasks[2]["run_id"]]["status"] = "complete"
    assert filtered.schedule_remote(tasks[:4], []) == tasks[3:4]
    retry = FakeController(root / "pair-peer-retry", experiment)
    retry.failures[tasks[1]["run_id"]] = 1
    retry.accounting_state = "FAILED"
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", return_value=activity_fixture()), patch("time.sleep"):
        result = retry.run_tasks("pair-retry", tasks[:2], force_hosts={task["run_id"]: "ubai" for task in tasks[:2]})
        starts = list(retry.starts)
        assert len(result) == 2 and len(retry.pair_starts) == 1
        assert Counter(row[0] for row in retry.starts) == {tasks[0]["run_id"]: 1, tasks[1]["run_id"]: 2}
        retry.run_tasks("pair-retry", tasks[:2])
        assert retry.starts == starts


def verify_pair_submission_and_recovery(experiment: dict, root: Path) -> None:
    tasks = noise_tasks(experiment)[:2]
    controller = FakeController(root / "pair-submission", experiment)
    for ordinal, task in enumerate(tasks):
        controller.prepare_task(task, ordinal, "ubai")
    with patch.object(controller, "remote", side_effect=AssertionError("No invalid pair may be submitted")):
        must_reject(lambda: Controller.start_remote_pair(controller, tasks[:1], []))
        must_reject(lambda: Controller.start_remote_pair(controller, [tasks[0], tasks[0]], []))
        must_reject(lambda: Controller.start_remote_pair(controller, [tasks[0], noise_tasks(experiment, 1)[1]], []))
        for status in ("complete", "running", "submitting"):
            controller.state["tasks"][tasks[0]["run_id"]]["status"] = status
            must_reject(lambda: Controller.start_remote_pair(controller, tasks, []))
        controller.state["tasks"][tasks[0]["run_id"]]["status"] = "pending"
        controller.state["tasks"][tasks[0]["run_id"]]["fixed_host"] = "local"
        must_reject(lambda: Controller.start_remote_pair(controller, tasks, []))
        controller.state["tasks"][tasks[0]["run_id"]]["fixed_host"] = "ubai"
    controller.pair_controller_verified = True
    controller.remote_controller = "/fixture/paired-controller"
    controller.remote_source = "/fixture/frozen-evaluator"
    controller.remote_root = "/fixture/results"
    controller.state["controller_identity"] = {"source_commit": "b" * 40, "pair_runtime_sha256": {"pair.py": "e" * 64}}
    (controller.root / "experiment.json").write_text(json.dumps(experiment))
    (controller.root / "ubai").mkdir()
    (controller.root / "ubai/deployment.json").write_text("{}")
    with patch.object(controller, "remote", return_value="7654\n") as remote:
        Controller.start_remote_pair(controller, tasks, [])
        assert remote.call_count == 1 and remote.call_args.args[0][0] == "sbatch"
    rows = [controller.state["tasks"][task["run_id"]] for task in tasks]
    assert {row["job_id"] for row in rows} == {"7654"}
    assert len({row["slurm_name"] for row in rows}) == 1
    assert len(list(controller.root.glob("pairs/*.json"))) == 1
    queue = [{"job_id": "7654", "state": "RUNNING", "name": rows[0]["slurm_name"], "gpus": 2}]
    # Resume after sbatch succeeded but before recording its response for both peers.
    for row in rows:
        row["status"] = "submitting"
        row.pop("job_id")
    with patch.object(controller, "remote", side_effect=AssertionError("Recovery must reuse the existing job")):
        for task in tasks:
            assert controller.poll_task(task, queue) is None
            assert controller.state["tasks"][task["run_id"]]["job_id"] == "7654"
    assert all(row["status"] == "running" for row in rows)
    rows[0]["status"] = "submitting"
    must_reject(lambda: controller.poll_task(tasks[0], queue + [{**queue[0], "job_id": "7655"}]))


def verify_locked_launch_recheck(experiment: dict, root: Path) -> None:
    controller = FakeController(root / "locked-launch", experiment)
    controller.root.joinpath("worker_logs").mkdir()
    task = noise_tasks(experiment)[0]
    controller.prepare_task(task, 0)
    lock_root = Path("/data/delayed-temporal/artifacts/runtime/gpu-locks")
    original_open, original_mkdir = Path.open, Path.mkdir

    def local_open(path, *args, **kwargs):
        if path.parent == lock_root:
            return tempfile.TemporaryFile(dir=controller.root)
        return original_open(path, *args, **kwargs)

    def local_mkdir(path, *args, **kwargs):
        if path == lock_root:
            return None
        return original_mkdir(path, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "open", local_open))
        stack.enter_context(patch.object(Path, "mkdir", local_mkdir))
        activity = stack.enter_context(patch("scripts.experiments.run_calibrated_three_sweeps.gpu_activity", return_value=activity_fixture()))
        spawn = stack.enter_context(patch("subprocess.Popen", return_value=SimpleNamespace(pid=123456789)))
        stack.enter_context(patch("os.sched_getaffinity", return_value=set(range(16))))
        stack.enter_context(patch("os.sched_setaffinity"))
        with patch("fcntl.flock", side_effect=BlockingIOError):
            assert not Controller.start_local(controller, task, 4)
        spawn.assert_not_called()
        activity.assert_not_called()  # A held lock blocks even an otherwise eligible device.
        activity.return_value = activity_fixture(memory=1025)
        assert not Controller.start_local(controller, task, 4)
        spawn.assert_not_called()
        activity.return_value = activity_fixture(memory=1024, utilization=5)
        assert Controller.start_local(controller, task, 4)
        assert spawn.call_count == 1
        assert spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
        assert controller.state["tasks"][task["run_id"]]["status"] == "running"
        assert not Controller.start_local(controller, task, 4)
        assert spawn.call_count == 1  # Own running assignment prevents another worker.
        for handle in controller.gpu_locks.values():
            handle.close()


class FakeCampaign(FakeController):
    def __init__(self, root: Path, experiment: dict, *, recovered: bool = False):
        super().__init__(root, experiment)
        self.phases = []
        self.snapshots = []
        self.all_results = []
        self.recovered = recovered

    def report(self, *, seed=None, final=False):
        self.snapshots.append((seed, final))
        return {}

    def results(self):
        return self.all_results

    def run_tasks(self, phase, tasks, *, force_hosts=None):
        self.phases.append(phase)
        scores = (4000, 4100, 4200, 4225, 4250, 4251, 4250, 4252, 4253)
        results = []
        for task in tasks:
            host = (force_hosts or {}).get(task["run_id"], task.get("host_label", "local"))
            correct = 100 if task["kind"].startswith("smoke") else 4250
            if task["kind"] == "collect":
                table = self.root / task["calibration_file"]
                table.parent.mkdir(parents=True, exist_ok=True)
                table.write_text(json.dumps({"theta": task["theta"]}))
            elif task["kind"] == "theta_train":
                correct = scores[task["theta_index"]]
            elif task["kind"] == "noise" and not self.recovered and task["time_noise_std_frac"] > 1e-5:
                correct = 4000
            result = result_fixture(self.experiment, task, correct=correct, host=host)
            results.append(result)
            self.all_results.append(result)
        return results


def verify_campaign_seed_order(experiment: dict, root: Path) -> None:
    campaign = FakeCampaign(root / "campaign", experiment)
    campaign.run()
    assert campaign.phases == ["calibration", "environment-smoke", "theta-evaluation", "theta-replay",
                               "noise-seed-0", "noise-seed-1", "noise-seed-2"]
    assert campaign.snapshots == [(0, False), (1, False), (2, True)]
    assert campaign.state["phase"] == "complete"
    assert Counter(row["seed"] for row in campaign.all_results if row["kind"] == "noise") == {0: 17, 1: 17, 2: 17}
    stopped = FakeCampaign(root / "range-stop", experiment, recovered=True)
    must_reject(stopped.run)
    assert stopped.phases[-1] == "noise-seed-0"
    assert "noise-seed-1" not in stopped.phases and stopped.snapshots == [(0, False)]


# @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Scheduling]]
def main() -> None:
    experiment = experiment_fixture()
    verify_default_distribution_and_order(experiment)
    verify_occupancy_and_quota(experiment)
    verify_paired_quota_and_compatibility(experiment)
    verify_activity_admission()
    verify_separate_controller_identity(experiment)
    verify_range_checks(experiment)
    runtime = ROOT / "artifacts/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-three-sweep-runner-", dir=runtime) as temporary:
        verify_resume_retries_and_reassignment(experiment, Path(temporary))
        verify_owned_worker_reservation(experiment, Path(temporary))
        verify_pair_schedule_and_retry(experiment, Path(temporary))
        verify_pair_submission_and_recovery(experiment, Path(temporary))
        verify_locked_launch_recheck(experiment, Path(temporary))
        verify_campaign_seed_order(experiment, Path(temporary))
    print("Calibrated three-sweep scheduling checks passed (12 groups).")


if __name__ == "__main__":
    main()
