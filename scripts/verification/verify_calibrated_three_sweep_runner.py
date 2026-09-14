"""Verify calibrated sweep scheduling without GPU execution or remote calls."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.calibrated_three_sweeps import make_task, make_tasks, task_sha256
from scripts.experiments.run_calibrated_three_sweeps import (
    Controller, LOCAL_GPUS, NeedsAttention, assert_seed_barrier, default_host,
    parse_gpu_occupancy, parse_queue, quota_available, seed_range_reason,
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
    # NVIDIA host PIDs remain occupied even when /proc in this container lacks them.
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
    assert quota_available([], "c3-test-") == 8
    rows = lambda count, state="RUNNING", own=False, gpus=1: [
        {"job_id": str(i), "state": state, "name": ("c3-test-" if own else "other-") + str(i), "gpus": gpus}
        for i in range(count)]
    assert quota_available(rows(8, own=True), "c3-test-") == 0
    assert quota_available(rows(10), "c3-test-") == 0
    assert quota_available(rows(20, state="PENDING", gpus=0), "c3-test-") == 0
    assert quota_available(rows(3, gpus=4), "c3-test-") == 0
    assert quota_available(rows(2, gpus=4), "c3-test-") == 4
    for queue in (rows(2), rows(4, state="PENDING"), rows(5, own=True), rows(9, gpus=0)):
        capacity = quota_available(queue, "c3-test-")
        assert 0 <= capacity <= 8
        assert len(queue) + capacity <= 20
        assert sum(row["state"] not in {"PENDING", "CONFIGURING"} for row in queue) + capacity <= 10
        assert sum(row["gpus"] for row in queue) + capacity <= 12


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
            return "COMPLETED|\n"
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


def verify_resume_retries_and_reassignment(experiment: dict, root: Path) -> None:
    free = {gpu: set() for gpu in LOCAL_GPUS}
    tasks = noise_tasks(experiment)
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_occupancy", return_value=free), patch("time.sleep"):
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
        remote_full.queue_text = "\n".join(f"{i}|PENDING|c3-test-reserved-{i}|gpu:1" for i in range(8))
        # Ordinal zero is local, ordinal one is UBAI; completed zero leaves only the latter.
        remote_full.outputs[tasks[0]["run_id"]] = result_fixture(experiment, tasks[0])
        remote_full.run_tasks("move-local", tasks[:2])
        assert remote_full.starts == [(tasks[1]["run_id"], "local", 4)]
    busy = {gpu: {999999900 + gpu} for gpu in LOCAL_GPUS}
    with patch("scripts.experiments.run_calibrated_three_sweeps.gpu_occupancy", return_value=busy), patch("time.sleep"):
        local_full = FakeController(root / "local-full", experiment)
        local_full.run_tasks("move-remote", tasks[:1])
        assert local_full.starts == [(tasks[0]["run_id"], "ubai", None)]


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
    verify_range_checks(experiment)
    runtime = ROOT / "artifacts/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-three-sweep-runner-", dir=runtime) as temporary:
        verify_resume_retries_and_reassignment(experiment, Path(temporary))
        verify_campaign_seed_order(experiment, Path(temporary))
    print("Calibrated three-sweep scheduling checks passed (5 groups).")


if __name__ == "__main__":
    main()
