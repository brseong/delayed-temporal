"""Verify partial means, complete intervals, immutable snapshots, and evidence checks."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.summarize_calibrated_three_sweeps import (
    aggregate_results, summarize, T95_THREE_SEEDS,
)
from scripts.experiments import calibrated_three_sweeps as contract
from scripts.runtime import identity


def expect_error(action, message: str) -> None:
    try:
        action()
    except (ValueError, FileNotFoundError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Expected a rejected condition")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.exp = {
            "source_commit": "a" * 40, "checkpoint_sha256": "b" * 64,
            "checkpoint_path": "/synthetic/checkpoint",
            "evaluator_sha256": "c" * 64, "calibration_evaluator_sha256": "d" * 64,
            "calibration_dataset_fingerprint": "training", "dataset_fingerprint": "validation",
        }
        write_json(root / "experiment.json", self.exp)
        self.table_hashes = {}
        for index, theta in enumerate(contract.theta_grid()):
            path = root / f"calibration/theta_{index:02d}.json"
            write_json(path, {"theta": theta, "synthetic": True})
            self.table_hashes[index] = identity.sha256_file(path)
        self.results: dict[str, dict] = {}
        for index in range(9):
            for kind in ("collect", "theta_train", "theta_validation"):
                self.add(kind, index=index)
        self.add("dense")
        self.add("theta_replay", index=4, host="ubai")
        training = [r for r in self.results.values() if r["kind"] == "theta_train"]
        selection = contract.confirm_selection(contract.select_theta(training),
            [r for r in self.results.values() if r["kind"] == "theta_validation"],
            self.results["theta_04_replay"], training)
        write_json(root / "selection.json", selection)

    def add(self, kind: str, *, index: int = 4, seed=None, rt: float = 0.0,
            ratio: float = 0.0, host: str = "local") -> dict:
        task = contract.make_task(self.exp, kind, theta_index=index, seed=seed, rt=rt, ratio=ratio,
            calibration_sha256="" if kind in {"collect", "dense"} else self.table_hashes[index])
        result = {**task, "success": True, "task_sha256": identity.json_sha256(task),
            "experiment_sha256": identity.json_sha256(self.exp), "host_label": host,
            "elapsed_seconds": 10.0, "calibration_sha256": self.table_hashes[index]}
        log = self.root / task["log_file"]
        log.parent.mkdir(parents=True, exist_ok=True)
        if kind == "collect":
            result["sites"] = 48
            log.write_text(f"Calibration identity — mode: collect, sha256: {self.table_hashes[index]}\n")
        else:
            correct = (3800, 4000, 4300, 4490, 4550, 4560, 4560, 4560, 4560)[index]
            if kind in {"theta_validation", "dense"}:
                correct -= 250
            if kind == "noise":
                correct = 4200 + 10 * seed
            result.update(correct=correct, samples=5000, accuracy=correct/5000,
                prediction_sha256="e" * 64, events=0, misses=0, deadline_events=0,
                outputs=0, underflows=0, overflows=0, sites=[])
            if kind == "noise":
                counts = {"events": 100*(seed+1), "misses": seed,
                          "deadline_events": 10, "outputs": 200*(seed+1),
                          "underflows": 1, "overflows": 2}
                result.update(counts)
                result["sites"] = [{"site": "layer.0", **counts,
                                    "deadline_ulp_min": 1e-14, "deadline_ulp_max": 2e-14}]
            log.write_text(f"Synthetic complete evaluation {task['run_id']}\n")
        result["log_sha256"] = identity.sha256_file(log)
        write_json(self.root / "tasks" / (task["run_id"] + ".json"), task)
        write_json(self.root / task["result_file"], result)
        self.results[task["run_id"]] = result
        return result

    def add_seed(self, seed: int) -> None:
        for rt in contract.rt_grid():
            self.add("noise", seed=seed, rt=rt, ratio=4.0)
        for ratio in contract.RATIOS:
            if ratio != 4.0:
                self.add("noise", seed=seed, rt=1e-5, ratio=ratio)

    def mocks(self, *, plot: bool = False) -> ExitStack:
        stack = ExitStack()
        # Metadata and raw-log parsing have independent contract tests. Here the
        # actual result identity, complete counts, log hashes and table hashes remain checked.
        stack.enter_context(patch.object(contract, "validate_table", return_value={}))
        stack.enter_context(patch.object(contract, "parse_result_log", side_effect=lambda task, root:
            {"sites": json.loads((root / task["result_file"]).read_text())["sites"]}))
        if not plot:
            stack.enter_context(patch("scripts.analysis.summarize_calibrated_three_sweeps._plot"))
        return stack


def verify_partial_means_and_counts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        rows = [fixture.add("noise", seed=s, rt=1e-5, ratio=4) for s in (0, 1, 2)]
        for count in (1, 2):
            summary, _ = aggregate_results(rows[:count])
            assert summary[0]["accuracy_ci95_half_width"] is None
            assert summary[0]["provisional"] is True
            assert summary[0]["accuracy_mean"] == statistics.mean(r["accuracy"] for r in rows[:count])
        summary, sites = aggregate_results(rows)
        expected = T95_THREE_SEEDS * statistics.stdev(r["accuracy"] for r in rows) / math.sqrt(3)
        assert math.isclose(summary[0]["accuracy_ci95_half_width"], expected)
        assert summary[0]["provisional"] is False
        assert summary[0]["miss_rate"] == 3/600
        assert summary[0]["rail_saturation_rate"] == 9/1200
        assert sites[0]["miss_rate"] == 3/600 and sites[0]["replicas"] == 3
        clean, _ = aggregate_results([fixture.results["theta_04_validation"]])
        assert clean[0]["miss_rate"] is None and clean[0]["rail_saturation_rate"] is None
        expect_error(lambda: aggregate_results(rows + [rows[0]]), "Duplicate")
        broken = copy.deepcopy(rows)
        broken[0]["accuracy"] = float("nan")
        expect_error(lambda: aggregate_results(broken), "finite")


def verify_seed_barriers_and_snapshots() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        with fixture.mocks():
            initial = summarize(fixture.root)
            assert initial["completed_noise_runs"] == 0
            expect_error(lambda: summarize(fixture.root, snapshot_seed=0), "barrier")
            fixture.add_seed(0)
            first = summarize(fixture.root, snapshot_seed=0)
            assert first["noise_runs_by_seed"] == {"0": 17, "1": 0, "2": 0}
            assert summarize(fixture.root, snapshot_seed=0) == first
            fixture.add_seed(1)
            second = summarize(fixture.root, snapshot_seed=1)
            assert second["completed_noise_runs"] == 34
            assert summarize(fixture.root, snapshot_seed=0) == first
            with (fixture.root / "outputs/seed-1/summary.csv").open() as handle:
                assert all(not r["accuracy_ci95_half_width"] for r in csv.DictReader(handle))
            expect_error(lambda: summarize(fixture.root, require_complete=True), "barrier")
            fixture.add_seed(2)
            final = summarize(fixture.root, snapshot_seed=2, require_complete=True)
            assert final["complete"] and final["completed_evaluations"] == 71
            assert final["completed_collections"] == 9 and final["completed_noise_runs"] == 51
            with (fixture.root / "outputs/seed-2/summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert len([r for r in rows if r["kind"] == "noise"]) == 17
            assert len([r for r in rows if r["kind"] == "theta_validation"]) == 9


def verify_early_seed_and_identity_rejection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        with fixture.mocks():
            result = fixture.add("noise", seed=1, rt=1e-5, ratio=4)
            expect_error(lambda: summarize(fixture.root), "preceding global")
            fixture.add_seed(0)
            summarize(fixture.root)
            path = fixture.root / result["result_file"]
            broken = {**result, "source_commit": "f" * 40}
            write_json(path, broken)
            expect_error(lambda: summarize(fixture.root), "identity mismatch")
            write_json(path, result)
            log = fixture.root / result["log_file"]
            original = log.read_text()
            log.write_text(original + "tampered\n")
            expect_error(lambda: summarize(fixture.root), "log identity changed")
            log.write_text(original)
            selected_path = fixture.root / "selection.json"
            selection = json.loads(selected_path.read_text())
            write_json(selected_path, {**selection, "validation_correct": 1})
            expect_error(lambda: summarize(fixture.root), "Confirmed theta evidence")
            write_json(selected_path, selection)
            summarize(fixture.root, snapshot_seed=0)
            seed0_path = fixture.root / "results/noise_rt_00_seed0.json"
            changed = json.loads(seed0_path.read_text())
            changed["elapsed_seconds"] += 1
            write_json(seed0_path, changed)
            expect_error(lambda: summarize(fixture.root, snapshot_seed=0), "immutable snapshot")


def verify_plot_and_partial_theta() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        with fixture.mocks(plot=True):
            fixture.add_seed(0)
            progress = summarize(fixture.root)
            assert progress["completed_seed_barriers"] == [0]
            for filename in ("latest_accuracy.pdf", "latest_accuracy.png",
                             "latest_physical_rates.pdf", "latest_physical_rates.png"):
                assert (fixture.root / "outputs" / filename).stat().st_size > 1000
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        for path in (fixture.root / "results").glob("*.json"):
            if path.stem != "theta_00_validation":
                path.unlink()
        with fixture.mocks(plot=True):
            progress = summarize(fixture.root)
            assert progress["completed_evaluations"] == 1
            with (fixture.root / "outputs/summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1 and rows[0]["theta"] == "10.0"


# @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Reporting]]
def main() -> None:
    for verify in (verify_partial_means_and_counts, verify_seed_barriers_and_snapshots,
                   verify_early_seed_and_identity_rejection, verify_plot_and_partial_theta):
        verify()
        print(f"PASS {verify.__name__}")
    print("All calibrated three-sweep summary checks passed")


if __name__ == "__main__":
    main()
