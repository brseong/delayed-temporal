"""Verify calibrated ViT UBAI preparation using small CPU-only fixtures."""

from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.summarize_sigma_margin_sweep import HASH_FIELDS, read_manifest
from scripts.experiments.ubai import prepare_calibrated_noise_ubai as prepare
from scripts.experiments.ubai import run_calibrated_noise_task as worker
from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from scripts.setup.hash_artifact import artifact_identity


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="calibrated-ubai-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.source = self.base / "source"
        self.root = self.repo / "artifacts/logs/noise_scan" / prepare.TAG
        self.output = self.base / "deployment"
        self.image = self.base / "image.sqsh"
        self.image.write_bytes(b"small container fixture")
        wrapper = self.repo / "scripts/analysis/evaluate_calibrated_vit.py"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_bytes(b"# evaluator fixture\n")
        evaluator = self.source / "scripts/analysis/gelu_cubic_phi_nl_vit.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_bytes(b"# numerical source fixture\n")
        for name in ("calibrated_noise_task.sbatch", "calibrated_noise_prep.sbatch",
                     "run_calibrated_noise_task.py", "calibrated_git.sh"):
            target = self.repo / "scripts/experiments/ubai" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / "scripts/experiments/ubai" / name).read_bytes())
        for name in ("files.py", "identity.py", "local_gpu.py"):
            target = self.repo / "scripts/runtime" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / "scripts/runtime" / name).read_bytes())
        self.experiment = {
            "source_commit": prepare.SOURCE_COMMIT,
            "checkpoint_path": str(self.base / "checkpoint"),
            "checkpoint_sha256": "a" * 64,
            "calibration_dataset_path": str(self.base / "training"),
            "calibration_dataset_sha256": "b" * 64,
            "calibration_dataset_fingerprint": "cabf903d14d1b1ac",
            "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
            "calibration_evaluator_path": str(wrapper),
            "calibration_evaluator_sha256": identity.sha256_file(wrapper),
            "evaluator_path": str(evaluator.relative_to(self.source)),
            "evaluator_sha256": identity.sha256_file(evaluator),
            "precision": "float64", "batch_size": 32, "theta": 40,
            "runs": 65, "seeds": [0, 1, 2],
            "calibration_samples": 5000, "calibration_seed": 0,
            "calibration_bins": 2048, "calibration_lower_quantile": 0.0,
            "calibration_upper_quantile": 1.0, "calibration_margin_fraction": 0.05,
        }
        self.table = {
            "layers": {f"layer_{index}": {"lower": -1, "upper": 1} for index in range(48)},
            "metadata": {
                "model_id": self.experiment["checkpoint_path"],
                "model_options": [[key, self.experiment[key]] for key in (
                    "source_commit", "checkpoint_sha256", "gelu_cubic_implementation",
                    "gelu_cubic_floor", "calibration_dataset_fingerprint",
                )],
            },
        }
        logs = self.root / "logs"
        logs.mkdir(parents=True)
        (logs / "calibration_collect.log").write_bytes(b"completed calibration fixture\n")
        (logs / "dense_reference.log").write_bytes(b"dense fixture\n")
        (logs / "clean_spiking_baseline.log").write_bytes(b"clean fixture\n")
        self.refresh_evidence()
        self.rows = self.make_rows()
        self.write_rows(self.root / "manifests/grid.tsv", self.rows)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, name, value in (
            (prepare, "REPO", self.repo), (prepare, "CANONICAL_REPO", self.repo),
            (prepare, "CANONICAL_SOURCE", self.source),
            (worker, "SOURCE_ROOT", self.source), (worker, "REPOSITORY_ROOT", self.repo),
            (worker, "WRAPPER_PATH", wrapper), (worker, "EXPERIMENT_ROOT", self.root),
            (worker, "DEPLOYMENT_ROOT", self.output),
        ):
            self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.object(sys, "path", list(sys.path)))

    def refresh_evidence(self) -> None:
        write_json(self.root / "experiment.json", self.experiment)
        write_json(self.root / "calibration.json", self.table)
        self.evidence = {
            "experiment_sha256": identity.sha256_file(self.root / "experiment.json"),
            "calibration_sha256": identity.sha256_file(self.root / "calibration.json"),
            "collection_log_sha256": identity.sha256_file(self.root / "logs/calibration_collect.log"),
            "sites": 48,
        }
        write_json(self.root / "calibration-evidence.json", self.evidence)

    def make_rows(self) -> list[dict[str, str]]:
        conditions = [("baseline", "spiking", 0.0, 0.0, -1), ("baseline", "hf", 0.0, 0.0, -1)]
        cells = {(1e-5 * 10 ** (index / 8), 4.0) for index in range(9)}
        cells.update((1e-5, ratio) for ratio in (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12))
        conditions.extend(("sigma_margin", "spiking", scale, ratio, seed)
                          for scale, ratio in sorted(cells) for seed in (0, 1, 2))
        rows = []
        for index, (stage, backend, scale, ratio, seed) in enumerate(conditions):
            row = {
                "run_id": f"condition_{index}", "stage": stage, "backend": backend,
                "theta": "40", "time_noise_std_frac": repr(scale),
                "time_noise_std_abs": repr(80 * scale), "deadline_margin_std": repr(ratio),
                "deadline_margin_abs": repr(80 * scale * ratio), "seed": str(seed),
                "split": "validation", "expected_samples": "5000", "precision": "float64",
                "dataset_path": str(self.base / "validation"), "dataset_fingerprint": "260dc8e69ecaea24",
                "source_commit": prepare.SOURCE_COMMIT, "gpu_family": "rtxa6000",
                "checkpoint_path": self.experiment["checkpoint_path"],
                "checkpoint_sha256": self.experiment["checkpoint_sha256"],
                "log_file": f"condition_{index}.log", **{name: "c" * 64 for name in HASH_FIELDS},
                "calibration_sha256": self.evidence["calibration_sha256"],
                "calibration_mode": "none" if backend == "hf" else "validate",
            }
            rows.append(row)
        return rows

    @staticmethod
    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prepare.serialize(list(rows[0]), rows))

    def deployment(self) -> dict:
        value = prepare.prepare(self.root, self.output, image=self.image, host_base=self.base / "host")
        value["assigned_manifest_path"] = str(self.output / "manifests/pending.tsv")
        for tool in value.get("runtime_tools", []):
            tool["path"] = str(self.output / Path(tool["path"]).relative_to("/calibrated-deployment"))
        return value

    def validated_contract(self, deployment: dict):
        with patch.object(worker.subprocess, "check_output", side_effect=[prepare.SOURCE_COMMIT + "\n", ""]):
            return worker.validate_contract(deployment, Path(deployment["assigned_manifest_path"]))

    def test_immutable_creation_replay_and_rejection(self) -> None:
        target = self.base / "immutable/data"
        runtime_files.immutable(target, b"original")
        runtime_files.immutable(target, b"original")
        with self.assertRaises(ValueError):
            runtime_files.immutable(target, b"changed")
        self.assertEqual(target.read_bytes(), b"original")

    def test_calibration_identity_and_tampering(self) -> None:
        self.assertEqual(prepare.validate_calibration(self.root), (self.experiment, self.evidence))
        for relative in ("experiment.json", "calibration.json", "logs/calibration_collect.log"):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaises(ValueError, msg=relative):
                prepare.validate_calibration(self.root)
            path.write_bytes(original)

    def test_calibration_rejects_metadata_floor_source_and_site_changes(self) -> None:
        original = copy.deepcopy(self.table)
        for field, value in (("gelu_cubic_floor", 1e-4), ("source_commit", "f" * 40),
                             ("checkpoint_sha256", "d" * 64), ("calibration_dataset_fingerprint", "wrong")):
            self.table = copy.deepcopy(original)
            self.table["metadata"]["model_options"] = [
                [key, value if key == field else existing]
                for key, existing in self.table["metadata"]["model_options"]
            ]
            self.refresh_evidence()
            with self.assertRaises(ValueError, msg=field):
                prepare.validate_calibration(self.root)
        self.table = copy.deepcopy(original)
        self.table["layers"].pop("layer_0")
        self.refresh_evidence()
        with self.assertRaises(ValueError):
            prepare.validate_calibration(self.root)

    def test_preparation_is_not_assignment_or_submission(self) -> None:
        with patch.object(subprocess, "Popen", side_effect=AssertionError("unexpected process")), \
             patch.object(subprocess, "run", side_effect=AssertionError("unexpected process")):
            deployment = self.deployment()
        self.assertEqual(deployment["state"], "prepared")
        self.assertTrue(deployment["assignment_required"])
        self.assertFalse(deployment["paper_promotion_allowed"])
        self.assertEqual(deployment["limits"]["max_running_jobs"], 10)
        self.assertEqual(deployment["limits"]["max_submitted_jobs"], 20)
        self.assertEqual(deployment["limits"]["max_gpus"], 12)
        _, pending = prepare.read_rows(self.output / "manifests/pending.tsv")
        self.assertEqual(len(pending), 63)
        self.assertEqual(len(read_manifest(self.root / "manifests/grid.tsv", require_canonical=False)), 65)
        again = self.deployment()
        self.assertEqual(deployment, again)
        with patch.object(worker, "validate_contract", side_effect=AssertionError("must refuse before validation")):
            with self.assertRaisesRegex(ValueError, "Prepared deployment"):
                worker.execute(self.output / "deployment.json", deployment, self.output / "manifests/pending.tsv", 0)

    def test_worker_contract_and_manifest_numeric_equalities(self) -> None:
        deployment = self.deployment()
        _, specs, assigned = self.validated_contract(deployment)
        self.assertEqual((len(specs), len(assigned)), (65, 63))
        manifest = self.root / "manifests/grid.tsv"
        for field in ("time_noise_std_abs", "deadline_margin_abs"):
            modified = copy.deepcopy(self.rows)
            modified[2][field] = "1"
            self.write_rows(manifest, modified)
            deployment["grid_sha256"] = identity.sha256_file(manifest)
            with self.assertRaises(ValueError, msg=field):
                self.validated_contract(deployment)
        self.write_rows(manifest, self.rows)
        deployment["grid_sha256"] = identity.sha256_file(manifest)
        pending_path = Path(deployment["assigned_manifest_path"])
        _, pending = prepare.read_rows(pending_path)
        pending[0]["seed"] = "2"
        self.write_rows(pending_path, pending)
        deployment["assigned_manifest_sha256"] = identity.sha256_file(pending_path)
        with self.assertRaises(ValueError):
            self.validated_contract(deployment)

    def test_worker_rejects_hash_and_fixed_floor_changes(self) -> None:
        deployment = self.deployment()
        for name in ("worker_sha256", "task_script_sha256", "calibration_sha256", "assigned_manifest_sha256"):
            changed = dict(deployment, **{name: "0" * 64})
            with self.assertRaises(ValueError, msg=name):
                self.validated_contract(changed)
        self.experiment["gelu_cubic_floor"] = 1e-4
        self.refresh_evidence()
        for field, path in (("experiment_sha256", "experiment.json"), ("calibration_evidence_sha256", "calibration-evidence.json")):
            deployment[field] = identity.sha256_file(self.root / path)
        with self.assertRaises(ValueError):
            self.validated_contract(deployment)

    def test_worker_command_preserves_all_conditions(self) -> None:
        for row in self.rows:
            if row["backend"] == "hf":
                continue
            command = worker.evaluator_command(row, self.experiment)
            self.assertEqual(command[0], worker.PYTHON)
            for flag, value in (("--calibration-mode", "validate"), ("--theta", "40"),
                                ("--batch_size", "32"), ("--precision", "float64"),
                                ("--gelu-cubic-floor", "1e-5"), ("--calibration-margin-fraction", "0.05"),
                                ("--calibration-samples", "5000"), ("--calibration-seed", "0"),
                                ("--time-noise-std-frac", row["time_noise_std_frac"]),
                                ("--time-noise-deadline-margin-std", row["deadline_margin_std"]),
                                ("--checkpoint-sha256", row["checkpoint_sha256"]),
                                ("--source-commit", prepare.SOURCE_COMMIT)):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--no-tensorboard", command)
            self.assertIn("--no-mismatch-enabled", command)
            noisy = row["stage"] == "sigma_margin"
            self.assertIn("--gaussian-time-noise" if noisy else "--no-gaussian-time-noise", command)
            self.assertEqual(command[command.index("--time-noise-seed") + 1], row["seed"] if noisy else "0")
            self.assertTrue(math.isclose(float(row["time_noise_std_abs"]), 80 * float(row["time_noise_std_frac"])))
            self.assertTrue(math.isclose(float(row["deadline_margin_abs"]), float(row["deadline_margin_std"]) * float(row["time_noise_std_abs"])))

    def test_worker_runtime_paths_hashes_and_single_gpu(self) -> None:
        deployment = self.deployment()
        self.assertEqual(len(worker.runtime_values(deployment)), 11)
        for value in ("relative/path", "/data/../other", "/data/name:other", "/data/name\nother"):
            with self.assertRaises(ValueError):
                runtime_files.absolute_path(value)
        changed = copy.deepcopy(deployment)
        changed["runtime"]["env_unpacked_bytes"] = 0
        with self.assertRaises(ValueError):
            worker.runtime_values(changed)
        with patch.object(local_gpu.subprocess, "check_output", return_value=json.dumps({"count": 1, "model": "NVIDIA RTX A6000"})) as probe, \
             patch.dict(worker.os.environ, {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_ID": "123"}, clear=True):
            self.assertEqual(local_gpu.require_single_gpu(worker.PYTHON, "ubai"), "NVIDIA RTX A6000")
            for value in ("", "0,1", "-1", "all", "none"):
                with patch.dict(worker.os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                    with self.assertRaises(ValueError):
                        local_gpu.require_single_gpu(worker.PYTHON, "ubai")
            probe.return_value = json.dumps({"count": 2, "model": "NVIDIA RTX A6000"})
            with self.assertRaises(ValueError):
                local_gpu.require_single_gpu(worker.PYTHON, "ubai")
            probe.return_value = json.dumps({"count": 1, "model": "NVIDIA A10"})
            with self.assertRaises(ValueError):
                local_gpu.require_single_gpu(worker.PYTHON, "ubai")

    def test_python_and_asset_checks_require_matching_identity(self) -> None:
        deployment = self.deployment()
        with patch.object(worker.platform, "python_version", return_value="3.12.13"):
            worker.check_python_version(deployment)
        with patch.object(worker.platform, "python_version", return_value="3.11.0"):
            with self.assertRaises(ValueError):
                worker.check_python_version(deployment)
        asset = self.base / "tiny-asset"
        asset.write_bytes(b"known fixture")
        deployment["assets"] = [{"path": str(asset), "aggregate_sha256": artifact_identity(asset)["aggregate_sha256"]}]
        with patch.dict(worker.os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                worker.check_assets(deployment)
        with patch.dict(worker.os.environ, {"SLURM_JOB_ID": "123"}, clear=True):
            self.assertEqual(worker.check_assets(deployment), [{**deployment["assets"][0], "bytes": asset.stat().st_size}])
            asset.write_bytes(b"changed fixture")
            with self.assertRaises(ValueError):
                worker.check_assets(deployment)

    def test_assigned_manifest_duplicates_and_source_changes(self) -> None:
        deployment = self.deployment()
        pending = self.output / "manifests/pending.tsv"
        self.write_rows(pending, [self.rows[2], self.rows[2]])
        with self.assertRaises(ValueError):
            worker.read_assigned(pending)
        self.write_rows(pending, self.rows[2:])
        for head, dirty in (("f" * 40, ""), (prepare.SOURCE_COMMIT, " M source.py\n")):
            with patch.object(worker.subprocess, "check_output", side_effect=[head + "\n", dirty]):
                with self.assertRaises(ValueError):
                    worker.validate_contract(deployment, pending)

    def test_slurm_disk_and_resource_guards(self) -> None:
        script = ROOT / "scripts/experiments/ubai/calibrated_noise_task.sbatch"
        text = script.read_text()
        subprocess.run(["bash", "-n", str(script)], check=True)
        for directive in ("--partition=gpu4,gpu5", "--ntasks=1", "--gres=gpu:1", "--cpus-per-task=4", "--mem=64G"):
            self.assertIn(f"#SBATCH {directive}", text)
        self.assertNotRegex(text, r"(?m)^#SBATCH.*--container-image")
        self.assertIn('enroot_data="/enroot/$task_uid/data"', text)
        self.assertIn("tmpfs|ramfs", text)
        self.assertIn("findmnt", text)
        self.assertIn('mktemp -d --tmpdir="$enroot_data"', text)
        self.assertIn('stat -c %u "$runtime_parent"', text)
        self.assertIn('! -L "$runtime_parent"', text)
        self.assertIn('trap cleanup EXIT', text)
        self.assertIn('trap \'exit 143\' TERM INT', text)
        removals = re.findall(r"(?m)^\s*rm\s+[^\n]+", text)
        self.assertEqual([line.strip() for line in removals], ['rm -rf -- "$runtime_parent"'])
        self.assertNotRegex(text, r"(?m)^\s*find\s")
        self.assertNotIn("/tmp", text)
        self.assertIn('flock 9', text)
        self.assertIn("available_bytes < required_bytes", text)
        self.assertIn("TMPDIR=/work-tmp", text)
        self.assertIn("WANDB_MODE=disabled", text)
        self.assertIn("/usr/bin/env -u WANDB_API_KEY", text)
        prep_script = ROOT / "scripts/experiments/ubai/calibrated_noise_prep.sbatch"
        subprocess.run(["bash", "-n", str(prep_script)], check=True)
        prep_text = prep_script.read_text()
        self.assertIn("#SBATCH --partition=cpu1", prep_text)
        self.assertNotRegex(prep_text, r"(?m)^#SBATCH.*(?:--gres|--gpus)")
        self.assertIn("CALIBRATED_CHECK_ONLY=1", prep_text)


# @lat: [[evaluation#Evaluation and Verification#Calibrated ViT UBAI Preparation]]
def main() -> None:
    """Verify immutable files, calibration identity, and assigned conditions."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
