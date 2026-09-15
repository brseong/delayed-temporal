#!/usr/bin/env python3
"""CPU-only checks for the ViT comparison UBAI deployment and paired execution."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.experiments.ubai import run_vit_comparison_ubai as runner


def fixture() -> dict:
    return {
        "tag": "comparison-test", "source_commit": "a" * 40,
        "source_root": "/data/delayed-temporal-worktrees/comparison-test",
        "experiment_root": "/data/delayed-temporal/artifacts/comparison/comparison-test",
        "host_source_root": "/home1/sizz1997/myubai/comparison-source",
        "host_experiment_root": "/home1/sizz1997/myubai/comparison-test",
        "host_git_metadata_paths": ["/home1/sizz1997/myubai/repo/.git"],
        "assets": [{"path": "/data/assets/model", "host_path": "/home1/sizz1997/assets/model", "aggregate_sha256": "b" * 64}],
        "dependency_sources": [
            {"package": name, "path": f"/data/assets/{name}/{subtree}",
             "host_path": f"/home1/sizz1997/assets/{name}/{subtree}", "aggregate_sha256": "c" * 64}
            for name, subtree in (("transformers", "src"), ("spikingjelly", "spikingjelly"))
        ],
        "runtime_tools": [{"path": "tools/git", "sha256": "d" * 64}],
        "runtime": {
            "env_archive": "/home1/sizz1997/assets/dt.tar.zst", "env_archive_sha256": "e" * 64,
            "container_image": "/home1/sizz1997/assets/ubuntu.sqsh", "container_image_sha256": "f" * 64,
            "env_unpacked_bytes": 96 * runner.GIB, "minimum_scratch_bytes": 16 * runner.GIB,
        },
    }


def rejects(callable_) -> None:
    try:
        callable_()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("invalid deployment was accepted")


def verify_deployment_contract() -> None:
    helper, pair = runner.load_helpers()
    value = fixture()
    runner.validate_deployment(value, helper)
    for change in (
        lambda v: v.update(source_commit="branch-name"),
        lambda v: v["runtime"].update(minimum_scratch_bytes=8 * runner.GIB),
        lambda v: v["runtime"].update(env_unpacked_bytes=48 * runner.GIB),
        lambda v: v["assets"][0].update(path="/data"),
        lambda v: v["assets"].append(dict(v["assets"][0])),
        lambda v: v["dependency_sources"].pop(),
        lambda v: v["runtime_tools"][0].update(path="../key"),
    ):
        invalid = copy.deepcopy(value)
        change(invalid)
        rejects(lambda: runner.validate_deployment(invalid, helper))
    base = {
        "SLURM_JOB_ID": "12345", "SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_NODE": "131072",
        "SLURM_JOB_PARTITION": "gpu4", "SLURM_GPUS_ON_NODE": "2",
    }
    with patch.dict(os.environ, base, clear=True):
        assert runner.allocation(helper, prep=False) == "12345"
        for field, invalid in (("SLURM_CPUS_PER_TASK", "4"), ("SLURM_MEM_PER_NODE", "65536"),
                               ("SLURM_GPUS_ON_NODE", "1"), ("SLURM_JOB_PARTITION", "gpu3")):
            with patch.dict(os.environ, {field: invalid}):
                rejects(lambda: runner.allocation(helper, prep=False))
        with patch.object(helper.socket, "gethostname", return_value="gate1.hpc"):
            rejects(lambda: runner.allocation(helper, prep=False))
    for visible in ("0", "0,0", "0,1,2", "-1,1"):
        rejects(lambda: pair.gpu_tokens(visible))


def verify_container_mounts() -> None:
    value = fixture()
    runtime = Path("/enroot/1000/data/vit-comparison/vit-comparison-runtime-12345-pair-abc")
    deployment = Path(value["host_experiment_root"]) / "ubai/deployment.json"
    for prep in (True, False):
        command = runner.container_command(value, deployment, runtime, prep=prep)
        assert command[0] == "srun"
        assert ("--gres=none" if prep else "--gres=gpu:2") in command
        assert ("--cpus-per-task=4" if prep else "--cpus-per-task=8") in command
        mounts = next(arg for arg in command if arg.startswith("--container-mounts=")).split("=", 1)[1].split(",")
        assert f'{value["host_source_root"]}:/data/delayed-temporal:ro' in mounts
        assert f'{value["host_source_root"]}:{value["source_root"]}:ro' in mounts
        assert f"{runtime}/dt:/opt/conda/envs/dt:ro" in mounts
        for target in ("/tmp", "/var/tmp", "/work-tmp"):
            assert f"{runtime}/scratch:{target}" in mounts
        assert f'{value["host_experiment_root"]}:{value["experiment_root"]}' in mounts
        assert "TMPDIR=/work-tmp" in command
        assert "WANDB_MODE=disabled" in command
        for secret in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            index = command.index(secret)
            assert command[index - 1] == "-u"
        assert "--inside" in command
        assert ("--prep" in command) == prep
    assert not any(line.startswith("#SBATCH --container-image") for line in
                   (REPO / "scripts/experiments/ubai/vit_comparison_task.sbatch").read_text().splitlines())


def verify_preparation_records() -> None:
    helper, _pair = runner.load_helpers()
    with tempfile.TemporaryDirectory(prefix="verify-vit-comparison-") as temporary:
        base = Path(temporary)
        value = fixture()
        value["assets"] = []
        value["dependency_sources"] = []
        for group, names in (("assets", ("model", "dataset")),
                             ("dependency_sources", ("transformers", "spikingjelly"))):
            for name in names:
                root = base / name
                root.mkdir()
                (root / "content").write_bytes(name.encode())
                method = helper.artifact_records if group == "assets" else helper.package_source_identity
                digest, _records = method(root)
                value[group].append({"path": f"/data/artifacts/{name}", "host_path": str(root),
                                     "aggregate_sha256": digest, "package": name})
        for key in ("env_archive", "container_image"):
            path = base / key
            path.write_bytes(key.encode())
            value["runtime"][key] = str(path)
            value["runtime"][key + "_sha256"] = helper.sha256(path)
        deployment = base / "deployment.json"
        deployment.write_text(json.dumps(value))
        records = runner.file_records(value, helper)
        report = {
            "state": "verified", "deployment_sha256": helper.sha256(deployment),
            "source_commit": value["source_commit"], "python_version": "3.12.13", **records,
        }
        report_path = base / "prep-result.json"
        report_path.write_text(json.dumps(report))
        runner.verify_preparation(value, deployment, helper)
        wrong = copy.deepcopy(report)
        wrong["assets"][0]["files"].append(dict(wrong["assets"][0]["files"][0]))
        report_path.write_text(json.dumps(wrong))
        rejects(lambda: runner.verify_preparation(value, deployment, helper))
        report_path.write_text(json.dumps(report))
        extra = Path(value["assets"][0]["host_path"]) / "unexpected"
        extra.write_text("unexpected")
        rejects(lambda: runner.verify_preparation(value, deployment, helper))
        extra.unlink()
        (Path(value["assets"][0]["host_path"]) / "content").write_text("changed")
        rejects(lambda: runner.verify_preparation(value, deployment, helper))


def verify_disk_reservations() -> None:
    helper, pair = runner.load_helpers()
    assert helper.RUNTIME_PREFIX == "vit-comparison-runtime-"
    with tempfile.TemporaryDirectory(prefix="verify-vit-runtime-") as temporary:
        base = Path(temporary)
        with patch.object(helper.subprocess, "check_output", return_value="tmpfs"):
            rejects(lambda: helper.disk_directory(base))
        configuration = dict(fixture()["runtime"], minimum_scratch_bytes=8 * runner.GIB)
        with patch.object(helper, "disk_directory"), patch.object(helper, "extract_environment") as unpack, \
             patch.object(helper.shutil, "disk_usage", return_value=SimpleNamespace(free=200 * runner.GIB)):
            runtime, reservation = pair.admit_pair_runtime(helper, base, "12345", "pair", configuration)
            assert unpack.call_count == 1
            assert sum(helper.owned_runtime(base, path)["scratch_bytes"]
                       for path in (runtime, reservation)) == 16 * runner.GIB
            assert runtime.name.startswith("vit-comparison-runtime-12345-")
            with patch.object(helper, "terminal_job", return_value=False):
                assert len(helper.sweep_old_runtime(base, "54321")) == 2
            helper.release_runtime(base, reservation, "12345")
            helper.release_runtime(base, runtime, "12345")
            assert not runtime.exists() and not reservation.exists()
        with patch.object(helper, "disk_directory"), patch.object(helper, "extract_environment") as unpack, \
             patch.object(helper.shutil, "disk_usage", return_value=SimpleNamespace(free=50 * runner.GIB)):
            rejects(lambda: pair.admit_pair_runtime(helper, base, "12345", "pair", configuration))
            assert unpack.call_count == 0


def verify_environment_gate_and_workers() -> None:
    _helper, pair = runner.load_helpers()
    value = fixture()
    for exit_codes in ([1], [0, 0, 0], [0, 2, 0]):
        spawned = []
        pending = iter(exit_codes)
        def launch(command, **kwargs):
            code = next(pending)
            spawned.append((command, kwargs))
            return SimpleNamespace(wait=lambda: code)
        with tempfile.TemporaryDirectory(prefix="verify-vit-workers-") as temporary, \
             patch.object(pair, "probe_allocated_devices", return_value=["GPU-a", "GPU-b"]), \
             patch.object(pair, "reap_children") as reaper, \
             patch.object(runner.os, "sched_getaffinity", return_value=set(range(8))), \
             patch.object(runner.subprocess, "Popen", side_effect=launch):
            if exit_codes[0]:
                rejects(lambda: runner.run_workers(value, Path(temporary), pair))
                assert len(spawned) == 1
            else:
                assert runner.run_workers(value, Path(temporary), pair) == list(exit_codes[1:])
                assert len(spawned) == 3
                for index, (command, kwargs) in enumerate(spawned[1:]):
                    assert command[2] == "pipeline"
                    assert command[command.index("--model") + 1] == runner.MODELS[index]
                    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ("GPU-a", "GPU-b")[index]
                    assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
                    assert kwargs["env"]["WANDB_MODE"] == "disabled"
                    assert kwargs["start_new_session"]
                    assert not kwargs["env"]["TMPDIR"].startswith("/tmp/worker")
            assert spawned[0][0][2] == "environment"
            reaper.assert_called_once()


def verify_batch_headers() -> None:
    for name, cpus, memory in (("prep", "4", "64G"), ("task", "8", "128G")):
        path = REPO / f"scripts/experiments/ubai/vit_comparison_{name}.sbatch"
        subprocess.run(["bash", "-n", str(path)], check=True)
        content = path.read_text()
        assert f"#SBATCH --cpus-per-task={cpus}" in content
        assert f"#SBATCH --mem={memory}" in content
        assert "--ntasks=1" in content
        assert "/home1/sizz1997/miniconda3/bin/python" in content
        if name == "task":
            assert "#SBATCH --gres=gpu:2" in content
            assert "#SBATCH --partition=gpu4,gpu5" in content
        else:
            assert "#SBATCH --partition=cpu1" in content


def verify_vit_comparison_ubai() -> None:
    for check in (
        verify_deployment_contract, verify_container_mounts, verify_preparation_records,
        verify_disk_reservations, verify_environment_gate_and_workers, verify_batch_headers,
    ):
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    verify_vit_comparison_ubai()
