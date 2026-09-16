#!/usr/bin/env python3
"""Prepare and run two ViT comparison models on disk-backed UBAI allocations."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime import environment as runtime_environment
from scripts.runtime import files as runtime_files
from scripts.runtime import identity

HELPER_PATH = "scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py"
PAIR_PATH = "scripts/experiments/ubai/run_calibrated_three_sweep_pair.py"
SCRIPT_PATH = "scripts/experiments/ubai/run_vit_comparison_ubai.py"
WORKER_PATH = "scripts/experiments/run_vit_comparison.py"
REPOSITORY = Path("/data/delayed-temporal")
DEPLOYMENT_MOUNT = Path("/vit-comparison-deployment")
PYTHON = "/opt/conda/envs/dt/bin/python"
MODELS = ("imagenet_vit_small", "imagenet_vit_base")
GIB = 1024 ** 3


def load_helpers() -> tuple[Any, Any]:
    """Use the maintained cleanup and device helpers without editing their files."""
    modules = []
    for index, relative in enumerate((HELPER_PATH, PAIR_PATH)):
        spec = importlib.util.spec_from_file_location(
            f"_vit_comparison_ubai_helper_{index}", REPO / relative,
        )
        if spec is None or spec.loader is None:
            raise ValueError("Unable to load a frozen runtime helper")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    modules[0].RUNTIME_PREFIX = "vit-comparison-runtime-"
    return modules[0], modules[1]


def validate_deployment(value: dict[str, Any], helper: Any) -> None:
    """Require explicit source, data, environment, and path identities."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value.get("tag", "")):
        raise ValueError("Invalid comparison tag")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("source_commit", "")):
        raise ValueError("An exact source commit is required")
    for key in ("source_root", "experiment_root", "host_source_root", "host_experiment_root"):
        path = runtime_files.absolute_path(value[key])
        if len(path.parts) < 4:
            raise ValueError("A specific source or experiment directory is required")
    experiment = Path(value["experiment_root"])
    if experiment.name != value["tag"] or not experiment.is_relative_to(REPOSITORY / "artifacts"):
        raise ValueError("The experiment must use its tagged artifact directory")
    if "experiment_sha256" in value:
        identity.checked_hash(value["experiment_sha256"])
    for path in value.get("host_git_metadata_paths", []):
        if runtime_files.absolute_path(path).name != ".git":
            raise ValueError("Git metadata mounts must name a .git directory")
    if not value.get("assets"):
        raise ValueError("Artifact identities are required")
    for item in [*value["assets"], *value["dependency_sources"]]:
        for key in ("path", "host_path"):
            if len(runtime_files.absolute_path(item[key]).parts) < 4:
                raise ValueError("An artifact mount cannot be a broad filesystem root")
        identity.checked_hash(item["aggregate_sha256"])
    if len({item["path"] for item in value["assets"]}) != len(value["assets"]):
        raise ValueError("Duplicate artifact mounts are not allowed")
    dependencies = value["dependency_sources"]
    if len(dependencies) != 2 or {item["package"] for item in dependencies} != {"transformers", "spikingjelly"}:
        raise ValueError("Both editable package source identities are required")
    runtime = value["runtime"]
    for key in ("env_archive", "container_image"):
        runtime_files.absolute_path(runtime[key])
        identity.checked_hash(runtime[key + "_sha256"])
    if runtime.get("env_unpacked_bytes") != 96 * GIB or runtime.get("minimum_scratch_bytes") != 16 * GIB:
        raise ValueError("The paired runtime requires a 96 GiB environment and 16 GiB scratch allowance")
    tools = value.get("runtime_tools", [])
    for item in tools:
        path = Path(item["path"])
        if path.is_absolute() or not path.parts or path.parts[0] != "tools" or ".." in path.parts:
            raise ValueError("Runtime tools must use relative tools paths")
        identity.checked_hash(item["sha256"])


def allocation(helper: Any, *, prep: bool) -> str:
    job = helper.require_compute_allocation()
    if os.environ.get("SLURM_NTASKS", "1") != "1" or os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1":
        raise ValueError("The comparison requires one Slurm task on one node")
    cpus, memory = ("4", "65536") if prep else ("8", "131072")
    if os.environ.get("SLURM_CPUS_PER_TASK") != cpus or os.environ.get("SLURM_MEM_PER_NODE") != memory:
        raise ValueError("Slurm CPU or memory allocation differs from the comparison contract")
    if not prep:
        if os.environ.get("SLURM_JOB_PARTITION") not in {"gpu4", "gpu5"}:
            raise ValueError("The paired comparison requires RTX A6000 partitions")
        if os.environ.get("SLURM_GPUS_ON_NODE", "2") != "2":
            raise ValueError("The paired comparison requires exactly two allocated GPUs")
    return job


def verify_source(value: dict[str, Any], helper: Any, *, inside: bool) -> None:
    source = Path(value["source_root" if inside else "host_source_root"])
    if source.resolve() != REPO.resolve():
        raise ValueError("The executing checkout is not the frozen source path")
    helper.source_identity(source, value["source_commit"])
    if not inside:
        metadata = Path(subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--absolute-git-dir"], text=True,
        ).strip()).resolve()
        allowed = [source.resolve(), *(Path(path).resolve() for path in value.get("host_git_metadata_paths", []))]
        if not any(metadata.is_relative_to(path) for path in allowed):
            raise ValueError("External Git metadata has no declared mount")


def file_records(value: dict[str, Any], helper: Any) -> dict[str, Any]:
    """Hash large immutable assets only in the CPU preparation allocation."""
    records: dict[str, Any] = {}
    for group in ("assets", "dependency_sources"):
        collected = []
        for item in value[group]:
            path = Path(item["host_path"])
            method = identity.artifact_records if group == "assets" else identity.package_source_identity
            digest, files = method(path)
            if digest != item["aggregate_sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            collected.append({"root": str(path), "aggregate_sha256": digest, "files": files})
        records[group] = collected
    records["runtime"] = []
    for key in ("env_archive", "container_image"):
        path = Path(value["runtime"][key])
        digest = identity.sha256_file(path)
        if digest != value["runtime"][key + "_sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        stat = path.stat()
        records["runtime"].append({
            "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest,
        })
    return records


def verify_tools(value: dict[str, Any], deployment: Path, helper: Any) -> None:
    for item in value.get("runtime_tools", []):
        if identity.sha256_file(deployment.parent / item["path"]) != item["sha256"]:
            raise ValueError("Runtime Git tool checksum mismatch")
    if "experiment_sha256" in value:
        path = Path(value["host_experiment_root"]) / "experiment.json"
        if identity.sha256_file(path) != value["experiment_sha256"]:
            raise ValueError("Experiment identity differs from the deployment")


def verify_preparation(value: dict[str, Any], deployment: Path, helper: Any) -> dict[str, Any]:
    """Require exact file membership and unchanged stat records after hashing."""
    report_path = deployment.parent / "prep-result.json"
    if report_path.is_symlink():
        raise ValueError("Preparation result must be a regular immutable file")
    report = json.loads(report_path.read_text())
    if (report.get("state") != "verified"
            or report.get("deployment_sha256") != identity.sha256_file(deployment)
            or report.get("source_commit") != value["source_commit"]
            or report.get("python_version") != "3.12.13"):
        raise ValueError("A matching successful CPU preparation is required")
    for group in ("assets", "dependency_sources"):
        records = report.get(group, [])
        expected = {item["host_path"]: item for item in value[group]}
        if len(records) != len(expected) or {record.get("root") for record in records} != set(expected):
            raise ValueError("Preparation roots differ from the deployment")
        for record in records:
            root = Path(record["root"])
            if record.get("aggregate_sha256") != expected[str(root)]["aggregate_sha256"]:
                raise ValueError("Preparation aggregate identity differs")
            files = (
                [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
            ) if group == "assets" else identity.package_source_files(root)
            stored = record.get("files", [])
            if not stored or len(stored) != len({item["path"] for item in stored}):
                raise ValueError("Preparation file records are empty or duplicated")
            if {str(path) for path in files} != {item["path"] for item in stored}:
                raise ValueError("Artifact file membership changed after CPU preparation")
            for item in stored:
                stat = Path(item["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                    raise ValueError("Artifact content changed after CPU preparation")
    runtime = report.get("runtime", [])
    expected_runtime = {value["runtime"][key]: value["runtime"][key + "_sha256"]
                        for key in ("env_archive", "container_image")}
    if len(runtime) != 2 or {item["path"] for item in runtime} != set(expected_runtime):
        raise ValueError("Environment preparation records are incomplete")
    for item in runtime:
        stat = Path(item["path"]).stat()
        if (item["sha256"] != expected_runtime[item["path"]]
                or (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"])):
            raise ValueError("Portable environment or container changed after CPU preparation")
    return report


def container_command(value: dict[str, Any], deployment: Path, runtime: Path, *, prep: bool) -> list[str]:
    source, experiment = Path(value["source_root"]), Path(value["experiment_root"])
    mounts = [
        f'{value["host_source_root"]}:{source}:ro',
        f'{value["host_source_root"]}:{REPOSITORY}:ro',
        f'{deployment.parent}:{DEPLOYMENT_MOUNT}:ro',
        f"{runtime}/dt:/opt/conda/envs/dt:ro",
    ]
    mounts.extend(f"{path}:{path}:ro" for path in value.get("host_git_metadata_paths", []))
    for item in [*value["assets"], *value["dependency_sources"]]:
        mounts.append(f'{item["host_path"]}:{item["path"]}:ro')
    for item in value["dependency_sources"]:
        subtree = "src/transformers/src" if item["package"] == "transformers" else "src/spikingjelly/spikingjelly"
        for root in dict.fromkeys((source, REPOSITORY)):
            mounts.append(f'{item["host_path"]}:{root}/{subtree}:ro')
    # Replace default container tmpfs paths with this allocation's disk scratch.
    for target in ("/tmp", "/var/tmp", "/work-tmp"):
        mounts.append(f"{runtime}/scratch:{target}")
    mounts.append(f'{value["host_experiment_root"]}:{experiment}')
    environment = [
        f"PATH={DEPLOYMENT_MOUNT}/tools:/opt/conda/envs/dt/bin:/usr/bin:/bin",
        f"PYTHONPATH={source}:{source}/src/transformers/src:{source}/src/spikingjelly",
        f"GIT_WORK_TREE={source}", "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
        "TMPDIR=/work-tmp", "TMP=/work-tmp", "TEMP=/work-tmp",
        "WANDB_MODE=disabled", "WANDB_DISABLED=true", "WANDB_SILENT=true",
        "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
    ]
    arguments = [str(source / SCRIPT_PATH), "--inside", "--deployment", str(DEPLOYMENT_MOUNT / deployment.name)]
    if prep:
        arguments.append("--prep")
    return [
        "srun", "--ntasks=1", "--cpus-per-task=4" if prep else "--cpus-per-task=8",
        "--gres=none" if prep else "--gres=gpu:2", "--cpu-bind=cores", "--kill-on-bad-exit=1",
        "--container-image=" + value["runtime"]["container_image"],
        "--container-mounts=" + ",".join(dict.fromkeys(mounts)),
        "--container-workdir=" + str(source),
        "/usr/bin/env", "-u", "WANDB_API_KEY", "-u", "HF_TOKEN", "-u", "HUGGING_FACE_HUB_TOKEN",
        *environment, PYTHON, *arguments,
    ]


def run_workers(value: dict[str, Any], scratch: Path, pair: Any) -> list[int]:
    """Replay the local environment check first; never start models after mismatch."""
    devices = pair.probe_allocated_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        raise ValueError("The paired comparison needs eight allocated CPU cores")
    source, experiment = Path(value["source_root"]), Path(value["experiment_root"])
    children = []
    previous = {}
    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Comparison interrupted by signal {signum}")
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        environments = []
        for index, device in enumerate(devices):
            directory = scratch / f"worker-{index}"
            directory.mkdir(mode=0o700)
            (directory / "xdg-runtime").mkdir(mode=0o700)
            environments.append(runtime_environment.worker_environment(
                dict(os.environ), directory, device
            ))
        common = [PYTHON, str(source / WORKER_PATH)]
        probe = subprocess.Popen(
            [*common, "environment", "--root", str(experiment), "--host-label", "ubai"],
            env=environments[0], start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, set(cpus[:4])),
        )
        children.append(probe)
        code = probe.wait()
        if code:
            raise RuntimeError(f"Cross-environment prediction verification failed: {code}")
        for index, model in enumerate(MODELS):
            affinity = set(cpus[index * 4:index * 4 + 4])
            children.append(subprocess.Popen(
                [*common, "pipeline", "--root", str(experiment), "--model", model, "--host-label", "ubai"],
                env=environments[index], start_new_session=True,
                preexec_fn=lambda selected=affinity: os.sched_setaffinity(0, selected),
            ))
        return [child.wait() for child in children[1:]]
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            pair.reap_children(children)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def inside(value: dict[str, Any], deployment: Path, helper: Any, pair: Any, *, prep: bool) -> None:
    allocation(helper, prep=prep)
    verify_source(value, helper, inside=True)
    scratch = runtime_files.absolute_path(os.environ.get("TMPDIR", ""))
    if scratch == Path("/tmp") or scratch.is_relative_to("/tmp"):
        raise ValueError("Comparison scratch cannot use an unverified /tmp path")
    helper.disk_directory(scratch)
    environment = runtime_environment.worker_environment(
        dict(os.environ), scratch, os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    os.environ.update(environment)
    experiment = Path(value["experiment_root"])
    if prep:
        import platform
        import torch
        import transformers
        import datasets
        import spikingjelly
        if platform.python_version() != "3.12.13":
            raise ValueError("The portable interpreter must be Python 3.12.13")
        from scripts.experiments.run_vit_comparison import package_versions
        versions = package_versions()
        if versions != json.loads((experiment / "experiment.json").read_text())["package_versions"]:
            raise ValueError("Portable package versions differ from the frozen local environment")
        record = {"python_version": platform.python_version(), "torch_version": torch.__version__,
                  "transformers_version": transformers.__version__, "datasets_version": datasets.__version__,
                  "spikingjelly_path": spikingjelly.__file__,
                  "deployment_sha256": identity.sha256_file(deployment), "package_versions": versions}
        runtime_files.atomic_json(experiment / "ubai" / "prep-environment.json", record)
    else:
        codes = run_workers(value, scratch, pair)
        record = {"source_commit": value["source_commit"], "deployment_sha256": identity.sha256_file(deployment),
                  "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
                  "models": dict(zip(MODELS, codes)), "gpus": 2, "cpus_per_worker": 4}
        runtime_files.immutable(
            experiment / "ubai" / f'pair-result-{os.environ["SLURM_JOB_ID"]}-{time.time_ns()}.json',
            runtime_files.json_bytes(record),
        )
        if any(codes):
            raise RuntimeError("One or more comparison models failed; completed results are preserved")


def host(value: dict[str, Any], deployment: Path, helper: Any, pair: Any, *, prep: bool) -> None:
    job = allocation(helper, prep=prep)
    if deployment.parent.resolve() != (Path(value["host_experiment_root"]) / "ubai").resolve():
        raise ValueError("Deployment must be installed in the experiment ubai directory")
    verify_source(value, helper, inside=False)
    verify_tools(value, deployment, helper)
    if prep and (deployment.parent / "prep-result.json").exists():
        verify_preparation(value, deployment, helper)
        return
    records = file_records(value, helper) if prep else verify_preparation(value, deployment, helper)
    if not prep:
        pair.gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    base = Path("/enroot") / str(os.getuid()) / "data" / "vit-comparison"
    helper.disk_directory(base)
    # The maintained ownership format has 8 GiB per reservation. Two records
    # reserve the approved 16 GiB without extracting the environment twice.
    configuration = dict(value["runtime"], minimum_scratch_bytes=8 * GIB)
    runtime, reservation = pair.admit_pair_runtime(helper, base, job, "prep" if prep else "pair", configuration)
    try:
        scratch = runtime / "scratch"
        (scratch / "xdg-runtime").mkdir(mode=0o700)
        environment = runtime_environment.worker_environment(
            dict(os.environ), scratch, os.environ.get("CUDA_VISIBLE_DEVICES", "")
        )
        for key, name in (("ENROOT_RUNTIME_PATH", "enroot-runtime"),
                          ("ENROOT_DATA_PATH", "enroot-data"), ("ENROOT_CACHE_PATH", "enroot-cache")):
            (scratch / name).mkdir(mode=0o700)
            environment[key] = str(scratch / name)
        environment["NVIDIA_VISIBLE_DEVICES"] = "void" if prep else environment["CUDA_VISIBLE_DEVICES"]
        environment["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
        status = helper.run_and_reap(container_command(value, deployment, runtime, prep=prep), environment)
        if status:
            raise RuntimeError(f"Comparison container failed with exit status {status}")
        if prep:
            environment_record = json.loads((Path(value["host_experiment_root"]) / "ubai" / "prep-environment.json").read_text())
            if (environment_record.get("python_version") != "3.12.13"
                    or environment_record.get("deployment_sha256") != identity.sha256_file(deployment)):
                raise ValueError("Portable environment verification is incomplete")
            runtime_files.immutable(deployment.parent / "prep-result.json", runtime_files.json_bytes({
                "state": "verified", "deployment_sha256": identity.sha256_file(deployment),
                "source_commit": value["source_commit"], "python_version": "3.12.13",
                "job_id": job, "node": socket.gethostname(), **records,
            }))
    finally:
        previous = {signum: signal.signal(signum, signal.SIG_IGN) for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            try:
                helper.release_runtime(base, reservation, job)
            finally:
                helper.release_runtime(base, runtime, job)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--inside", action="store_true")
    arguments = parser.parse_args()
    helper, pair = load_helpers()
    deployment = runtime_files.absolute_path(str(arguments.deployment))
    value = json.loads(deployment.read_text())
    validate_deployment(value, helper)
    previous = {}
    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Comparison runtime interrupted by signal {signum}")
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        if arguments.inside:
            inside(value, deployment, helper, pair, prep=arguments.prep)
        else:
            host(value, deployment, helper, pair, prep=arguments.prep)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
