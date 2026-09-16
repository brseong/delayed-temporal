"""Resume four ViT comparison pipelines across the local host and UBAI."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import uuid

from scripts.experiments.vit_comparison import (
    MODEL_KEYS, TAG, check_source, model_by_key, read_json,
    validate_experiment, validate_result, validate_task,
)
from scripts.experiments.calibrated_three_sweep_rebalance import parse_slurm_job, TERMINAL
from scripts.runtime import environment as runtime_environment
from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from scripts.runtime import slurm

REMOTE_BASE = "/home1/sizz1997/myubai"
USER = "sizz1997"
ACCOUNT = "uos"
LOCAL_GPUS = (4, 5, 6, 7)
REMOTE_MODELS = ("imagenet_vit_small", "imagenet_vit_base")
LOCAL_MODELS = ("imagenet_vit_large", "cifar10_vit_small")
OLD_DEPLOYMENT = Path("/data/delayed-temporal/artifacts/logs/noise_scan/"
                      "vit_base_calibrated_theta_rt_ratio_float64_bounds3_v1/ubai/deployment.json")
WORKER = "scripts/experiments/run_vit_comparison.py"
MAX_ATTEMPTS = 3


def quota_available(queue: list[dict], gpus: int) -> bool:
    """Reserve submitted GPU requests while respecting running and queued job limits."""
    if gpus not in (0, 2):
        raise ValueError("Comparison allocations request zero or two GPUs")
    running = sum(row["state"] not in {"PENDING", "PD"} for row in queue)
    return len(queue) < 20 and running < 10 and sum(row["gpus"] for row in queue) + gpus <= 12


def process_identity(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return {"pid": pid, "start_ticks": int(fields[19]), "command": command, "state": fields[0],
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")}
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def matches_process(record: dict) -> bool:
    current = process_identity(int(record.get("pid", -1)))
    return bool(current and current["state"] != "Z" and
                all(current[field] == record.get(field) for field in ("pid", "start_ticks", "command")))


def recover_starting(record: dict) -> dict | None:
    """Recover a spawn whose PID was not persisted, using its unique environment token."""
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        current = process_identity(int(path.name))
        if not current or current["state"] == "Z" or current["command"] != record["command"]:
            continue
        try:
            environment = (path / "environ").read_bytes().split(b"\0")
        except OSError:
            raise RuntimeError("Cannot establish ownership of an existing comparison worker")
        token = ("VIT_COMPARISON_LAUNCH_ID=" + record["launch_id"]).encode()
        if token not in environment:
            raise RuntimeError("An unassigned comparison worker already uses this model")
        matches.append(current)
    if len(matches) > 1:
        raise ValueError("Multiple processes match one launch identity")
    return matches[0] if matches else None


def resident_tree(pid: int) -> list[dict]:
    """Read one worker and its descendants without trusting recycled process IDs."""
    pending, seen, records = [pid], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        identity = process_identity(current)
        if not identity or identity["state"] == "Z":
            continue
        records.append(identity)
        try:
            pending.extend(int(child) for child in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except OSError:
            pass
    return records


def validate_job_identity(info: dict, job: dict) -> None:
    if (info["JobId"] != str(job["job_id"]) or info["JobName"] != job["name"]
            or info["UserId"].split("(")[0] != USER or info["Account"] != ACCOUNT):
        raise ValueError("Slurm ownership differs from the recorded comparison assignment")


def never_started(records: list[dict]) -> bool:
    return bool(records) and all(
        row["state"] == "CANCELLED" and row["start"].lower() in {"unknown", "none", "(null)"}
        and row["elapsed_raw"] == 0 for row in records
    )


def deployment_for(experiment: dict, root: Path, old: dict) -> dict:
    """Map only the two remote models onto verified reusable cluster assets."""
    host_assets = Path(old["runtime"]["host_assets_root"])
    old_assets = {item["aggregate_sha256"]: item for item in old["assets"]}
    assets = {}
    for key in REMOTE_MODELS:
        model = model_by_key(experiment, key)
        for field in ("checkpoint", "dataset", "calibration_dataset"):
            canonical, digest = model[field + "_path"], model[field + "_sha256"]
            if canonical in assets:
                if assets[canonical]["aggregate_sha256"] != digest:
                    raise ValueError("Conflicting shared dataset identities")
                continue
            if digest in old_assets:
                original = Path(old_assets[digest]["path"])
                host = host_assets / original.relative_to(old["assets_root"])
            elif key == "imagenet_vit_small" and field == "checkpoint":
                host = Path(REMOTE_BASE) / "delayed-temporal-assets/vit-conversion-comparison-v1/checkpoints" / Path(canonical).name
            else:
                raise ValueError("An expected reusable UBAI asset is unavailable")
            assets[canonical] = {"path": canonical, "host_path": str(host), "aggregate_sha256": digest}
    dependencies = []
    for item in old["dependency_sources"]:
        package = item["name"]
        if item["aggregate_sha256"] != experiment["dependency_sha256"][package]:
            raise ValueError("Existing UBAI editable source differs from the frozen comparison")
        dependencies.append({
            "package": package, "path": item["path"],
            "host_path": str(host_assets / Path(item["path"]).relative_to(old["assets_root"])),
            "aggregate_sha256": item["aggregate_sha256"],
        })
    runtime = {key: old["runtime"][key] for key in (
        "env_archive", "env_archive_sha256", "container_image", "container_image_sha256", "env_unpacked_bytes")}
    runtime["minimum_scratch_bytes"] = 16 * 1024 ** 3
    return {
        "tag": TAG, "source_root": experiment["source_root"], "source_commit": experiment["source_commit"],
        "experiment_root": str(root),
        "experiment_sha256": identity.sha256_file(root / "experiment.json"),
        "host_source_root": f'{REMOTE_BASE}/delayed-temporal-comparisons/{experiment["source_commit"]}',
        "host_experiment_root": f"{REMOTE_BASE}/delayed-temporal-experiments/{TAG}",
        "host_git_metadata_paths": [], "assets": list(assets.values()),
        "dependency_sources": dependencies, "runtime": runtime, "runtime_tools": old["runtime_tools"],
    }


class Controller:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.experiment = read_json(self.root / "experiment.json")
        validate_experiment(self.experiment)
        check_source(self.experiment)
        if socket.gethostname() != "baekryun-cuda129" or self.root.name != TAG:
            raise ValueError("The central comparison controller belongs on the local host")
        self.source = Path(self.experiment["source_root"])
        self.remote_root = f"{REMOTE_BASE}/delayed-temporal-experiments/{TAG}"
        self.state_path = self.root / "assignments.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "experiment_sha256": identity.json_sha256(self.experiment), "models": {
                key: {"owner": "ubai" if key in REMOTE_MODELS else "local", "status": "admission_pending",
                      "admission_attempts": 0, "pipeline_attempts": 0} for key in MODEL_KEYS},
            "local": {}, "remote": {"staged": False, "prep": {}, "pair": {}}, "phase": "running",
        }
        if (self.state["experiment_sha256"] != identity.json_sha256(self.experiment)
                or set(self.state["models"]) != set(MODEL_KEYS)):
            raise ValueError("Central assignments differ from the immutable experiment")
        self.children: dict[str, subprocess.Popen] = {}
        self.prefix = "vc-" + self.experiment["source_commit"][:7] + "-"
        self.last_summary = None
        self.root.joinpath("worker_logs").mkdir(exist_ok=True)
        self.root.joinpath("ubai").mkdir(exist_ok=True)
        self.save()

    def save(self) -> None:
        self.state["updated_at"] = time.time()
        runtime_files.atomic_json(self.state_path, self.state)

    def event(self, name: str, **fields: Any) -> None:
        from scripts.experiments.run_vit_comparison import event
        event(self.root, name, **fields)

    def remote(self, arguments: list[str], *, timeout: int = 45) -> str:
        return subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "gate1", shlex.join(arguments)],
            text=True, timeout=timeout,
        )

    def remote_exists(self, path: str, kind: str = "f") -> bool:
        try:
            self.remote(["test", "-" + kind, path])
            return True
        except subprocess.CalledProcessError as error:
            if error.returncode == 1:
                return False
            raise

    def transfer(self, names: list[str], *, pull: bool = False, target: Path | None = None) -> None:
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or "\n" in name:
                raise ValueError("Unsafe comparison transfer path")
        local = str(target or self.root) + "/"
        remote = "gate1:" + self.remote_root + "/"
        source, destination = (remote, local) if pull else (local, remote)
        subprocess.run(["rsync", "-a", "--files-from=-", "--ignore-missing-args", source, destination],
                       input="\n".join(names) + "\n", text=True, check=True, timeout=180)

    def stage(self) -> None:
        if self.state["remote"]["staged"]:
            return
        old = read_json(OLD_DEPLOYMENT)
        deployment = deployment_for(self.experiment, self.root, old)
        from scripts.experiments.ubai.run_vit_comparison_ubai import load_helpers, validate_deployment
        helper, _pair = load_helpers()
        validate_deployment(deployment, helper)
        destination = self.root / "ubai"
        for item in deployment["runtime_tools"]:
            original = OLD_DEPLOYMENT.parent / item["path"]
            if identity.sha256_file(original) != item["sha256"]:
                raise ValueError("Preserved portable Git tool differs")
            target = destination / item["path"]
            runtime_files.immutable(target, original.read_bytes())
            target.chmod(original.stat().st_mode & 0o777)
        runtime_files.immutable_json(destination / "deployment.json", deployment)
        self.remote(["mkdir", "-p", self.remote_root + "/ubai",
                     str(Path(deployment["host_source_root"]).parent)])
        self.transfer(["experiment.json", "ubai/deployment.json",
                       *(str(Path("ubai") / item["path"]) for item in deployment["runtime_tools"])])
        remote_source = deployment["host_source_root"]
        if not self.remote_exists(remote_source, "d"):
            bundle = destination / f'source-{self.experiment["source_commit"]}.bundle'
            if not bundle.exists():
                subprocess.run(["git", "-C", str(self.source), "bundle", "create", str(bundle), "HEAD"], check=True)
            self.transfer([str(bundle.relative_to(self.root))])
            self.remote(["git", "clone", "--no-checkout", self.remote_root + "/" + str(bundle.relative_to(self.root)),
                         remote_source], timeout=180)
            self.remote(["git", "-C", remote_source, "checkout", "--detach", self.experiment["source_commit"]], timeout=90)
        head = self.remote(["git", "-C", remote_source, "rev-parse", "HEAD"]).strip()
        dirty = self.remote(["git", "-C", remote_source, "status", "--porcelain", "--untracked-files=no"]).strip()
        if head != self.experiment["source_commit"] or dirty:
            raise ValueError("Remote source differs from the immutable comparison checkout")
        for asset in deployment["assets"]:
            small = model_by_key(self.experiment, "imagenet_vit_small")
            if asset["path"] != small["checkpoint_path"]:
                if not self.remote_exists(asset["host_path"], "d"):
                    raise ValueError("Required reusable remote data is missing; no bulk copy is authorized")
                continue
            # Complete an interrupted first transfer without replacing existing files.
            self.remote(["mkdir", "-p", asset["host_path"]])
            subprocess.run(["rsync", "-a", "--ignore-existing", asset["path"] + "/",
                            "gate1:" + asset["host_path"] + "/"], check=True, timeout=300)
        # Empty mount targets are not source changes and avoid nested read-only mount creation.
        targets = [remote_source + "/src/transformers/src", remote_source + "/src/spikingjelly/spikingjelly"]
        for item in [*deployment["assets"], *deployment["dependency_sources"]]:
            canonical = Path(item["path"])
            if canonical.is_relative_to("/data/delayed-temporal"):
                targets.append(str(Path(remote_source) / canonical.relative_to("/data/delayed-temporal")))
        targets.append(str(Path(remote_source) / self.root.relative_to("/data/delayed-temporal")))
        self.remote(["mkdir", "-p", *targets])
        self.state["remote"]["staged"] = True
        self.save()
        self.event("ubai_staged", source_commit=head)

    def admitted(self, key: str) -> bool:
        if not (self.root / "admissions" / (key + ".json")).exists():
            return False
        from scripts.experiments.run_vit_comparison import admission
        admission(self.root, self.experiment, key, "local")
        return True

    def model_complete(self, key: str) -> bool:
        for kind in ("collect", "dense", "spiking"):
            run_id = f"{key}_{kind}"
            path = self.root / "results" / (run_id + ".json")
            if not path.exists():
                return False
            validate_result(read_json(self.root / "tasks" / (run_id + ".json")), read_json(path),
                            self.experiment, self.root)
        return True

    def launch(self, key: str, mode: str, gpu: int) -> None:
        if (gpu not in LOCAL_GPUS or key in self.state["local"]
                or any(row["gpu"] == gpu for row in self.state["local"].values())):
            raise ValueError("Duplicate or unauthorized local assignment")
        if mode == "pipeline" and self.state["models"][key]["owner"] != "local":
            raise ValueError("Remote-owned work cannot start locally")
        cpus = sorted(os.sched_getaffinity(0))[(gpu - 4) * 4:(gpu - 4) * 4 + 4]
        if len(cpus) != 4:
            raise ValueError("Four separate CPU cores per local worker are required")
        command = [self.experiment["python_bin"], "-u", str(self.source / WORKER),
                   mode, "--root", str(self.root), "--model", key, "--host-label", "local"]
        attempt_key = "admission_attempts" if mode == "admit" else "pipeline_attempts"
        model = self.state["models"][key]
        model[attempt_key] += 1
        record = {"mode": mode, "gpu": gpu, "command": command, "status": "starting",
                  "launch_id": uuid.uuid4().hex, "started_at": time.time(), "cpus": cpus}
        self.state["local"][key] = record
        model["status"] = "admitting" if mode == "admit" else "running"
        self.save()
        scratch = Path(self.experiment["runtime_root"]) / "controllers" / record["launch_id"]
        scratch.mkdir(parents=True)
        environment = runtime_environment.worker_environment(dict(os.environ), scratch, str(gpu))
        environment.update(VIT_COMPARISON_LAUNCH_ID=record["launch_id"], PYTHONUNBUFFERED="1")
        log = self.root / "worker_logs" / f"{key}-{mode}-{record['launch_id']}.log"
        with log.open("x") as handle:
            child = subprocess.Popen(command, env=environment, cwd=self.source, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)))
        self.children[key] = child
        identity = process_identity(child.pid)
        if identity is None:
            record.update(pid=child.pid, start_ticks=-1)
        else:
            record.update(identity)
        record.update(status="running", worker_log=str(log.relative_to(self.root)))
        self.save()
        self.event("local_worker_started", model_key=key, mode=mode, gpu=gpu, pid=child.pid)

    def poll_local(self) -> None:
        for key, record in list(self.state["local"].items()):
            if record["status"] == "starting":
                recovered = recover_starting(record)
                if recovered:
                    record.update(recovered, status="running")
                    self.save()
            if matches_process(record):
                members = resident_tree(record["pid"])
                record["rss_bytes"] = sum(member["rss_bytes"] for member in members)
                if record.get("memory_stop_at") is not None:
                    if time.time() - record["memory_stop_at"] >= 60:
                        for member in record["memory_stop_members"]:
                            if matches_process(member):
                                try:
                                    os.kill(member["pid"], signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                elif record["rss_bytes"] > 64 * 1024 ** 3:
                    record.update(memory_stop_at=time.time(), memory_stop_members=members)
                    self.save()
                    self.event("local_memory_limit", model_key=key, rss_bytes=record["rss_bytes"])
                    if matches_process(record):
                        try:
                            os.kill(record["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                continue
            child = self.children.pop(key, None)
            code = child.wait() if child is not None else None
            mode = record["mode"]
            success = self.admitted(key) if mode == "admit" else self.model_complete(key)
            model = self.state["models"][key]
            history = dict(record, exit_code=code, success=success, finished_at=time.time())
            model.setdefault("history", []).append(history)
            attempts = model["admission_attempts" if mode == "admit" else "pipeline_attempts"]
            model["status"] = ("ready" if mode == "admit" else "complete") if success else (
                "failed" if attempts >= MAX_ATTEMPTS else "admission_pending" if mode == "admit" else "ready")
            del self.state["local"][key]
            self.save()
            self.event("local_worker_finished", model_key=key, mode=mode, success=success, exit_code=code)

    def free_gpus(self) -> list[int]:
        occupied = {record["gpu"] for record in self.state["local"].values()}
        samples = local_gpu.gpu_activity(gpu_ids=LOCAL_GPUS)
        return [
            gpu for gpu in LOCAL_GPUS
            if gpu not in occupied and local_gpu.gpu_available(samples[gpu])
        ]

    def queue(self) -> list[dict]:
        return slurm.parse_queue(
            self.remote(["squeue", "--noheader", "--user", USER, "--format=%i|%T|%j|%b"])
        )

    def accounting(self, job: dict) -> list[dict]:
        text = self.remote(["sacct", "-n", "-X", "-j", str(job["job_id"]),
                            "--format=JobIDRaw,JobName%200,User%64,Account%64,State,Start,ElapsedRaw", "--parsable2"])
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.rstrip("|").split("|")
            if len(fields) != 7:
                raise ValueError("Incomplete Slurm accounting identity")
            identifier, name, user, account, state, started, elapsed = [value.strip() for value in fields]
            if identifier != str(job["job_id"]) or name != job["name"] or user != USER or account != ACCOUNT:
                raise ValueError("Slurm accounting ownership differs")
            if not elapsed.isdecimal():
                raise ValueError("Invalid Slurm elapsed time")
            records.append({"state": state.split()[0].rstrip("+"), "start": started, "elapsed_raw": int(elapsed)})
        return records

    def submit(self, kind: str, queue: list[dict]) -> None:
        row = self.state["remote"][kind]
        if row.get("status") not in (None, "retry") or not quota_available(queue, 0 if kind == "prep" else 2):
            return
        attempt = row.get("attempt", 0) + 1
        if attempt > MAX_ATTEMPTS:
            row["status"] = "failed"
            self.save()
            return
        row.update(status="submitting", name=f"{self.prefix}{kind}-{attempt}", attempt=attempt, submitted_at=time.time())
        self.save()
        deployment = read_json(self.root / "ubai/deployment.json")
        script = "vit_comparison_prep.sbatch" if kind == "prep" else "vit_comparison_task.sbatch"
        arguments = [
            "sbatch", "--parsable", "--account=" + ACCOUNT, "--job-name=" + row["name"],
            "--output=" + self.remote_root + "/ubai/" + row["name"] + "-%j.log",
            "--export=ALL,EXPERIMENT_SOURCE=" + deployment["host_source_root"] +
            ",EXPERIMENT_DEPLOYMENT=" + self.remote_root + "/ubai/deployment.json",
            deployment["host_source_root"] + "/scripts/experiments/ubai/" + script,
        ]
        response = self.remote(arguments).strip().split(";")[0]
        if not response.isdecimal():
            raise ValueError("Ambiguous Slurm submission response")
        row.update(status="submitted", job_id=response)
        self.save()
        self.event("ubai_submitted", kind=kind, job_id=response)

    def recover_submission(self, kind: str, queue: list[dict]) -> None:
        row = self.state["remote"][kind]
        if row.get("status") != "submitting":
            return
        matches = [job for job in queue if job["name"] == row["name"]]
        if not matches:
            # A short job may have finished while its submission response was lost.
            day = time.strftime("%Y-%m-%d", time.gmtime(row["submitted_at"] - 86400))
            text = self.remote(["sacct", "-n", "-X", "--user=" + USER, "--account=" + ACCOUNT,
                                "--name=" + row["name"], "--starttime=" + day,
                                "--format=JobIDRaw,JobName%200,User%64,Account%64", "--parsable2"])
            found = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                fields = [item.strip() for item in line.rstrip("|").split("|")]
                if (len(fields) != 4 or not fields[0].isdecimal() or fields[1:] != [row["name"], USER, ACCOUNT]):
                    raise ValueError("Recovered Slurm accounting identity differs")
                found.append(fields[0])
            if len(found) == 1:
                row.update(status="submitted", job_id=found[0])
                self.save()
                return
        if len(matches) != 1:
            raise RuntimeError("An interrupted Slurm submission needs an unambiguous job identity before resuming")
        row.update(status="submitted", job_id=matches[0]["job_id"])
        info = parse_slurm_job(self.remote(["scontrol", "show", "job", "-o", str(row["job_id"])]))
        validate_job_identity(info, row)
        self.save()

    def transfer_admissions(self) -> None:
        names = ["experiment.json"]
        for key in REMOTE_MODELS:
            if not self.admitted(key):
                raise ValueError("Both remote models require completed local admission")
            record = read_json(self.root / "admissions" / (key + ".json"))
            names.append(f"admissions/{key}.json")
            for run_id in record["result_sha256"]:
                task = read_json(self.root / "tasks" / (run_id + ".json"))
                names.extend([f"tasks/{run_id}.json", task["result_file"], task["log_file"], task["calibration_file"]])
        self.transfer(list(dict.fromkeys(names)))

    def sync_remote(self) -> None:
        names = ["ubai/prep-result.json", "ubai/prep-environment.json"]
        for key in REMOTE_MODELS:
            if self.state["models"][key]["owner"] != "ubai":
                continue
            for kind in ("collect", "dense", "spiking"):
                run_id = f"{key}_{kind}"
                names.extend([f"tasks/{run_id}.json", f"logs/{run_id}.log", f"results/{run_id}.json"])
            names.append(f"calibration/{key}.json")
        admission_path = self.root / "admissions/imagenet_vit_base.json"
        if admission_path.exists():
            batch = read_json(admission_path)["batch_size"]
            run_id = f"imagenet_vit_base_environment_spiking_bs{batch}_ubai"
            names.extend([f"tasks/{run_id}.json", f"logs/{run_id}.log", f"results/{run_id}.json"])
        for kind in ("prep", "pair"):
            job = self.state["remote"][kind]
            if job.get("job_id"):
                names.append(f"ubai/{job['name']}-{job['job_id']}.log")
        incoming = Path(tempfile.mkdtemp(prefix="incoming-", dir=self.root / "ubai"))
        try:
            self.transfer(names, pull=True, target=incoming)
        except BaseException:
            shutil.rmtree(incoming)
            raise
        accepted_results = set()
        for result_path in (incoming / "results").glob("*.json"):
            try:
                result = read_json(result_path)
                task = read_json(incoming / "tasks" / (result["run_id"] + ".json"))
            except (OSError, json.JSONDecodeError):
                continue
            validate_task(task, self.experiment)
            if task["calibration_file"] and not (incoming / task["calibration_file"]).exists():
                # Environment replay intentionally uses the copied local smoke table.
                original = self.root / task["calibration_file"]
                if not original.exists():
                    continue
                (incoming / task["calibration_file"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, incoming / task["calibration_file"])
            log = incoming / task["log_file"]
            table = incoming / task["calibration_file"] if task["calibration_file"] else None
            if (not log.exists() or identity.sha256_file(log) != result.get("log_sha256")
                    or (table is not None
                        and identity.sha256_file(table) != result.get("calibration_sha256"))):
                # A result may appear after its earlier live log snapshot was copied.
                # Wait for one internally consistent snapshot instead of accepting it.
                continue
            validate_result(task, result, self.experiment, incoming)
            accepted_results.add(result["run_id"])
        for run_id in list(accepted_results):
            task = read_json(incoming / "tasks" / (run_id + ".json"))
            if task["kind"] != "spiking":
                continue
            collection = task["model_key"] + "_collect"
            if collection in accepted_results:
                continue
            prior = self.root / "results" / (collection + ".json")
            table = self.root / task["calibration_file"]
            if (not prior.exists() or not table.exists()
                    or identity.sha256_file(table) != task["calibration_sha256"]):
                accepted_results.remove(run_id)
                continue
            validate_result(read_json(self.root / "tasks" / (collection + ".json")), read_json(prior),
                            self.experiment, self.root)
        # Install dependencies before results so no accepted JSON points at a partial log.
        for name in sorted(set(names), key=lambda name: name.startswith("results/")):
            source, destination = incoming / name, self.root / name
            if not source.exists():
                continue
            if name.startswith("results/") and source.stem not in accepted_results:
                continue
            if name.startswith("calibration/") and source.stem + "_collect" not in accepted_results:
                continue
            if (destination.exists()
                    and identity.sha256_file(source) == identity.sha256_file(destination)):
                continue
            if destination.exists() and not name.endswith(".log"):
                raise ValueError("Remote transfer would replace an immutable local result")
            if name.startswith("logs/") and (self.root / "results" / (destination.stem + ".json")).exists():
                raise ValueError("Remote transfer would replace an accepted local log")
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
            os.close(descriptor)
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        shutil.rmtree(incoming)
        for key in REMOTE_MODELS:
            if self.state["models"][key]["owner"] == "ubai" and self.model_complete(key):
                self.state["models"][key]["status"] = "complete"
        self.save()

    def prep_ready(self) -> bool:
        path = self.root / "ubai/prep-result.json"
        if not path.exists():
            return False
        report = read_json(path)
        if (report.get("state") != "verified" or report.get("python_version") != "3.12.13"
                or report.get("source_commit") != self.experiment["source_commit"]
                or report.get("deployment_sha256")
                != identity.sha256_file(self.root / "ubai/deployment.json")):
            raise ValueError("UBAI preparation identity differs")
        self.state["remote"]["prep"]["status"] = "complete"
        return True

    def tick_handoff(self, queue: list[dict], free_gpus: list[int]) -> None:
        if len(set(free_gpus)) != len(free_gpus) or not set(free_gpus).issubset(LOCAL_GPUS):
            raise ValueError("Handoff requires distinct authorized local GPU slots")
        job = self.state["remote"]["pair"]
        if job.get("status") not in {"submitted", "cancelling"}:
            return
        queued = [row for row in queue if str(row["job_id"]) == str(job["job_id"])]
        if not queued:
            if job.get("status") != "cancelling":
                return
            records = self.accounting(job)
            if not records or any(record["state"] not in TERMINAL for record in records):
                return
            if not never_started(records):
                job.update(status="retry", handoff_rejected=True)
                self.save()
                return
            for key in REMOTE_MODELS:
                self.state["models"][key].update(owner="local", status="ready")
            job.update(status="cancelled_for_local", accounting=records)
            self.save()
            self.event("ubai_pending_pair_moved_local", job_id=job["job_id"])
            return
        if len(queued) != 1 or queued[0]["name"] != job["name"]:
            raise ValueError("Ambiguous queued comparison identity")
        if queued[0].get("gpus") != 2:
            raise ValueError("The queued pair no longer reserves exactly two GPUs")
        if job.get("status") != "cancelling" and (
                queued[0]["state"] != "PENDING" or len(free_gpus) < 2
                or time.time() - job["submitted_at"] < 60):
            return
        info = parse_slurm_job(self.remote(["scontrol", "show", "job", "-o", str(job["job_id"])]))
        validate_job_identity(info, job)
        if info["JobState"] != "PENDING":
            if job.get("hold_requested") and info["Priority"] == "0" and info["Reason"] != "JobHeldAdmin":
                self.remote(["scontrol", "release", str(job["job_id"])])
            job.update(status="submitted", hold_requested=False)
            self.save()
            return
        if not job.get("hold_requested"):
            if info["Priority"] == "0" or info["Reason"].startswith("JobHeld"):
                return
            job.update(status="cancelling", hold_requested=True)
            self.save()
            self.remote(["scontrol", "hold", str(job["job_id"])])
            info = parse_slurm_job(self.remote(["scontrol", "show", "job", "-o", str(job["job_id"])]))
            validate_job_identity(info, job)
            if info["JobState"] != "PENDING":
                if info["Priority"] == "0" and info["Reason"] != "JobHeldAdmin":
                    self.remote(["scontrol", "release", str(job["job_id"])])
                job.update(status="submitted", hold_requested=False)
                self.save()
                return
        if info["JobState"] == "PENDING" and info["Priority"] == "0" and info["Reason"] == "JobHeldUser":
            self.remote(["scancel", "--state=PENDING", "--user=" + USER, "--name=" + job["name"], str(job["job_id"])])

    def poll_remote(self, queue: list[dict]) -> None:
        for kind in ("prep", "pair"):
            self.recover_submission(kind, queue)
            job = self.state["remote"][kind]
            if job.get("status") != "submitted" or any(str(row["job_id"]) == str(job["job_id"]) for row in queue):
                continue
            records = self.accounting(job)
            if not records or any(record["state"] not in TERMINAL for record in records):
                continue
            complete = self.prep_ready() if kind == "prep" else all(self.model_complete(key) for key in REMOTE_MODELS)
            job["status"] = "complete" if complete else "failed" if job["attempt"] >= MAX_ATTEMPTS else "retry"
            job["accounting"] = records
            self.save()

    def update_summary(self) -> None:
        from scripts.experiments.run_vit_comparison import summarize
        files = sorted((self.root / "results").glob("*.json"))
        summary_identity = tuple((str(path), identity.sha256_file(path)) for path in files)
        if summary_identity != self.last_summary:
            summarize(self.root, self.experiment)
            self.last_summary = summary_identity

    def step(self) -> bool:
        self.poll_local()
        free = self.free_gpus()
        for key in (*LOCAL_MODELS, *REMOTE_MODELS):
            row = self.state["models"][key]
            if not free or key in self.state["local"] or row["status"] in {"failed", "complete"}:
                continue
            if not self.admitted(key):
                self.launch(key, "admit", free.pop(0))
            elif row["owner"] == "local":
                self.launch(key, "pipeline", free.pop(0))
        if not all(self.state["models"][key]["owner"] == "local" for key in REMOTE_MODELS):
            try:
                self.stage()
                queue = self.queue()
                self.sync_remote()
                self.poll_remote(queue)
                self.tick_handoff(queue, self.free_gpus())
                if not self.prep_ready():
                    self.submit("prep", queue)
                elif all(self.admitted(key) for key in REMOTE_MODELS):
                    if self.state["remote"]["pair"].get("status") in (None, "retry"):
                        self.transfer_admissions()
                        self.submit("pair", queue)
                if self.state["remote"]["prep"].get("status") == "failed":
                    raise RuntimeError("UBAI preparation exhausted its technical retries")
                if self.state["remote"]["pair"].get("status") == "failed":
                    for key in REMOTE_MODELS:
                        if self.state["models"][key]["status"] != "complete":
                            self.state["models"][key]["status"] = "failed"
            except (subprocess.SubprocessError, OSError) as error:
                self.state["remote"]["last_error"] = str(error)
                self.event("ubai_connection_retry", error=str(error))
        self.update_summary()
        self.save()
        if all(row["status"] == "complete" for row in self.state["models"].values()):
            from scripts.experiments.run_vit_comparison import summarize
            summarize(self.root, self.experiment, require_complete=True)
            self.state["phase"] = "complete"
            self.save()
            self.event("comparison_complete")
            return True
        if all(row["status"] in {"failed", "complete"} for row in self.state["models"].values()):
            self.state["phase"] = "needs_attention"
            self.save()
            raise RuntimeError("Comparison has failed model pipelines; completed results are preserved")
        return False


def run_controller(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(root)
        controller.event("controller_started", pid=os.getpid(), local_gpu_ids=list(LOCAL_GPUS))
        while not controller.step():
            time.sleep(10)


def run(root: Path) -> None:
    run_controller(root)
