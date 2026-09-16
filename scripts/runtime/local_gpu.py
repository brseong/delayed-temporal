"""Local NVIDIA GPU discovery and admission independent of any campaign."""
from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any, Mapping


DEFAULT_LOCAL_GPUS = (4, 5, 6, 7)
DEFAULT_ADMISSION_POLICY = {
    "max_memory_used_mib": 1024.0,
    "max_utilization_gpu_percent": 5.0,
}


def parse_gpu_occupancy(
    devices: str,
    applications: str,
    gpu_ids: tuple[int, ...] = DEFAULT_LOCAL_GPUS,
) -> dict[int, set[int]]:
    uuids: dict[str, int] = {}
    for line in devices.strip().splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        uuids[uuid] = int(index)
    if not set(gpu_ids).issubset(uuids.values()):
        raise ValueError("Required local GPU indices are missing")
    occupied = {index: set() for index in gpu_ids}
    for line in applications.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(",")]
        if uuid not in uuids or not pid.isdecimal():
            raise ValueError("Incomplete GPU occupancy information")
        if uuids[uuid] in occupied:
            # Host PIDs can belong to another container and need not exist in /proc here.
            occupied[uuids[uuid]].add(int(pid))
    return occupied


def _query_nvidia_smi(fields: str, kind: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )


def gpu_occupancy(
    *, gpu_ids: tuple[int, ...] = DEFAULT_LOCAL_GPUS,
) -> dict[int, set[int]]:
    return parse_gpu_occupancy(
        _query_nvidia_smi("index,uuid", "gpu"),
        _query_nvidia_smi("gpu_uuid,pid", "compute-apps"),
        gpu_ids,
    )


def parse_gpu_activity(
    devices: str,
    applications: str,
    gpu_ids: tuple[int, ...] = DEFAULT_LOCAL_GPUS,
) -> dict[int, dict[str, Any]]:
    identities: list[str] = []
    activity: dict[int, dict[str, Any]] = {}
    seen_uuids: set[str] = set()
    for line in devices.strip().splitlines():
        index_text, uuid, memory_text, utilization_text = [part.strip() for part in line.split(",")]
        index, memory, utilization = int(index_text), float(memory_text), float(utilization_text)
        if index in activity or uuid in seen_uuids or not uuid:
            raise ValueError("Duplicate or missing GPU identity")
        if (
            not math.isfinite(memory)
            or memory < 0
            or not math.isfinite(utilization)
            or not 0 <= utilization <= 100
        ):
            raise ValueError("GPU memory and utilization must be finite and valid")
        activity[index] = {
            "gpu_uuid": uuid,
            "memory_used_mib": memory,
            "utilization_gpu_percent": utilization,
        }
        identities.append(f"{index},{uuid}")
        seen_uuids.add(uuid)
    pids = parse_gpu_occupancy("\n".join(identities), applications, gpu_ids)
    return {index: {**activity[index], "pids": sorted(pids[index])} for index in gpu_ids}


def gpu_activity(
    *, gpu_ids: tuple[int, ...] = DEFAULT_LOCAL_GPUS,
) -> dict[int, dict[str, Any]]:
    return parse_gpu_activity(
        _query_nvidia_smi("index,uuid,memory.used,utilization.gpu", "gpu"),
        _query_nvidia_smi("gpu_uuid,pid", "compute-apps"),
        gpu_ids,
    )


def gpu_available(
    sample: Mapping[str, Any],
    policy: Mapping[str, float] = DEFAULT_ADMISSION_POLICY,
) -> bool:
    try:
        memory = float(sample["memory_used_mib"])
        utilization = float(sample["utilization_gpu_percent"])
        max_memory = float(policy["max_memory_used_mib"])
        max_utilization = float(policy["max_utilization_gpu_percent"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(memory)
        and math.isfinite(utilization)
        and math.isfinite(max_memory)
        and math.isfinite(max_utilization)
        and 0 <= memory <= max_memory
        and 0 <= utilization <= max_utilization
    )


def require_single_gpu(
    python_bin: str,
    host_label: str,
    *,
    allowed_local_gpus: tuple[int, ...] = DEFAULT_LOCAL_GPUS,
    required_model: str = "RTX A6000",
) -> str:
    """Validate one allocated GPU and the host-specific allocation boundary."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal() or int(visible) not in allowed_local_gpus:
        if host_label == "local":
            raise ValueError("Exactly one approved local GPU must be visible")
        if not visible or "," in visible or visible in {"-1", "all", "none"}:
            raise ValueError("Exactly one allocated GPU must be visible")
    if host_label == "ubai" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("UBAI evaluation requires a Slurm allocation")
    probe = subprocess.check_output([
        python_bin,
        "-c",
        "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),"
        "'model':torch.cuda.get_device_name(0) if torch.cuda.device_count() else ''}))",
    ], text=True)
    data = json.loads(probe)
    if data["count"] != 1 or required_model not in data["model"]:
        raise ValueError(f"Exactly one {required_model} GPU is required")
    return data["model"]
