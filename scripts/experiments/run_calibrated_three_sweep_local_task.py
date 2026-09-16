"""Delegate to the frozen evaluator with the approved local GPU device set."""

from __future__ import annotations

import argparse
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from typing import Any


CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from scripts.runtime import identity as runtime_identity

WORKER_PATH = "scripts/experiments/run_calibrated_three_sweep_task.py"
COMMON_PATH = "scripts/experiments/calibrated_three_sweeps.py"
LOCAL_GPU_IDS = (4, 5, 6, 7)
SUPPORTED_LOCAL_GPU_IDS = tuple(range(8))
TEMPORARY_TAG = "vit_base_calibrated_theta_rt_ratio_float64_bounds3_v1"
TEMPORARY_SOURCE_COMMIT = "36615ab4390f9817e3af0e4c4a6f840fc6bd57ee"


def clean_head(source: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head) or dirty:
        raise ValueError("A clean committed checkout is required")
    return head


def check_control_identity(output_root: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    head = clean_head(CONTROL_ROOT)
    identity_path = output_root / "controllers" / f"{head}.json"
    identity = json.loads(identity_path.read_text())
    if identity.get("source_commit") != head or Path(identity.get("source_root", "")).resolve() != CONTROL_ROOT:
        raise ValueError("Local wrapper controller identity mismatch")
    if identity.get("evaluator_source_commit") != experiment["source_commit"]:
        raise ValueError("Controller refers to another frozen experiment source")
    if identity.get("local_gpu_ids") != list(LOCAL_GPU_IDS):
        raise ValueError("The controller must record the approved local GPU set")
    if identity.get("supported_local_gpu_ids") != list(SUPPORTED_LOCAL_GPU_IDS):
        raise ValueError("The controller must record its supported local GPU set")
    if identity.get("local_worker_sha256") != runtime_identity.sha256_file(Path(__file__)):
        raise ValueError("Local wrapper content differs from the controller record")
    controller_path = CONTROL_ROOT / "scripts/experiments/run_calibrated_three_sweeps.py"
    if identity.get("controller_sha256") != runtime_identity.sha256_file(controller_path):
        raise ValueError("Controller content differs from its recorded identity")
    return identity


def load_frozen_worker(experiment: dict[str, Any]) -> ModuleType:
    source = Path(experiment["source_root"]).resolve()
    if clean_head(source) != experiment["source_commit"]:
        raise ValueError("Frozen evaluator source commit mismatch")
    for relative in (WORKER_PATH, COMMON_PATH):
        expected = experiment["runtime_sha256"][relative]
        if (not re.fullmatch(r"[0-9a-f]{64}", expected)
                or runtime_identity.sha256_file(source / relative) != expected):
            raise ValueError(f"Frozen experiment source content mismatch: {relative}")
    # This entry point imports no repository code before establishing the source.
    for name in ("scripts", "scripts.experiments", "scripts.experiments.calibrated_three_sweeps"):
        loaded = sys.modules.get(name)
        location = getattr(loaded, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(source):
            raise ValueError("Previously imported experiment code belongs to another checkout")
    sys.path[:0] = [str(source), str(source / "src/transformers/src"), str(source / "src/spikingjelly")]
    spec = importlib.util.spec_from_file_location("_frozen_calibrated_three_sweep_worker", source / WORKER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the frozen experiment worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_temporary_campaign(experiment: dict[str, Any], output_root: Path) -> None:
    if (experiment.get("tag") != TEMPORARY_TAG or output_root.resolve().name != TEMPORARY_TAG
            or experiment.get("source_commit") != TEMPORARY_SOURCE_COMMIT):
        raise ValueError("Additional local GPU devices are authorized only for the current frozen campaign")


def require_local_gpu(experiment: dict[str, Any], host_label: str, *, allow_temporary: bool = False) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = SUPPORTED_LOCAL_GPU_IDS if allow_temporary else LOCAL_GPU_IDS
    if host_label != "local" or visible not in {str(index) for index in allowed}:
        raise ValueError("Local evaluation requires one physical GPU from the approved local set")
    probe = subprocess.check_output([
        experiment["python_bin"], "-c", "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),"
        "'model':torch.cuda.get_device_name(0) if torch.cuda.device_count() else ''}))"], text=True)
    data = json.loads(probe)
    if data["count"] != 1 or "RTX A6000" not in data["model"]:
        raise ValueError("Exactly one RTX A6000 GPU is required")
    return data["model"]


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host-label", choices=("local",), required=True)
    parser.add_argument("--temporary-local-gpus", action="store_true")
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    check_control_identity(args.output_root, experiment)
    if args.temporary_local_gpus:
        validate_temporary_campaign(experiment, args.output_root)
    original_path = list(sys.path)
    original_argv = sys.argv
    try:
        worker = load_frozen_worker(experiment)
        original_require_gpu = worker.require_gpu
        worker.require_gpu = partial(require_local_gpu, allow_temporary=True) if args.temporary_local_gpus else require_local_gpu
        if args.temporary_local_gpus:
            sys.argv = [argument for argument in original_argv if argument != "--temporary-local-gpus"]
        try:
            # Only the wrapper-specific option is removed; task arguments stay unchanged.
            return worker.main()
        finally:
            worker.require_gpu = original_require_gpu
    finally:
        sys.path[:] = original_path
        sys.argv = original_argv


if __name__ == "__main__":
    main()
