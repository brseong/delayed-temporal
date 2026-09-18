"""Private process worker for memory-bounded BrainScaleS-2 PyNN acquisition."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend


def _setup_hardware_client() -> None:
    demos_root = Path(
        os.environ.get("BSS2_DEMOS_ROOT", "/tmp/brainscales2-demos")
    )
    if demos_root.is_dir():
        sys.path.insert(0, str(demos_root))
    try:
        from _static.common.helpers import setup_hardware_client
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyNN worker could not import the official hardware client helper; "
            "set BSS2_DEMOS_ROOT to the brainscales2-demos checkout"
        ) from error
    setup_hardware_client()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: primitive_pynn_worker.py REQUEST RESPONSE")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    _setup_hardware_client()
    backend = PrimitiveHardwareBackend()
    first, count, precharge, metadata = backend._run_pynn_code_chunks(
        request["primitive"],
        request["stage"],
        int(request["code"]),
        request["config"],
        repeats=int(request["repeats"]),
    )
    torch.save(
        {
            "first": first,
            "count": count,
            "precharge": precharge,
            "metadata": metadata,
        },
        response_path,
    )


if __name__ == "__main__":
    main()
