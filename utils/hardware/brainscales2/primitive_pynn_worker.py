"""Private process worker for memory-bounded BrainScaleS-2 PyNN acquisition."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend


class _WorkerInterrupt(BaseException):
    """Unwind the worker through its backend release path."""


def _install_cleanup_interrupt_handler() -> None:
    """Unwind once into backend cleanup and ignore later interrupt signals."""

    def interrupt(_signum, _frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise _WorkerInterrupt("hardware worker interrupted for bounded cleanup")

    signal.signal(signal.SIGINT, interrupt)


def _reuse_configured_hardware_endpoint() -> bool:
    """Keep a complete notebook-selected Quiggeldy endpoint in child workers."""
    if not (
        os.environ.get("QUIGGELDY_IP")
        and os.environ.get("QUIGGELDY_PORT")
    ):
        return False
    os.environ["QUIGGELDY_ENABLED"] = "1"
    username = os.environ.get("JUPYTERHUB_USER") or os.environ.get("USER")
    if username:
        os.environ["QUIGGELDY_USER_NO_MUNGE"] = username
    return True


def _setup_hardware_client() -> None:
    if _reuse_configured_hardware_endpoint():
        return
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
    _install_cleanup_interrupt_handler()
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
