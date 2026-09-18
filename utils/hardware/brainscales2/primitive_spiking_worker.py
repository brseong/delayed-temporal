"""Private process worker for memory-bounded spiking acquisition."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend
from utils.hardware.brainscales2.primitive_pynn_worker import _setup_hardware_client


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: primitive_spiking_worker.py REQUEST RESPONSE")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    _setup_hardware_client()

    import hxtorch
    import hxtorch.spiking as hxsnn

    initialized = False
    try:
        hxtorch.init_hardware()
        initialized = True
        backend = PrimitiveHardwareBackend()
        (
            baseline,
            observed,
            spike_count,
            saturated,
            calibration_loader,
            batch_count,
        ) = backend._run_psi_ne_chunk(
            hxsnn,
            request["config"],
            input_times=request["input_times"],
            runtime_steps=int(request["runtime_steps"]),
            observation_step=int(request["observation_step"]),
            repeats=int(request["repeats"]),
        )
        identifier = hxtorch.get_unique_identifier()
        chip_identifier = (
            [str(item) for item in identifier]
            if isinstance(identifier, (tuple, list))
            else [str(identifier)]
        )
        torch.save(
            {
                "baseline": baseline,
                "observed": observed,
                "spike_count": spike_count,
                "saturated": saturated,
                "calibration_loader": calibration_loader,
                "batch_count": batch_count,
                "metadata": {
                    "chip_identifier": chip_identifier,
                    "hxtorch_version": getattr(hxtorch, "__version__", "unknown"),
                },
            },
            response_path,
        )
    finally:
        if initialized:
            hxtorch.release_hardware()


if __name__ == "__main__":
    main()
