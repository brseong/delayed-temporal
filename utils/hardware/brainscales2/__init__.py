"""BrainScaleS-2 TTFS neuron-pooling hardware validation API."""

from .config import (
    BrainScaleS2PoolConfig,
    PoolRunResult,
    TTFSHardwareEncoding,
)
from .encoding import encode_potential_for_brainscales2
from .backend import (
    BrainScaleS2PoolBackend,
    MockPoolBackend,
    PoolBackend,
    calibration_sha256,
    resolve_physical_neuron_indices,
)

__all__ = [
    "BrainScaleS2PoolConfig",
    "PoolRunResult",
    "TTFSHardwareEncoding",
    "encode_potential_for_brainscales2",
    "BrainScaleS2PoolBackend",
    "MockPoolBackend",
    "PoolBackend",
    "calibration_sha256",
    "resolve_physical_neuron_indices",
]
