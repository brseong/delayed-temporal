"""BrainScaleS-2 TTFS neuron-pooling hardware validation API."""

from .config import (
    BrainScaleS2PoolConfig,
    CADCDiagnosticResult,
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
from .toy import (
    ARCHITECTURES,
    ConvertedToyModel,
    ToyMLP,
    convert_float_model,
    load_dataset_bundle,
)
from .toy_pooling import (
    GroupedHardwarePoolBackend,
    MockToyPoolBackend,
    ReplayToyPoolBackend,
    ToyPoolConfig,
    ToyPoolResult,
    resolve_grouped_physical_coordinates,
)
from .hagen import HagenConfig, HagenPWMBackend

__all__ = [
    "BrainScaleS2PoolConfig",
    "CADCDiagnosticResult",
    "PoolRunResult",
    "TTFSHardwareEncoding",
    "encode_potential_for_brainscales2",
    "BrainScaleS2PoolBackend",
    "MockPoolBackend",
    "PoolBackend",
    "calibration_sha256",
    "resolve_physical_neuron_indices",
    "ARCHITECTURES",
    "ConvertedToyModel",
    "ToyMLP",
    "convert_float_model",
    "load_dataset_bundle",
    "GroupedHardwarePoolBackend",
    "MockToyPoolBackend",
    "ReplayToyPoolBackend",
    "ToyPoolConfig",
    "ToyPoolResult",
    "resolve_grouped_physical_coordinates",
    "HagenConfig",
    "HagenPWMBackend",
]
