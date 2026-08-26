"""Configuration and result contracts for BrainScaleS-2 TTFS pooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import math

import torch

from utils.transforms.types import PotentialBounds, TimeBounds


EncodingKind = Literal["identity", "log"]
PlacementMode = Literal["same-quadrant", "cross-quadrant"]
RoutingMode = Literal["broadcast", "independent"]


def _require_finite_positive(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


@dataclass(frozen=True)
class BrainScaleS2PoolConfig:
    """Complete configuration for one physical or mock neuron-pool run."""

    encoding: EncodingKind = "identity"
    dt_s: float = 1.0e-6
    input_early_s: float = 5.0e-6
    input_late_s: float = 25.0e-6
    observation_deadline_s: float = 60.0e-6
    inter_batch_wait_s: float = 50.0e-6
    project_tau_s: float = 1.0

    tau_mem_s: float = 20.0e-6
    tau_syn_s: float = 1.0e-6
    leak: float = 80.0
    reset: float = 80.0
    threshold: float = 125.0
    refractory_time_s: float = 1.0e-6
    i_synin_gm: float = 500.0
    synapse_dac_bias: float = 600.0
    synaptic_weight: float = 63.0

    pool_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    placements: tuple[PlacementMode, ...] = (
        "same-quadrant",
        "cross-quadrant",
    )
    routings: tuple[RoutingMode, ...] = ("broadcast", "independent")
    trials: int = 256
    seed: int = 0
    calibration_path: Path | None = None
    allow_environment_calibration: bool = False
    raw_time_scale_s: float | None = None

    # The mock backend separates persistent offsets, trial-shared disturbances,
    # and neuron-local trial noise so the analysis can be exercised without BSS-2.
    mock_response_delay_s: float = 5.0e-6
    mock_static_std_s: float = 0.5e-6
    mock_shared_std_s: float = 0.25e-6
    mock_local_std_s: float = 0.8e-6
    mock_miss_probability: float = 0.01

    def __post_init__(self) -> None:
        if self.encoding not in ("identity", "log"):
            raise ValueError("encoding must be 'identity' or 'log'")

        for name in (
            "dt_s",
            "input_early_s",
            "input_late_s",
            "observation_deadline_s",
            "project_tau_s",
            "tau_mem_s",
            "tau_syn_s",
            "refractory_time_s",
        ):
            _require_finite_positive(name, getattr(self, name))

        if not math.isfinite(float(self.inter_batch_wait_s)) or self.inter_batch_wait_s < 0.0:
            raise ValueError("inter_batch_wait_s must be finite and non-negative")
        if not self.input_early_s < self.input_late_s:
            raise ValueError("input_early_s must be smaller than input_late_s")
        if not self.input_late_s < self.observation_deadline_s:
            raise ValueError(
                "input_late_s must be strictly before observation_deadline_s"
            )
        if self.trials < 2:
            raise ValueError("trials must be at least two for calibration/evaluation splitting")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not self.pool_sizes or any(size <= 0 or size > 128 for size in self.pool_sizes):
            raise ValueError("pool_sizes must contain integers in [1, 128]")
        if len(set(self.pool_sizes)) != len(self.pool_sizes):
            raise ValueError("pool_sizes must not contain duplicates")
        if any(value not in ("same-quadrant", "cross-quadrant") for value in self.placements):
            raise ValueError("unsupported placement mode")
        if any(value not in ("broadcast", "independent") for value in self.routings):
            raise ValueError("unsupported routing mode")
        if not 0.0 <= self.mock_miss_probability <= 1.0:
            raise ValueError("mock_miss_probability must lie in [0, 1]")
        if self.raw_time_scale_s is not None:
            _require_finite_positive("raw_time_scale_s", self.raw_time_scale_s)

    @property
    def runtime_steps(self) -> int:
        """Number of dense input bins required to include the deadline."""
        return math.ceil(self.observation_deadline_s / self.dt_s) + 1

    def require_reproducible_calibration(self) -> None:
        """Reject an unpinned hardware run unless explicitly allowed for smoke tests."""
        if self.calibration_path is None and not self.allow_environment_calibration:
            raise ValueError(
                "hardware runs require calibration_path; use "
                "allow_environment_calibration only for smoke tests"
            )
        if self.calibration_path is not None and not self.calibration_path.is_file():
            raise FileNotFoundError(self.calibration_path)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the configuration."""
        payload = asdict(self)
        payload["calibration_path"] = (
            str(self.calibration_path) if self.calibration_path is not None else None
        )
        return payload


@dataclass(frozen=True)
class TTFSHardwareEncoding:
    """Nominal project codewords converted into quantized BSS-2 input events."""

    dense_spikes: torch.Tensor
    ideal_time_s: torch.Tensor
    injected_time_s: torch.Tensor
    original_shape: tuple[int, ...]
    source_domain: PotentialBounds
    source_time_domain: TimeBounds
    physical_time_domain: TimeBounds
    clamped_values: torch.Tensor
    clamp_mask: torch.Tensor
    encoding: EncodingKind
    routing: RoutingMode

    @property
    def sample_count(self) -> int:
        return self.injected_time_s.numel()


@dataclass
class PoolRunResult:
    """Raw first-spike observations for one pool/placement/routing condition."""

    first_spike_s: torch.Tensor
    fired: torch.Tensor
    spike_count: torch.Tensor
    nominal_input_s: torch.Tensor
    ideal_input_s: torch.Tensor
    physical_coordinates: tuple[int, ...]
    pool_size: int
    placement: PlacementMode
    routing: RoutingMode
    original_input_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = self.first_spike_s.shape
        if len(expected) != 3:
            raise ValueError(
                "first_spike_s must have shape [trial, sample, neuron]"
            )
        if self.fired.shape != expected or self.spike_count.shape != expected:
            raise ValueError("fired and spike_count must match first_spike_s")
        if expected[-1] != self.pool_size:
            raise ValueError("result neuron dimension must equal pool_size")
        if self.nominal_input_s.numel() != expected[1]:
            raise ValueError("nominal input count must match the sample dimension")
        if len(self.physical_coordinates) != self.pool_size:
            raise ValueError("one physical coordinate is required per neuron")

