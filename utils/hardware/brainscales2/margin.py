"""Calibration-only selection of a BrainScaleS-2 TTFS deadline margin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch


@dataclass(frozen=True)
class DeadlineMarginConfig:
    """Fixed candidate grid and unlabeled acceptance rule for deadline extension."""

    dt_s: float = 1.0e-6
    base_deadline_s: float = 60.0e-6
    diagnostic_deadline_s: float = 100.0e-6
    maximum_margin_s: float = 40.0e-6
    margin_step_s: float = 1.0e-6
    target_sample_miss_rate: float = 0.05
    confidence: float = 0.95
    bootstrap_iterations: int = 2_000
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "dt_s",
            "base_deadline_s",
            "diagnostic_deadline_s",
            "maximum_margin_s",
            "margin_step_s",
            "target_sample_miss_rate",
            "confidence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.base_deadline_s <= 0.0:
            raise ValueError("base_deadline_s must be positive")
        if self.diagnostic_deadline_s <= self.base_deadline_s:
            raise ValueError("diagnostic_deadline_s must exceed base_deadline_s")
        if self.maximum_margin_s < 0.0 or self.margin_step_s <= 0.0:
            raise ValueError("margin range must be non-negative with a positive step")
        if self.base_deadline_s + self.maximum_margin_s > self.diagnostic_deadline_s + 1e-15:
            raise ValueError("candidate margins exceed the diagnostic deadline")
        if not 0.0 < self.target_sample_miss_rate < 1.0:
            raise ValueError("target_sample_miss_rate must lie in (0, 1)")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0.5, 1)")
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        for name in (
            "base_deadline_s",
            "diagnostic_deadline_s",
            "maximum_margin_s",
            "margin_step_s",
        ):
            grid_units = float(getattr(self, name)) / self.dt_s
            if not math.isclose(grid_units, round(grid_units), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"{name} must align to the hardware dt grid")

    def candidate_margins(self) -> tuple[float, ...]:
        """Return an inclusive, numerically stable margin grid."""
        steps = int(math.floor(self.maximum_margin_s / self.margin_step_s + 1e-9))
        candidates = [index * self.margin_step_s for index in range(steps + 1)]
        if self.maximum_margin_s - candidates[-1] > 1e-15:
            candidates.append(self.maximum_margin_s)
        return tuple(float(value) for value in candidates)


@dataclass(frozen=True)
class DeadlineMarginObservation:
    """Raw M=1 diagnostic events for one placement and one hidden code tensor."""

    first_spike_s: torch.Tensor
    hidden_uint5: torch.Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.first_spike_s.ndim != 4 or self.first_spike_s.shape[-1] != 1:
            raise ValueError("deadline observations must have shape [trial, sample, logical, 1]")
        if self.hidden_uint5.shape != self.first_spike_s.shape[1:3]:
            raise ValueError("hidden UInt5 values do not match diagnostic events")
        if bool((self.hidden_uint5 < 0).any() or (self.hidden_uint5 > 31).any()):
            raise ValueError("hidden calibration values must lie in UInt5")


@dataclass(frozen=True)
class DeadlineMarginSelection:
    """Selected common margin and the complete calibration-only decision curve."""

    config: DeadlineMarginConfig
    selected_margin_s: float | None
    selected_deadline_s: float | None
    placements: tuple[str, ...]
    curve: tuple[dict[str, Any], ...]
    structural_floor: tuple[dict[str, Any], ...]

    @property
    def viable(self) -> bool:
        return self.selected_margin_s is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "selected_margin_s": self.selected_margin_s,
            "selected_deadline_s": self.selected_deadline_s,
            "placements": list(self.placements),
            "curve": list(self.curve),
            "structural_floor": list(self.structural_floor),
            "viable": self.viable,
        }


def _miss_matrix(
    observation: DeadlineMarginObservation,
    deadline_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = observation.hidden_uint5 > 0
    supported_samples = positive.any(dim=-1)
    first = observation.first_spike_s[..., 0]
    delivered = torch.isfinite(first) & (first <= deadline_s)
    any_positive_miss = ((~delivered) & positive.unsqueeze(0)).any(dim=-1)
    return any_positive_miss[:, supported_samples], supported_samples


def _hierarchical_bootstrap_upper(
    miss: torch.Tensor,
    *,
    confidence: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if miss.ndim != 2 or miss.shape[0] == 0 or miss.shape[1] == 0:
        return math.nan, math.nan
    miss_float = miss.to(torch.float64)
    point = float(miss_float.mean())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    estimates = torch.empty(iterations, dtype=torch.float64)
    trials, samples = miss.shape
    for iteration in range(iterations):
        trial_index = torch.randint(trials, (trials,), generator=generator)
        sample_index = torch.randint(samples, (samples,), generator=generator)
        estimates[iteration] = miss_float[
            trial_index.reshape(-1, 1), sample_index.reshape(1, -1)
        ].mean()
    upper = float(
        torch.quantile(
            estimates,
            torch.tensor(confidence, dtype=torch.float64),
        )
    )
    return point, upper


def select_deadline_margin(
    observations: dict[str, DeadlineMarginObservation],
    config: DeadlineMarginConfig,
) -> DeadlineMarginSelection:
    """Choose the smallest margin whose per-placement miss UCB meets the target."""
    if not observations:
        raise ValueError("at least one placement observation is required")
    placements = tuple(sorted(observations))
    curve: list[dict[str, Any]] = []
    structural_floor: list[dict[str, Any]] = []
    selected: float | None = None
    for margin_s in config.candidate_margins():
        deadline_s = config.base_deadline_s + margin_s
        candidate_passes = True
        for placement_index, placement in enumerate(placements):
            observation = observations[placement]
            miss, support = _miss_matrix(observation, deadline_s)
            rate, upper = _hierarchical_bootstrap_upper(
                miss,
                confidence=config.confidence,
                iterations=config.bootstrap_iterations,
                seed=config.seed + 10_000 * placement_index,
            )
            passed = math.isfinite(upper) and upper <= config.target_sample_miss_rate
            candidate_passes = candidate_passes and passed
            curve.append(
                {
                    "placement": placement,
                    "margin_s": margin_s,
                    "deadline_s": deadline_s,
                    "supported_samples": int(support.sum()),
                    "trials": int(observation.first_spike_s.shape[0]),
                    "sample_any_positive_miss_rate": rate,
                    "bootstrap_upper": upper,
                    "target_rate": config.target_sample_miss_rate,
                    "passed": passed,
                }
            )
        if selected is None and candidate_passes:
            selected = margin_s

    for placement_index, placement in enumerate(placements):
        observation = observations[placement]
        miss, support = _miss_matrix(observation, config.diagnostic_deadline_s)
        rate, upper = _hierarchical_bootstrap_upper(
            miss,
            confidence=config.confidence,
            iterations=config.bootstrap_iterations,
            seed=config.seed + 100_000 + placement_index,
        )
        structural_floor.append(
            {
                "placement": placement,
                "diagnostic_deadline_s": config.diagnostic_deadline_s,
                "supported_samples": int(support.sum()),
                "sample_any_positive_miss_rate": rate,
                "bootstrap_upper": upper,
            }
        )
    return DeadlineMarginSelection(
        config=config,
        selected_margin_s=selected,
        selected_deadline_s=(
            None if selected is None else config.base_deadline_s + selected
        ),
        placements=placements,
        curve=tuple(curve),
        structural_floor=tuple(structural_floor),
    )
