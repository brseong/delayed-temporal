"""Digital weight selection from separate physical calibration observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .backend import calibration_sha256
from .config import BrainScaleS2PoolConfig


def weight_context(config: BrainScaleS2PoolConfig) -> dict[str, Any]:
    """Bind weights to the pinned analog state and input schedule."""
    keys = (
        "dt_s", "input_early_s", "input_late_s", "inter_batch_wait_s",
        "input_fan_in", "tau_mem_s", "tau_syn_s", "leak", "reset",
        "threshold", "refractory_time_s", "i_synin_gm", "synapse_dac_bias",
    )
    return {
        **{key: getattr(config, key) for key in keys},
        "calibration_sha256": calibration_sha256(config.calibration_path),
    }


def condition_key(config: Any) -> str:
    return f"{config.mapping}_{config.logical_neurons}_{config.pool_size}_{config.placement}"


def delivery_statistics(
    first: torch.Tensor, count: torch.Tensor, nominal: torch.Tensor,
    *, deadline_s: float,
) -> dict[str, torch.Tensor]:
    """Measure each neuron across active input rows and quiet control rows."""
    if first.shape != count.shape or first.shape != nominal.shape or first.ndim != 2:
        raise ValueError("first, count, and nominal must match [batch, neuron]")
    active = torch.isfinite(nominal)
    denominator = active.sum(0).clamp_min(1)
    finite = torch.isfinite(first)
    premature = finite & (first < nominal - 1e-9)
    delivered = finite & ~premature & (first <= deadline_s)
    single = delivered & (count == 1)
    quiet = ~active
    rates = {
        "fired_rate": (delivered & active).sum(0) / denominator,
        "single_spike_rate": (single & active).sum(0) / denominator,
        "multi_spike_rate": ((count > 1) & active).sum(0) / denominator,
        "premature_spike_rate": (premature & active).sum(0) / denominator,
        "quiet_fired_rate": ((count > 0) & quiet).sum(0) / quiet.sum(0).clamp_min(1),
    }
    # Prevent high average delivery from hiding a failing late code.
    worst = torch.ones(first.shape[1], dtype=torch.float64)
    for column in range(first.shape[1]):
        times = nominal[active[:, column], column].unique()
        if times.numel() == 0:
            worst[column] = 0.0
        for time in times:
            rows = active[:, column] & (nominal[:, column] == time)
            worst[column] = min(float(worst[column]), float(single[rows, column].double().mean()))
    rates["minimum_code_single_spike_rate"] = worst
    return rates


def passing_neurons(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        (stats["single_spike_rate"] >= 0.95)
        & (stats["minimum_code_single_spike_rate"] >= 0.90)
        & (stats["multi_spike_rate"] <= 0.05)
        & (stats["premature_spike_rate"] <= 0.01)
        & (stats["quiet_fired_rate"] == 0)
    )


def select_neuron_weights(
    candidates: list[int], statistics: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the smallest passing weight, or retain a diagnostic best effort."""
    if not candidates or len(candidates) != len(statistics):
        raise ValueError("each candidate needs calibration statistics")
    passed = torch.stack([passing_neurons(stats) for stats in statistics])
    score = torch.stack([
        stats["single_spike_rate"] - 10 * (
            stats["multi_spike_rate"] + stats["premature_spike_rate"]
            + stats["quiet_fired_rate"]
        ) for stats in statistics
    ])
    best = score.argmax(0)
    for index in sorted(range(len(candidates)), key=lambda i: candidates[i], reverse=True):
        best = torch.where(passed[index], index, best)
    return torch.tensor(candidates, dtype=torch.int64)[best], passed.any(0)


def load_neuron_weights(
    config: BrainScaleS2PoolConfig, pool: Any,
    coordinates: torch.Tensor, chip_identifier: list[str] | None,
) -> torch.Tensor:
    """Reject a failed or mismatched calibration before programming weights."""
    path = Path(config.neuron_weight_calibration_path)
    checksum = calibration_sha256(path)
    if checksum != config.neuron_weight_calibration_sha256 or checksum is None:
        raise ValueError("neuron weight calibration checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("context") != weight_context(config):
        raise ValueError("neuron weight calibration context mismatch")
    if not chip_identifier or payload.get("chip_identifier") != chip_identifier:
        raise ValueError("neuron weight calibration chip mismatch")
    condition = payload["conditions"].get(condition_key(pool))
    if condition is None or not condition.get("viable"):
        raise ValueError("no validated neuron weights for this physical graph")
    if condition.get("physical_coordinates") != coordinates.tolist():
        raise ValueError("neuron weight calibration placement mismatch")
    if config.observation_deadline_s < payload["validation_deadline_s"]:
        raise ValueError("requested deadline is shorter than weight validation")
    values = torch.tensor(condition["neuron_synaptic_weights"], dtype=torch.float64)
    expected = coordinates.numel() if pool.mapping == "dedicated" else pool.pool_size
    if values.shape != (expected,) or not bool(torch.isfinite(values).all()):
        raise ValueError("invalid neuron weight dimensions or nonfinite values")
    if bool(((values < 0) | (values > 63) | (values != values.round())).any()):
        raise ValueError("invalid digital neuron weights")
    return values
