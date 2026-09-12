"""Physical threshold selection and fixed, nested neuron placement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch

from .backend import calibration_sha256
from .neuron_weights import weight_context


def passing_threshold_neurons(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        (stats["single_spike_rate"] >= 0.99)
        & (stats["minimum_code_single_spike_rate"] >= 0.95)
        & (stats["multi_spike_rate"] <= 0.01)
        & (stats["premature_spike_rate"] <= 0.01)
        & (stats["quiet_fired_rate"] <= 0.01)
    )


def allocate_validated_coordinates(good: torch.Tensor) -> dict[str, list[list[int]]]:
    """Select 30 pools of 16; smaller pools use prefixes of these rows."""
    if good.shape != (512,) or good.dtype != torch.bool:
        raise ValueError("expected one boolean per physical neuron")
    available = [torch.where(good[q * 128:(q + 1) * 128])[0].add(q * 128).tolist()
                 for q in range(4)]
    capacity = [len(values) // 16 for values in available]
    if sum(capacity) < 30 or any(len(values) < 120 for values in available):
        raise ValueError(f"insufficient quadrant capacity: {[len(v) for v in available]}")
    used = [0] * 4
    local = []
    for _ in range(30):
        q = min((q for q in range(4) if used[q] < capacity[q]), key=lambda q: (used[q], q))
        local.append(available[q][16 * used[q]:16 * (used[q] + 1)])
        used[q] += 1
    cross = [[available[r % 4][logical * 4 + r // 4] for r in range(16)]
             for logical in range(30)]
    return {"local-pool": local, "cross-quadrant": cross}


def validate_coordinates(coordinates: torch.Tensor, logical: int, size: int) -> torch.Tensor:
    if coordinates.shape != (logical, size) or coordinates.dtype != torch.int64:
        raise ValueError("physical coordinates must be an int64 [logical, replica] array")
    if bool(((coordinates < 0) | (coordinates >= 512)).any()):
        raise ValueError("physical coordinate out of range")
    if coordinates.unique().numel() != coordinates.numel():
        raise ValueError("duplicate physical coordinates")
    return coordinates


def load_threshold_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("viable"):
        raise ValueError("threshold selection has not passed independent validation")
    if not payload.get("chip_identifier"):
        raise ValueError("threshold selection has no chip identifier")
    calibration = Path(payload["calibration_path"])
    if calibration_sha256(calibration) != payload["context"]["calibration_sha256"]:
        raise ValueError("selected analog calibration checksum mismatch")
    for placement in ("local-pool", "cross-quadrant"):
        values = validate_coordinates(torch.tensor(payload["coordinates"][placement]), 30, 16)
        quadrants = values // 128
        if placement == "local-pool" and not bool((quadrants == quadrants[:, :1]).all()):
            raise ValueError("local pool spans quadrants")
        if placement == "cross-quadrant" and not bool((quadrants == torch.arange(16) % 4).all()):
            raise ValueError("cross-quadrant replicas are not round robin")
    return payload


def selected_coordinates(config: Any, pool: Any, chip: list[str] | None) -> torch.Tensor:
    path = Path(config.threshold_selection_path)
    if calibration_sha256(path) != config.threshold_selection_sha256:
        raise ValueError("threshold selection checksum mismatch")
    payload = load_threshold_selection(path)
    # A longer deadline is deliberately allowed for the subsequent margin scan.
    if payload["context"] != weight_context(config) or chip != payload["chip_identifier"]:
        raise ValueError("threshold selection operating point or chip mismatch")
    if config.synaptic_weight != 63 or config.neuron_weight_calibration_path is not None:
        raise ValueError("threshold selection requires uniform weight 63")
    if pool.mapping != "dedicated" or pool.logical_neurons != 30 or pool.pool_size not in (1, 2, 4, 8, 16):
        raise ValueError("threshold selection requires the validated 30-neuron graph")
    return torch.tensor(payload["coordinates"][pool.placement], dtype=torch.long)[:, :pool.pool_size].contiguous()


def refine_result(result: Any, target: Any, connection: Any, refine: Any) -> Any:
    """Copy a base result; never accumulate candidate changes in the base."""
    candidate = deepcopy(result)
    refine(connection, candidate.neuron_result, target=deepcopy(target))
    candidate.target.neuron_target = deepcopy(candidate.neuron_result.target)
    return candidate


def apply_selection_to_args(args: Any) -> None:
    path = getattr(args, "threshold_selection_json", None)
    if path is None:
        return
    payload = load_threshold_selection(path)
    context = payload["context"]
    # Explicit calibration and analog arguments must agree. The notebook reads
    # the selection first; these checks also protect direct CLI invocations.
    for attribute, key in (("threshold", "threshold"), ("input_fan_in", "input_fan_in"),
                           ("tau_m_s", "tau_mem_s"), ("tau_syn_s", "tau_syn_s"),
                           ("i_synin_gm", "i_synin_gm"), ("leak", "leak"), ("reset", "reset")):
        if getattr(args, attribute) != context[key]:
            raise ValueError(f"{attribute} conflicts with selected physical calibration")
    if calibration_sha256(args.spiking_calibration) != context["calibration_sha256"]:
        raise ValueError("spiking calibration conflicts with threshold selection")
