"""Bridge bounded project potentials to BrainScaleS-2 input spike tensors."""

from __future__ import annotations

import math

import torch

from utils.transforms.noise import get_gaussian_time_noise
from utils.transforms.potential_to_spike import (
    neg_identity_transform,
    neg_log_transform,
)
from utils.transforms.types import Potential, SpikeSample, TimeBounds

from .config import BrainScaleS2PoolConfig, RoutingMode, TTFSHardwareEncoding


def _encode_project_time(
    potential: Potential,
    config: BrainScaleS2PoolConfig,
) -> tuple[torch.Tensor, TimeBounds, torch.Tensor, torch.Tensor]:
    if get_gaussian_time_noise().enabled:
        raise RuntimeError(
            "software Gaussian timing noise must be disabled for physical BSS-2 runs"
        )

    values = potential.value.detach().to(device="cpu")
    lower = values.new_tensor(float(potential.domain.min))
    upper = values.new_tensor(float(potential.domain.max))
    clamp_mask = (values < lower) | (values > upper)
    clamped = values.clamp(lower, upper)

    if config.encoding == "identity":
        encoded = neg_identity_transform(clamped, potential.domain)
    else:
        encoded = neg_log_transform(
            clamped,
            potential.domain,
            tau_s=config.project_tau_s,
        )
    if isinstance(encoded, SpikeSample):
        raise RuntimeError("nominal hardware encoding unexpectedly returned SpikeSample")
    project_time, project_domain = encoded
    return project_time, project_domain, clamped, clamp_mask


def encode_potential_for_brainscales2(
    potential: Potential,
    config: BrainScaleS2PoolConfig,
    *,
    pool_size: int,
    routing: RoutingMode,
) -> TTFSHardwareEncoding:
    """Encode one bounded potential tensor into quantized hxtorch input spikes."""
    if pool_size <= 0 or pool_size > 128:
        raise ValueError("pool_size must lie in [1, 128]")
    if routing not in ("broadcast", "independent"):
        raise ValueError("unsupported routing mode")

    project_time, project_domain, clamped, clamp_mask = _encode_project_time(
        potential,
        config,
    )
    project_width = float(project_domain.max) - float(project_domain.min)
    if not math.isfinite(project_width) or project_width <= 0.0:
        raise ValueError("project encoder produced a non-positive time window")

    normalized = (
        project_time.to(torch.float64) - float(project_domain.min)
    ) / project_width
    ideal_time_s = config.input_early_s + normalized.clamp(0.0, 1.0) * (
        config.input_late_s - config.input_early_s
    )

    bin_index = torch.round(ideal_time_s / config.dt_s).to(torch.long)
    if bool((bin_index < 0).any() or (bin_index >= config.runtime_steps).any()):
        raise ValueError("quantized input event falls outside the hardware runtime")
    injected_time_s = bin_index.to(torch.float64) * config.dt_s

    samples = project_time.numel()
    input_channels = (
        config.input_fan_in
        if routing == "broadcast"
        else pool_size * config.input_fan_in
    )
    dense = torch.zeros(
        (config.runtime_steps, samples, input_channels),
        dtype=torch.float32,
    )
    sample_index = torch.arange(samples, dtype=torch.long)
    if routing == "broadcast":
        channels = torch.arange(config.input_fan_in, dtype=torch.long)
        dense[
            bin_index.reshape(-1, 1),
            sample_index.reshape(-1, 1),
            channels.reshape(1, -1),
        ] = 1.0
    else:
        channels = torch.arange(input_channels, dtype=torch.long)
        dense[
            bin_index.reshape(-1, 1),
            sample_index.reshape(-1, 1),
            channels.reshape(1, -1),
        ] = 1.0

    return TTFSHardwareEncoding(
        dense_spikes=dense,
        ideal_time_s=ideal_time_s.reshape(-1),
        injected_time_s=injected_time_s.reshape(-1),
        original_shape=tuple(potential.value.shape),
        source_domain=potential.domain,
        source_time_domain=project_domain,
        physical_time_domain=TimeBounds(
            config.input_early_s,
            config.input_late_s,
        ),
        clamped_values=clamped,
        clamp_mask=clamp_mask,
        encoding=config.encoding,
        routing=routing,
        input_fan_in=config.input_fan_in,
    )
