"""Single source of truth for the neuromorphic noise model.

Everything noise-related lives here so the model can be understood by reading one file.
Three physically-separable effects (pure-simulation robustness study):

  A. Operating-point-dependent temporal jitter — injected in *potential* space so the
     encoder's own Jacobian yields σ_t = |dt/dV|·σ_V = σ_V/|dV/dt| (uniform on the linear
     data path, σ_V·τ/V on the log LayerNorm/attention path). A legacy time-space jitter is
     kept selectable for reproducing older sweeps.
  B. Drop/insertion escape-noise hazard — ρ(t) = ρ₀·exp((V−θ)/Δu) (Gerstner SRM), mapped to
     a per-neuron per-forward firing reliability for single-spike TTFS.
  C. Static device mismatch — frozen per-neuron threshold offset, installed as forward
     pre-hooks so the spiking modules themselves need no noise code.

A and B are applied at the encoder via the `inject_spike_time_noise` decorator (which every
potential→spike transform already carries). C is applied by `install_device_mismatch`.

Colored 1/f·RTN noise is intentionally excluded: single-spike TTFS has no intra-inference time
axis, so low-frequency components collapse to the quasi-static offset that C already models.

Scope & calibration caveats
---------------------------
- The decorated encoders (neg_linear/neg_log) are reused as internal arithmetic primitives:
  multiplication_operator, division_function and softmin all call an encoder, and the spiking
  LayerNorm/attention chain calls several per layer. So A/B perturb *every spike-time
  sub-computation*, not once per physical neuron. Noise therefore compounds across hundreds of
  sites and the model shows a fairly sharp robustness cliff — magnitudes must be small
  (jitter std and mismatch σ_θ ≈ 1e-6…1e-5 for ViT-S at θ=2000). This is a deliberate
  worst-case ("noise at every operation") reading, not a bug.
- The spiking LayerNorm is the sensitivity bottleneck; sweeps should bracket its cliff.
- A/B reference their scale to θ (`potential_scale`), not each encoder's domain range, because
  the LayerNorm variance encoder's range is θ² and would otherwise dominate. See reports/NOISE.md.
"""

import math
from dataclasses import dataclass
from functools import wraps
from typing import Callable, cast

import torch
from torch import Tensor

from .types import OpenBounds, Potential


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Global configuration for encoder-level spike-time noise (jitter + hazard).

    Device mismatch (C) is per-module state and is not held here.
    """
    std: float = 0.0
    kind: str = "gaussian"
    eval_mode: bool = False
    mode: str = "potential"            # "potential" (V-referred) | "time" (legacy, t-referred)
    potential_scale: float = 0.0       # σ_V = std·potential_scale (θ); 0 ⇒ fall back to domain.range
    jitter_enabled: bool = False
    hazard_enabled: bool = False
    hazard_delta_u: float = 0.0        # Δu, relative to the potential-domain range
    hazard_insert_rate: float = 0.0    # ρ₀, baseline spurious-spike probability

_GLOBAL_NOISE_CONFIG = NoiseConfig()


def set_spike_time_noise(
    std: float,
    kind: str = "gaussian",
    eval_mode: bool = False,
    *,
    mode: str = "potential",
    potential_scale: float = 0.0,
    jitter_enabled: bool | None = None,
    hazard_enabled: bool = False,
    hazard_delta_u: float = 0.0,
    hazard_insert_rate: float = 0.0,
):
    """Register global encoder-noise parameters.

    `jitter_enabled=None` (the default used by the older call sites) enables jitter iff
    `std > 0`, preserving backward compatibility. `potential_scale` is the voltage reference θ
    for potential-mode jitter (σ_V = std·θ); 0 falls back to each encoder's domain range.
    """
    global _GLOBAL_NOISE_CONFIG
    _GLOBAL_NOISE_CONFIG.std = std
    _GLOBAL_NOISE_CONFIG.kind = kind
    _GLOBAL_NOISE_CONFIG.eval_mode = eval_mode
    _GLOBAL_NOISE_CONFIG.mode = mode
    _GLOBAL_NOISE_CONFIG.potential_scale = potential_scale
    _GLOBAL_NOISE_CONFIG.jitter_enabled = (std > 0.0) if jitter_enabled is None else jitter_enabled
    _GLOBAL_NOISE_CONFIG.hazard_enabled = hazard_enabled
    _GLOBAL_NOISE_CONFIG.hazard_delta_u = hazard_delta_u
    _GLOBAL_NOISE_CONFIG.hazard_insert_rate = hazard_insert_rate


def get_spike_time_noise() -> NoiseConfig:
    """Retrieve global encoder-noise parameters."""
    return _GLOBAL_NOISE_CONFIG


# Bridge constant: Gumbel(first-passage) std / scale = σ_V / Δu.
_SIGMA_V_OVER_DELTA_U = math.pi / math.sqrt(6.0)


def sigma_v_frac(beta: float) -> float:
    """σ_V/θ implied by a dimensionless sharpness β (= β_phys·θ). σ_V/θ = (π/√6)/β."""
    return _SIGMA_V_OVER_DELTA_U / beta


def set_unified_noise(
    beta: float,
    theta: float,
    *,
    jitter_only: bool = False,
    drop_only: bool = False,
    insert_rate: float = 0.0,
    eval_mode: bool = True,
):
    """Physically-coupled single-β escape-noise (the faithful primary model).

    Jitter (A) and drop (B) are the *variance* and the *survival* of ONE escape-noise
    first-passage, so both derive from a single dimensionless sharpness β (= β_phys·θ):

        σ_V/θ = (π/√6)/β      # Gumbel std of the first-passage time (operating-point-free)
        Δu/θ  = 1/β           # soft-threshold width, Δu = 1/β_phys

    hence σ_V = (π/√6)·Δu. Larger β ⇒ sharper threshold ⇒ less noise. This replaces the two
    independent knobs with one physical parameter (uncalibrated but structurally faithful — a
    single value is *swept*, not fit to a device).

    `jitter_only`/`drop_only` are ABLATION masks: an analysis maneuver to isolate one channel,
    NOT a claim that the hardware exposes independent knobs. Leave both False for the faithful
    coupled model.
    """
    set_spike_time_noise(
        std=sigma_v_frac(beta),
        eval_mode=eval_mode,
        mode="potential",
        potential_scale=theta,
        jitter_enabled=not drop_only,
        hazard_enabled=not jitter_only,
        hazard_delta_u=1.0 / beta,
        hazard_insert_rate=insert_rate,
    )


# ---------------------------------------------------------------------------
# A — temporal jitter
# ---------------------------------------------------------------------------

def _emit_spike_time_core(
    input_value: Tensor,
    domain: OpenBounds,
    *,
    noise_std: float = 0.0,
    noise_kind: str = "gaussian",
) -> Tensor:
    """Legacy time-space jitter: add Gaussian noise to the emitted spike time and re-clamp.

    Kept for `mode="time"` so historical `jitter_analysis` / `theta_jitter_analysis` sweeps
    remain reproducible. `noise_std` is relative to the (time) domain range.
    """
    if noise_std <= 0.0:
        return domain.clamp(input_value)

    span = float(domain.range)
    if noise_kind == "gaussian":
        noise = torch.randn_like(input_value) * (noise_std * span)
    else:
        raise ValueError(f"Unsupported noise_kind: {noise_kind}. Use 'gaussian'")

    return domain.clamp(input_value + noise)


# ---------------------------------------------------------------------------
# B — escape-noise hazard (drop / insertion)
# ---------------------------------------------------------------------------

def _apply_escape_hazard(
    t: Tensor,
    V: Tensor,
    in_domain: OpenBounds,
    out_domain: OpenBounds,
    cfg: NoiseConfig,
) -> Tensor:
    """Escape-noise hazard: probabilistic spike drop / insertion near threshold.

    ρ(t) = ρ₀·exp((V−θ)/Δu) mapped to a per-neuron per-forward firing reliability. The drive
    margin is measured from the encoding floor `in_domain.min`, so weak (near-floor /
    late-spiking) neurons drop and strongly-driven neurons fire reliably (Mainen–Sejnowski).

    Dropped neurons are pushed to `out_domain.max` (the silent / latest spike time); a fraction
    `hazard_insert_rate` of spurious early spikes are set to `out_domain.min`.

    Δu is referenced to the same voltage scale θ (`potential_scale`) as the jitter, falling back
    to the encoder's domain range — otherwise the θ² range of the LayerNorm variance encoder
    would make the soft-threshold width meaningless there.
    """
    scale = cfg.potential_scale if cfg.potential_scale > 0.0 else float(in_domain.range)
    du = float(cfg.hazard_delta_u) * scale
    if du <= 0.0:
        return t

    p_fire = (1.0 - torch.exp(-(V - in_domain.min) / du)).clamp(0.0, 1.0)
    drop = torch.rand_like(t) > p_fire
    t = torch.where(drop, torch.full_like(t, float(out_domain.max)), t)

    insert_rate = float(cfg.hazard_insert_rate)
    if insert_rate > 0.0:
        insert = torch.rand_like(t) < insert_rate
        t = torch.where(insert, torch.full_like(t, float(out_domain.min)), t)
    return t


# ---------------------------------------------------------------------------
# Encoder decorator (applies A and B)
# ---------------------------------------------------------------------------

def inject_spike_time_noise[**P, OutT: OpenBounds](func: Callable[P, tuple[Tensor, OutT]]) -> Callable[P, tuple[Tensor, OutT]]:
    """Decorator that injects operating-point jitter and escape-noise hazard.

    Attached to every potential→spike encoder. Because it wraps the transform, it sees both the
    pre-transform potential `V` (`args[0]` / `kwargs["input_value"]`) with its potential domain
    (`args[1]` / `kwargs["domain"]`) and the emitted spike time. Jitter is injected in potential
    space by default (the encoder's Jacobian then yields σ_t = σ_V/|dV/dt|); the legacy
    time-space path is preserved under `mode="time"`.

    When jitter and hazard are both disabled the output is only re-clamped into its declared
    domain — numerically identical to the pre-refactor behaviour.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[Tensor, OutT]:
        cfg = get_spike_time_noise()

        # Priority: explicit kwarg > global config.
        std_raw = kwargs.get("noise_std", cfg.std)
        std = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        kind = cast(str, kwargs.get("noise_kind", cfg.kind))
        mode = cast(str, kwargs.get("noise_mode", cfg.mode))
        jitter_on = bool(kwargs.get("jitter_enabled", cfg.jitter_enabled))
        hazard_on = bool(kwargs.get("hazard_enabled", cfg.hazard_enabled))

        # Locate the input potential and its declared domain.
        V = args[0] if len(args) > 0 else kwargs.get("input_value")
        in_domain = args[1] if len(args) > 1 else kwargs.get("domain")
        have_V = torch.is_tensor(V) and in_domain is not None

        # A (potential-referred): perturb V before the transform, re-clamp into the potential
        # domain so `@check_domain` and neg_log's V>0 precondition still hold. σ_V is referenced
        # to a single voltage scale θ (`potential_scale`) rather than each encoder's domain range
        # — the internal LayerNorm variance encoder has domain range θ², which would otherwise
        # inject wildly (∼10³×) more noise there than on the data path.
        if jitter_on and std > 0.0 and mode == "potential" and have_V:
            sigma_V = std * (cfg.potential_scale if cfg.potential_scale > 0.0 else float(in_domain.range))
            V = in_domain.clamp(V + torch.randn_like(V) * sigma_V)
            if len(args) > 0:
                args = (V,) + args[1:]
            else:
                kwargs = {**kwargs, "input_value": V}

        output, out_domain = func(*args, **kwargs)

        # A (legacy, time-referred): perturb the emitted spike time.
        if jitter_on and std > 0.0 and mode == "time":
            output = _emit_spike_time_core(output, out_domain, noise_std=std, noise_kind=kind)

        # B: escape-noise drop/insertion, acting on the effective potential V.
        if hazard_on and have_V:
            output = _apply_escape_hazard(output, V, in_domain, out_domain, cfg)

        # Always project into the declared output domain (matches prior behaviour).
        return out_domain.clamp(output), out_domain

    return wrapper


# ---------------------------------------------------------------------------
# C — static device mismatch (per-neuron frozen threshold offset)
# ---------------------------------------------------------------------------

def _mismatch_pre_hook(module, args):
    """forward_pre_hook: add the module's frozen per-neuron offset to the input potential."""
    pot: Potential = args[0]
    return (Potential(pot.value + module._mismatch_offset, pot.domain),) + tuple(args[1:])


def install_device_mismatch(model, theta_std: float, enabled: bool = True):
    """Attach static device mismatch to every spiking encoder module via forward pre-hooks.

    θ_i = θ·(1+N(0,σ_θ)) is realised as a fixed potential shift −δ_i per encoding neuron, so the
    interval-arithmetic / domain-clamp machinery keeps a scalar θ (per-neuron *saturation* is
    intentionally not modelled). Offsets are sampled once (reproducible via the caller's torch
    seed) and frozen as non-persistent buffers — not resampled per forward, and excluded from
    `state_dict`. Call after the model is built, weights loaded, and moved to its device.

    Returns the list of hook handles (for optional removal); empty when disabled.
    """
    if not enabled or theta_std <= 0.0:
        return []

    # Lazy import to avoid a package import cycle (spiking_ops imports utils.transforms).
    from utils.transformers.models.spiking_ops import (
        SpikingLayerNorm, SpikingLinear, SpikingConv2d,
    )

    handles = []
    for _, m in model.named_modules():
        if isinstance(m, SpikingLayerNorm):
            shape = tuple(m.normalized_shape)          # broadcasts over the trailing feature dim
        elif isinstance(m, SpikingConv2d):
            shape = (1, m.in_channels, 1, 1)           # broadcasts over the channel dim of [B,C,H,W]
        elif isinstance(m, SpikingLinear):
            shape = (m.in_features,)                   # broadcasts over the trailing feature dim
        else:
            continue

        w = m.weight
        offset = torch.randn(*shape, device=w.device, dtype=w.dtype) * (float(m.theta) * float(theta_std))
        m.register_buffer("_mismatch_offset", offset, persistent=False)
        handles.append(m.register_forward_pre_hook(_mismatch_pre_hook))

    return handles


# ---------------------------------------------------------------------------
# Per-module noise scoping (attribution experiments)
# ---------------------------------------------------------------------------

def _noise_off():
    set_spike_time_noise(std=0.0, jitter_enabled=False, hazard_enabled=False)


def install_noise_scope(model, is_noisy, **on_kwargs):
    """Scope encoder noise (jitter/hazard) to a subset of modules.

    Sets the global config OFF, then brackets each module where `is_noisy(module)` is True with a
    forward pre-hook that turns noise ON (with `on_kwargs`, e.g. std/mode/potential_scale/
    jitter_enabled) and a post-hook that turns it OFF again. Only the encoder calls executed
    *inside* a matching module are perturbed, so "only attention" / "everything but the FFN" /
    … can be measured. Returns hook handles; call `.remove()` on each to restore.

    Single-threaded eval only — it mutates the global NoiseConfig, so do NOT use with
    DataParallel. Assumes matching modules are not nested inside one another (group containers).
    """
    _noise_off()

    def pre(module, args):
        set_spike_time_noise(**on_kwargs)

    def post(module, args, output):
        _noise_off()

    handles = []
    for m in model.modules():
        if is_noisy(m):
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))
    return handles
