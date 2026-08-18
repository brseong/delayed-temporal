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
from typing import Callable, TypedDict, cast

import torch
from torch import Tensor

from .types import OpenBounds, Potential, SpikeSample, TimeBounds


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

@dataclass
class GaussianTimeNoiseConfig:
    """Process-wide configuration for direct Gaussian spike-time noise.

    ``time_mean`` and ``time_std`` are absolute time quantities shared by every
    event-aware encoder in one evaluation replica. ``generator`` is seeded once
    during configuration and then advances across calls, making a seed identify a
    complete replica rather than restarting the random sequence for each layer.
    A disabled configuration holds no generator so accidental sampling cannot
    silently consume an unrelated global RNG stream.
    """

    enabled: bool = False  # Select the event-aware Gaussian path at encoder boundaries.
    time_std: float = 0.0  # Absolute standard deviation applied to every encoded time.
    time_mean: float = 0.0  # Absolute additive bias applied before deadline classification.
    seed: int = 0  # Replica seed used once when constructing the dedicated generator.
    generator: torch.Generator | None = None  # Stateful RNG owned by this configuration.


class GaussianNoiseCounts(TypedDict):
    """Statically known schema for one site's mutable Gaussian measurements.

    The set of metric names is fixed for type checking, while each integer remains
    mutable so repeated encoder and readout observations can accumulate in place.
    Event and output totals are intentionally separate statistical denominators.
    """

    events: int  # Number of sampled spike events at this site.
    misses: int  # Number of sampled events arriving after the fixed deadline.
    outputs: int  # Number of analog readout values observed before rail clamping.
    output_underflows: int  # Readouts strictly below the declared output minimum.
    output_overflows: int  # Readouts strictly above the declared output maximum.


# The direct timing model has one process-wide configuration so every decorated
# encoder in a replica consumes the same stateful random stream. Configuration is
# replaced atomically by the setter instead of mutating its fields across calls.
_GLOBAL_GAUSSIAN_TIME_CONFIG = GaussianTimeNoiseConfig()

# Statistics are grouped first by a stable call-site name and then by counter name.
# Keeping this store separate from the configuration lets a new replica clear its
# measurements without coupling counter mutation to dataclass replacement.
_GAUSSIAN_NOISE_STATS: dict[str, GaussianNoiseCounts] = {}


def clear_gaussian_noise_stats() -> None:
    """Clear all process-wide Gaussian event and output counters.

    Counter reset is intentionally independent of timing configuration. Callers can
    begin a fresh measurement interval without changing the enabled flag, absolute
    noise parameters, replica seed, or current generator state.
    """
    # Reset measurement state only; the active configuration and its advancing RNG
    # must remain untouched when a caller starts a new reporting interval.

    # Clear the shared mapping in place rather than rebinding it so any diagnostic
    # code holding a reference observes the same empty process-wide store.
    _GAUSSIAN_NOISE_STATS.clear()


def get_gaussian_noise_stats() -> dict[str, GaussianNoiseCounts]:
    """Return a detached snapshot of all per-site Gaussian noise counters.

    Diagnostics may freely aggregate or annotate the returned mapping without
    mutating the process-wide measurements that continue accumulating during model
    execution. Each copied site retains the statically known counter schema, and
    integer values make a two-level copy sufficient to detach the full structure.
    """
    # Build a new outer mapping so adding or removing sites in the caller's snapshot
    # cannot change which locations are tracked by the active evaluation process.

    # Copy every nested counter mapping as well; an outer-only copy would still let
    # callers overwrite live event, miss, or saturation counts through shared dicts.
    return {site: counts.copy() for site, counts in _GAUSSIAN_NOISE_STATS.items()}


def _stats_for(site: str) -> GaussianNoiseCounts:
    """Return the live Gaussian counter mapping for one instrumentation site.

    A site is created lazily on its first event or output observation. Event and
    output denominators remain separate because one encoded event can influence an
    operator output with a different shape, and their rates must not be mixed.

    Args:
        site: Stable, non-empty name identifying an encoder or output location.

    Returns:
        The mutable, statically keyed process-wide counter mapping owned by ``site``.

    Raises:
        TypeError: If ``site`` is not a string.
        ValueError: If ``site`` is empty or contains only whitespace.
    """
    # Reject anonymous locations before touching global state; empty keys would
    # merge unrelated measurements and make per-site attribution meaningless.
    if not isinstance(site, str):
        raise TypeError("Gaussian statistics site must be a string")
    if not site.strip():
        raise ValueError("Gaussian statistics site must not be empty")

    # Reuse an existing live mapping so increments from repeated encoder calls
    # accumulate into one stable process-wide measurement interval.
    counts = _GAUSSIAN_NOISE_STATS.get(site)
    if counts is not None:
        return counts

    # Initialize every supported metric together, keeping a fixed schema across
    # sites even when a location has observed only events or only output values.
    counts = {
        "events": 0,
        "misses": 0,
        "outputs": 0,
        "output_underflows": 0,
        "output_overflows": 0,
    }
    _GAUSSIAN_NOISE_STATS[site] = counts

    # Writers intentionally receive the live mapping; public readers use the
    # detached snapshot returned by get_gaussian_noise_stats instead.
    return counts


def clamp_gaussian_output(
    value: Tensor,
    domain: OpenBounds,
    *,
    site: str,
    name: str,
) -> Tensor:
    """Clamp an analog readout and record its pre-clamp rail saturation.

    Output clamping is part of the bounded operator contract regardless of whether
    Gaussian spike-time noise is active. When it is active, this function also
    records how many raw output elements fall outside the representable rails so
    saturation rates can be reported independently from event miss rates.

    Args:
        value: Unclamped analog output produced by a physical readout.
        domain: Representable output interval whose endpoints are the clamp rails.
        site: Stable statistics key identifying the operator output location.
        name: Diagnostic label forwarded to the bounds clamp implementation.

    Returns:
        ``value`` clamped to ``domain`` with its tensor metadata preserved.

    Raises:
        TypeError: If ``site`` is not a string while Gaussian noise is enabled.
        ValueError: If ``site`` is empty while Gaussian noise is enabled.
    """
    # Rail enforcement belongs to the deterministic operator contract as well as
    # the noisy path, so compute the bounded result before consulting noise state.
    clamped = domain.clamp(value, name=name)

    # A noise-disabled evaluation must not create statistics sites or alter an
    # existing measurement interval merely because its outputs were clamped.
    if not _GLOBAL_GAUSSIAN_TIME_CONFIG.enabled:
        return clamped

    # Count every raw output element in the saturation denominator. The live,
    # statically keyed mapping lets repeated calls accumulate at the same site.
    counts = _stats_for(site)
    counts["outputs"] += value.numel()

    # Compare the raw readout with strict inequalities before returning the clamp.
    # Values exactly on a rail remain representable and are not saturation events.
    counts["output_underflows"] += int((value < domain.min).sum().item())
    counts["output_overflows"] += int((value > domain.max).sum().item())

    # Return the previously computed bounded tensor; statistics never modify the
    # physical value delivered to the next operator.
    return clamped


def set_gaussian_time_noise(
    *,
    enabled: bool,
    time_std: float = 0.0,
    time_mean: float = 0.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> None:
    """Install process-wide direct Gaussian spike-time noise configuration.

    Each successful call starts a new experiment replica: it constructs and seeds
    one generator for the requested sampling device, replaces the complete global
    configuration, and clears measurements from the previous replica. The generator
    then advances across encoder calls; individual forwards must never reseed it.

    Args:
        enabled: Whether event-aware encoders apply direct Gaussian timing noise.
        time_std: Non-negative standard deviation in absolute time units.
        time_mean: Additive Gaussian mean in absolute time units.
        seed: Integer seed used once to initialize the replica generator.
        device: Device on which the encoder's spike-time samples will be drawn.

    Raises:
        TypeError: If ``enabled`` is not boolean or ``seed`` is not an integer.
        ValueError: If a timing parameter is non-finite, ``time_std`` is negative,
            or the legacy dynamic-noise path is already active.
        RuntimeError: If PyTorch cannot create or seed a generator on ``device``.
    """
    global _GLOBAL_GAUSSIAN_TIME_CONFIG

    # Reject ambiguous flag and seed values rather than accepting truthy strings or
    # truncating fractional seeds, both of which would obscure replica identity.
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    # Normalize numeric inputs once, then require finite physical parameters before
    # creating random state or disturbing the currently installed configuration.
    normalized_std = float(time_std)
    normalized_mean = float(time_mean)
    if not math.isfinite(normalized_std) or not math.isfinite(normalized_mean):
        raise ValueError("Gaussian time-noise parameters must be finite")
    if normalized_std < 0.0:
        raise ValueError("time_std must be non-negative")

    # During the staged migration, activating both models would perturb one event
    # twice and make its timing distribution uninterpretable, so fail explicitly.
    if enabled and (
        _GLOBAL_NOISE_CONFIG.jitter_enabled
        or _GLOBAL_NOISE_CONFIG.hazard_enabled
    ):
        raise ValueError(
            "direct Gaussian noise and legacy dynamic noise are mutually exclusive"
        )

    # A disabled configuration owns no generator. An enabled replica gets exactly
    # one device-matched stream seeded here and advanced later by encoder sampling.
    generator = None
    if enabled:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(seed)

    # Build the full replacement before touching shared state. Consequently, any
    # validation, device, or seed failure preserves both the old replica and counts.
    new_config = GaussianTimeNoiseConfig(
        enabled=enabled,
        time_std=normalized_std,
        time_mean=normalized_mean,
        seed=seed,
        generator=generator,
    )

    # Installing a new configuration defines a new measurement interval. Replace
    # the config atomically, then clear counters without altering its fresh RNG.
    _GLOBAL_GAUSSIAN_TIME_CONFIG = new_config
    clear_gaussian_noise_stats()


def get_gaussian_time_noise() -> GaussianTimeNoiseConfig:
    """Return the process-wide direct Gaussian timing configuration.

    The returned object is the currently installed replica configuration, including
    its stateful generator. Callers should treat its fields as read-only and use
    :func:`set_gaussian_time_noise` to start or disable a replica.

    Returns:
        The shared configuration whose generator advances across encoder calls.
    """
    # Preserve generator identity across every read; copying or reconstructing the
    # configuration here could accidentally fork the replica's random sequence.
    config = _GLOBAL_GAUSSIAN_TIME_CONFIG

    # Configuration replacement is centralized in the setter, so this accessor has
    # no validation, counter reset, or other observable side effect.
    return config


def _broadcast_gaussian_time_inputs(
    nominal_time: Tensor,
    *,
    time_mean: Tensor | float,
    time_std: Tensor | float,
    domain: TimeBounds,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate and broadcast direct Gaussian timing inputs.

    The sampler and its analytic deadline-miss calculation must apply identical
    shape, dtype, device, and domain rules. This helper establishes that shared
    contract before either path performs distribution-specific arithmetic.

    Args:
        nominal_time: Deterministic encoder output inside ``domain``.
        time_mean: Scalar or broadcastable absolute Gaussian mean.
        time_std: Scalar or broadcastable non-negative absolute standard deviation.
        domain: Fixed TTFS interval whose maximum is the observation deadline.

    Returns:
        Broadcast nominal time, mean, and standard-deviation tensors sharing the
        nominal tensor's floating dtype and device.

    Raises:
        TypeError: If the nominal tensor is not floating-point or the domain is not
            a ``TimeBounds`` instance.
        ValueError: If the domain is invalid, an input is non-finite, a standard
            deviation is negative, or a nominal time lies outside the code interval.
    """
    # Gaussian arithmetic must preserve fractional timing values. Reject integer
    # tensors instead of silently promoting them and changing the caller's dtype.
    if not torch.is_floating_point(nominal_time):
        raise TypeError("nominal_time must be a floating-point tensor")

    # TimeBounds.max is both the code endpoint and the physical observation
    # deadline, so invalid endpoints would make every later miss decision ambiguous.
    if not isinstance(domain, TimeBounds):
        raise TypeError("domain must be a TimeBounds instance")
    domain_min = float(domain.min)
    domain_max = float(domain.max)
    if not math.isfinite(domain_min) or not math.isfinite(domain_max):
        raise ValueError("time-domain endpoints must be finite")
    if domain_min > domain_max:
        raise ValueError("time domain must satisfy min <= max")

    # Materialize scalar or tensor parameters beside the nominal data. This keeps
    # downstream sampling on one device and prevents unintended dtype promotion.
    mean = torch.as_tensor(
        time_mean,
        dtype=nominal_time.dtype,
        device=nominal_time.device,
    )
    std = torch.as_tensor(
        time_std,
        dtype=nominal_time.dtype,
        device=nominal_time.device,
    )

    # Broadcast all distribution parameters together so both the analytic and
    # sampled paths observe exactly the same elementwise parameter layout.
    nominal, mean, std = torch.broadcast_tensors(nominal_time, mean, std)

    # NaN or infinity would make both Gaussian probabilities and deadline masks
    # undefined, so fail at the shared boundary rather than propagating them.
    if not bool(
        torch.isfinite(nominal).all()
        and torch.isfinite(mean).all()
        and torch.isfinite(std).all()
    ):
        raise ValueError("nominal_time, time_mean, and time_std must be finite")

    # Standard deviation is a magnitude, while nominal times must already obey the
    # deterministic encoder contract before any stochastic perturbation is applied.
    if bool((std < 0.0).any()):
        raise ValueError("time_std must be non-negative")
    if bool((nominal < domain_min).any() or (nominal > domain_max).any()):
        raise ValueError("nominal_time must lie within the declared time domain")

    return nominal, mean, std


def gaussian_deadline_miss_probability(
    nominal_time: Tensor,
    *,
    time_std: Tensor | float,
    domain: TimeBounds,
    time_mean: Tensor | float = 0.0,
) -> Tensor:
    """Return the analytic probability that a Gaussian event misses its deadline.

    This function evaluates ``P(t_nominal + N(time_mean, time_std) > domain.max)``
    without drawing a random sample. Equality with ``domain.max`` is a delivered
    event, matching the fixed-deadline rule used by the event sampler.

    Args:
        nominal_time: Deterministic encoder output inside ``domain``.
        time_std: Scalar or broadcastable absolute Gaussian standard deviation.
        domain: Fixed TTFS interval whose maximum is the observation deadline.
        time_mean: Scalar or broadcastable absolute Gaussian mean.

    Returns:
        A floating tensor of miss probabilities with the shared broadcast shape,
        dtype, and device established from ``nominal_time``.
    """
    # Reuse the sampler's validation and broadcasting contract so the analytic
    # expectation is defined for exactly the same parameter combinations.
    nominal, mean, std = _broadcast_gaussian_time_inputs(
        nominal_time,
        time_mean=time_mean,
        time_std=time_std,
        domain=domain,
    )

    # Tensorize the fixed deadline beside the inputs to avoid CPU scalar promotion
    # and to preserve the nominal tensor's floating dtype on accelerators.
    deadline = nominal.new_tensor(float(domain.max))

    # The closed-form Gaussian tail divides by sigma. Substitute one only for the
    # unused zero-sigma elements so the vectorized expression remains finite.
    zero_std = std == 0.0
    safe_std = torch.where(zero_std, torch.ones_like(std), std)
    standardized_margin = (deadline - nominal - mean) / safe_std

    # erfc computes the upper Gaussian tail directly and avoids cancellation from
    # evaluating one minus a CDF when the miss probability is very small.
    gaussian_tail = 0.5 * torch.erfc(standardized_margin / math.sqrt(2.0))

    # Zero noise is deterministic and must not be approximated by the artificial
    # safe sigma. A timestamp exactly at the deadline is delivered, not missed.
    deterministic_miss = (nominal + mean > deadline).to(nominal.dtype)
    return torch.where(zero_std, deterministic_miss, gaussian_tail)


def _sample_gaussian_spike_time(
    nominal_time: Tensor,
    *,
    time_std: Tensor | float,
    domain: TimeBounds,
    generator: torch.Generator,
    time_mean: Tensor | float = 0.0,
) -> SpikeSample:
    """Sample one Gaussian timing error and classify event delivery.

    A single sampled timestamp determines both the delivered time and whether the
    event misses ``domain.max``. Misses retain the deadline as finite tensor storage
    while ``SpikeSample.fired`` preserves the physical distinction between a miss
    and an event delivered exactly at the deadline.

    Args:
        nominal_time: Deterministic encoder output inside ``domain``.
        time_std: Scalar or broadcastable absolute Gaussian standard deviation.
        domain: Fixed TTFS interval whose maximum is the observation deadline.
        generator: Dedicated stateful RNG for the current evaluation replica.
        time_mean: Scalar or broadcastable absolute Gaussian mean.

    Returns:
        A finite ``SpikeSample`` with broadcast timestamps and a boolean delivery
        mask on the nominal tensor's dtype and device.

    Raises:
        TypeError: If ``generator`` is not an explicit ``torch.Generator``.
        TypeError: If timing tensors violate the shared floating-point contract.
        ValueError: If timing parameters or the fixed time domain are invalid.
    """
    # Require an explicit generator so sampling cannot silently consume PyTorch's
    # process-global RNG and make a configured experiment seed ineffective.
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")

    # Apply the exact validation and broadcasting contract used by the analytic
    # miss probability before any random state is consumed.
    nominal, mean, std = _broadcast_gaussian_time_inputs(
        nominal_time,
        time_mean=time_mean,
        time_std=time_std,
        domain=domain,
    )
    shifted_mean = nominal + mean

    # The zero-noise path is deliberately deterministic and leaves the generator
    # state untouched, which makes exact event-path parity tests reproducible.
    if bool((std == 0.0).all()):
        raw_time = shifted_mean
    else:
        # torch.normal owns the Gaussian draw while the explicit generator keeps
        # one reproducible random stream advancing across all encoder calls.
        raw_time = torch.normal(
            mean=shifted_mean,
            std=std,
            generator=generator,
        )

    # The upper endpoint is inclusive: arrival exactly at the observation deadline
    # is a delivered event, while any later raw timestamp is a deadline miss.
    start = nominal.new_tensor(float(domain.min))
    deadline = nominal.new_tensor(float(domain.max))
    fired = raw_time <= deadline

    # Events earlier than the modeled interval are observable from its start. Do
    # not upper-clamp here because late samples must first remain identifiable misses.
    delivered_time = torch.clamp(raw_time, min=start)

    # Replace every missed raw timestamp with a finite deadline carrier. Consumers
    # must use fired—not the stored value—to distinguish misses from on-time arrivals.
    stored_time = torch.where(fired, delivered_time, deadline)
    return SpikeSample(time=stored_time, domain=domain, fired=fired)


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

def inject_spike_time_noise[**P, OutT: OpenBounds](
    func: Callable[P, tuple[Tensor, OutT]],
) -> Callable[P, tuple[Tensor, OutT] | SpikeSample]:
    """Decorate a potential-to-spike encoder with shared noise injection.

    Tensor-only calls retain the deterministic or legacy ``(time, bounds)``
    interface during migration. An event-aware caller explicitly requests
    ``return_spike_sample=True``; when direct Gaussian noise is enabled, one sampled
    timestamp then determines both its finite stored time and deadline-delivery mask.

    The decorator remains the sole dynamic-noise boundary. This guarantees that
    linear and logarithmic encoders use the same absolute Gaussian parameters,
    generator stream, deadline rule, and statistics schema while legacy call sites
    continue to work until their later removal.

    Args:
        func: Deterministic encoder returning a time tensor and its declared bounds.

    Returns:
        A wrapped encoder supporting the existing tuple interface and the explicit
        event-aware ``SpikeSample`` interface.

    Raises:
        ValueError: If direct Gaussian and legacy dynamic noise are both active.
        RuntimeError: If an event-aware result is requested while Gaussian noise is
            disabled or its enabled configuration has no generator.
        TypeError: If an event-aware encoder does not declare ``TimeBounds``.
    """

    @wraps(func)
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Tensor, OutT] | SpikeSample:
        # Read both process-wide configurations once so this encoder call observes a
        # consistent migration state even though the legacy path still exists.
        cfg = get_spike_time_noise()
        gaussian_cfg = get_gaussian_time_noise()
        return_spike_sample = bool(kwargs.get("return_spike_sample", False))

        # Preserve legacy keyword precedence until its callers and implementation
        # are deleted in the final cleanup phase.
        std_raw = kwargs.get("noise_std", cfg.std)
        std = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        kind = cast(str, kwargs.get("noise_kind", cfg.kind))
        mode = cast(str, kwargs.get("noise_mode", cfg.mode))
        jitter_on = bool(kwargs.get("jitter_enabled", cfg.jitter_enabled))
        hazard_on = bool(kwargs.get("hazard_enabled", cfg.hazard_enabled))

        # Applying both models would perturb and suppress the same physical event
        # twice. Guard the boundary as well as the setter because legacy state can
        # still be enabled independently while the staged migration is in progress.
        if gaussian_cfg.enabled and (jitter_on or hazard_on):
            raise ValueError(
                "direct Gaussian noise and legacy dynamic noise are mutually exclusive"
            )

        # Locate the potential and domain for the temporary potential-referred
        # legacy branch. Gaussian timing noise is applied only after encoding.
        potential = args[0] if len(args) > 0 else kwargs.get("input_value")
        in_domain = args[1] if len(args) > 1 else kwargs.get("domain")
        have_potential = torch.is_tensor(potential) and in_domain is not None

        # Legacy potential jitter must precede the deterministic transform. Its
        # perturbed value is projected back into the encoder's valid input domain.
        if jitter_on and std > 0.0 and mode == "potential" and have_potential:
            sigma_v = std * (
                cfg.potential_scale
                if cfg.potential_scale > 0.0
                else float(in_domain.range)
            )
            potential = in_domain.clamp(potential + torch.randn_like(potential) * sigma_v)
            if len(args) > 0:
                args = (potential,) + args[1:]
            else:
                kwargs = {**kwargs, "input_value": potential}

        # Always run the deterministic encoder exactly once. Gaussian noise acts on
        # its bounded nominal spike time rather than perturbing the input potential.
        output, out_domain = func(*args, **kwargs)

        if return_spike_sample:
            # Event-aware semantics belong exclusively to the maintained Gaussian
            # path; refusing an inactive request avoids silently returning all-fired
            # events under a configuration the caller expected to be stochastic.
            if not gaussian_cfg.enabled:
                raise RuntimeError(
                    "return_spike_sample requires enabled Gaussian time noise"
                )
            if not isinstance(out_domain, TimeBounds):
                raise TypeError(
                    "event-aware spike encoders must return TimeBounds"
                )
            if not isinstance(gaussian_cfg.generator, torch.Generator):
                raise RuntimeError(
                    "enabled Gaussian time noise requires a torch.Generator"
                )

            # Enforce the deterministic encoder's existing bounds contract before
            # sampling; TimeBounds.max is directly the inclusive observation deadline.
            nominal_time = out_domain.clamp(output)
            sample = _sample_gaussian_spike_time(
                nominal_time,
                time_std=gaussian_cfg.time_std,
                domain=out_domain,
                generator=gaussian_cfg.generator,
                time_mean=gaussian_cfg.time_mean,
            )

            # Count the same sampled events used by downstream physical readout.
            # Misses are strict deadline exceedances encoded by the fired mask.
            site = kwargs.get("noise_site", func.__name__)
            counts = _stats_for(site)
            counts["events"] += sample.time.numel()
            counts["misses"] += int((~sample.fired).sum().item())
            return sample

        # The remaining branches are temporary compatibility behavior for tuple
        # callers and will disappear once every production consumer is event-aware.
        if jitter_on and std > 0.0 and mode == "time":
            output = _emit_spike_time_core(
                output,
                out_domain,
                noise_std=std,
                noise_kind=kind,
            )
        if hazard_on and have_potential:
            output = _apply_escape_hazard(
                output,
                potential,
                in_domain,
                out_domain,
                cfg,
            )

        # Tuple callers retain the pre-migration rail projection and declared domain.
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
