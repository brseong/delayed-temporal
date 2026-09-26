"""Gaussian spike-time noise and static range mismatch for TTFS models.

Dynamic non-ideality is represented by one additive Gaussian error in absolute
time units at each event-aware potential-to-spike boundary. The sampled raw time
jointly determines the delivered timestamp and whether the event arrived after
the encoder's fixed observation deadline. All encoder sites share one seeded,
advancing generator per evaluation replica.

Static range mismatch remains an independent experimental axis. It is sampled
once per supported module, stored as a frozen non-persistent buffer, and applied by
forward pre-hooks. Gaussian event timing and static mismatch therefore have distinct
configuration, sampling, and reporting paths.

This module also owns per-site Gaussian event and output-saturation counters. Its
configuration and counters are mutable process-wide state, so one noisy replica must
run in one process without DataParallel replication.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Literal, TypedDict

import torch
from torch import Tensor

from .clock import get_clock_driven, quantize_encoder_output
from .types import ClosedBounds, Potential, SpikeSample, TimeBounds


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

@dataclass
class GaussianTimeNoiseConfig:
    """Process-wide configuration for direct Gaussian spike-time noise.

    Standard deviations are fractions of each encoder's declared time window;
    linear and logarithmic encoders may override the shared fraction. Deadline
    grace is expressed in standard deviations. ``generator`` is seeded once and advances
    across calls, making a seed identify a complete replica rather than restarting
    the random sequence for each layer. A disabled configuration holds no generator
    so accidental sampling cannot silently consume an unrelated global RNG stream.
    """

    enabled: bool = False  # Select the event-aware Gaussian path at encoder boundaries.
    time_std_fraction: float = 0.0  # Standard deviation divided by the local window.
    linear_time_std_fraction: float | None = None  # Optional linear-code override.
    log_time_std_fraction: float | None = None  # Optional logarithmic-code override.
    time_mean: float = 0.0  # Absolute additive bias applied before deadline classification.
    deadline_margin_std_ratio: float = 0.0  # Grace measured in local noise stds.
    linear_deadline_margin_std_ratio: float | None = None
    log_deadline_margin_std_ratio: float | None = None
    seed: int = 0  # Replica seed used once when constructing the dedicated generator.
    generator: torch.Generator | None = None  # First selected device's stream.
    generators: dict[str, torch.Generator] = field(default_factory=dict)
    explicit_scope_required: bool = False


@dataclass(frozen=True)
class GaussianTimeNoiseScope:
    """One explicit model location in which timing noise may be active."""

    active: bool
    label: str


class GaussianNoiseCounts(TypedDict):
    """Statically known schema for one site's mutable Gaussian measurements.

    The set of metric names is fixed for type checking while repeated encoder and
    readout observations accumulate in place. Event and output totals are separate
    statistical denominators; deadline diagnostics retain scalar timing metadata.
    """

    events: int  # Sampled events excluding explicitly inactive signed branches.
    misses: int  # Number of sampled events arriving after the fixed deadline.
    deadline_events: int  # Nominal codewords exactly at the inclusive deadline.
    deadline_ulp_min: float  # Smallest deadline ULP observed for this site.
    deadline_ulp_max: float  # Largest deadline ULP observed for this site.
    outputs: int  # Number of analog readout values observed before rail clamping.
    output_underflows: int  # Readouts strictly below the declared output minimum.
    output_overflows: int  # Readouts strictly above the declared output maximum.


# The direct timing model has one shared configuration and one random stream per
# configured device. Configuration is
# replaced atomically by the setter instead of mutating its fields across calls.
_GLOBAL_GAUSSIAN_TIME_CONFIG = GaussianTimeNoiseConfig()

# Scoped ViT experiments activate timing noise only while a selected encoder block
# executes. Context-local state restores correctly across nested calls and exceptions,
# while the shared configuration continues to own the device random streams.
_GAUSSIAN_TIME_SCOPE: ContextVar[GaussianTimeNoiseScope | None] = ContextVar(
    "gaussian_time_noise_scope",
    default=None,
)

# Statistics are grouped first by a stable call-site name and then by counter name.
# Keeping this store separate from the configuration lets a new replica clear its
# measurements without coupling counter mutation to dataclass replacement.
_GAUSSIAN_NOISE_STATS: dict[str, GaussianNoiseCounts] = {}

# This mask affects accounting only, never sampling or the delivered values.
_GAUSSIAN_STATISTICS_MASK: ContextVar[Tensor | None] = ContextVar(
    "gaussian_statistics_mask", default=None,
)


@contextmanager
def gaussian_noise_statistics_mask(active: Tensor):
    """Exclude unused signed branches, including their internal readouts, from counts.

    Carrier calculations and random draws are deliberately retained so this change
    cannot change a seeded model output. Nested masks intersect and restore on exit.
    """
    if not isinstance(active, Tensor) or active.dtype != torch.bool:
        raise TypeError("Gaussian statistics mask must be a boolean tensor")
    parent = _GAUSSIAN_STATISTICS_MASK.get()
    token = _GAUSSIAN_STATISTICS_MASK.set(active if parent is None else parent & active)
    try:
        yield
    finally:
        _GAUSSIAN_STATISTICS_MASK.reset(token)


def _statistics_mask_for(value: Tensor) -> Tensor | None:
    mask = _GAUSSIAN_STATISTICS_MASK.get()
    if mask is None:
        return None
    if mask.device != value.device:
        raise ValueError("Gaussian statistics mask and value must share a device")
    # Never expand a scalar reference into one event per consumer.
    return torch.broadcast_to(mask, value.shape)


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


@contextmanager
def gaussian_time_noise_scope(*, active: bool, label: str):
    """Temporarily identify a model region selected for Gaussian timing noise.

    The scope changes neither the configured generator nor its counters. An inactive
    region therefore consumes no random draw and creates no statistics entry when
    ``explicit_scope_required`` is enabled for the current replica.
    """
    if not isinstance(active, bool):
        raise TypeError("Gaussian timing-noise scope active must be a bool")
    if not isinstance(label, str):
        raise TypeError("Gaussian timing-noise scope label must be a string")
    normalized_label = label.strip()
    if not normalized_label:
        raise ValueError("Gaussian timing-noise scope label must not be empty")
    token = _GAUSSIAN_TIME_SCOPE.set(
        GaussianTimeNoiseScope(active=active, label=normalized_label)
    )
    try:
        yield
    finally:
        _GAUSSIAN_TIME_SCOPE.reset(token)


def gaussian_time_noise_is_active() -> bool:
    """Return whether the current call site should use the noisy event path."""
    config = _GLOBAL_GAUSSIAN_TIME_CONFIG
    if not config.enabled:
        return False
    if not config.explicit_scope_required:
        return True
    scope = _GAUSSIAN_TIME_SCOPE.get()
    return scope is not None and scope.active


def _scoped_statistics_site(site: str) -> str:
    """Prefix a statistics site with its active explicit scope when required."""
    config = _GLOBAL_GAUSSIAN_TIME_CONFIG
    if not config.explicit_scope_required:
        return site
    scope = _GAUSSIAN_TIME_SCOPE.get()
    if scope is None or not scope.active:
        raise RuntimeError("Gaussian statistics require an active explicit scope")
    return f"{scope.label}/{site}"


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
        "deadline_events": 0,
        "deadline_ulp_min": math.inf,
        "deadline_ulp_max": 0.0,
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
    domain: ClosedBounds,
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
    if not gaussian_time_noise_is_active():
        return clamped

    # Count only readouts belonging to active branches, without changing values.
    counts = _stats_for(_scoped_statistics_site(site))
    mask = _statistics_mask_for(value)
    underflows, overflows = value < domain.min, value > domain.max
    if mask is None:
        total = value.new_tensor(value.numel(), dtype=torch.int64)
    else:
        total = mask.sum()
        underflows, overflows = underflows & mask, overflows & mask
    totals = torch.stack((total, underflows.sum(), overflows.sum())).to("cpu").tolist()
    counts["outputs"] += int(totals[0])
    counts["output_underflows"] += int(totals[1])
    counts["output_overflows"] += int(totals[2])

    # Return the previously computed bounded tensor; statistics never modify the
    # physical value delivered to the next operator.
    return clamped


def set_gaussian_time_noise(
    *,
    enabled: bool,
    time_std_fraction: float = 0.0,
    linear_time_std_fraction: float | None = None,
    log_time_std_fraction: float | None = None,
    time_mean: float = 0.0,
    deadline_margin_std_ratio: float = 0.0,
    linear_deadline_margin_std_ratio: float | None = None,
    log_deadline_margin_std_ratio: float | None = None,
    seed: int = 0,
    device: torch.device | str | tuple[torch.device | str, ...] = "cpu",
    explicit_scope_required: bool = False,
) -> None:
    """Install process-wide direct Gaussian spike-time noise configuration.

    Each successful call starts a new experiment replica: it constructs and seeds
    one generator for each requested sampling device, replaces the complete global
    configuration, and clears measurements from the previous replica. Each generator
    then advances across encoder calls on its device; individual forwards must never reseed it.

    Args:
        enabled: Whether event-aware encoders apply direct Gaussian timing noise.
        time_std_fraction: Non-negative default fraction of each code window.
        linear_time_std_fraction: Optional non-negative linear-code override.
        log_time_std_fraction: Optional non-negative logarithmic-code override.
        time_mean: Additive Gaussian mean in absolute time units.
        deadline_margin_std_ratio: Non-negative grace measured in local standard deviations.
        linear_deadline_margin_std_ratio: Optional linear-code grace ratio.
        log_deadline_margin_std_ratio: Optional logarithmic-code grace ratio.
        seed: Integer seed for the first configured device; further devices use deterministic offsets.
        device: Device or ordered tuple of devices on which Gaussian timing samples are drawn.
        explicit_scope_required: Require an active :func:`gaussian_time_noise_scope`
            before sampling or recording statistics.

    Raises:
        TypeError: If ``enabled`` is not boolean or ``seed`` is not an integer.
        ValueError: If a timing parameter is non-finite or a fraction is negative.
        RuntimeError: If PyTorch cannot create or seed a generator on ``device``.
    """
    global _GLOBAL_GAUSSIAN_TIME_CONFIG

    # Reject ambiguous flag and seed values rather than accepting truthy strings or
    # truncating fractional seeds, both of which would obscure replica identity.
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not isinstance(explicit_scope_required, bool):
        raise TypeError("explicit_scope_required must be a bool")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    # Normalize numeric inputs once, then require finite physical parameters before
    # creating random state or disturbing the currently installed configuration.
    normalized_std = float(time_std_fraction)
    normalized_linear_std = (
        None if linear_time_std_fraction is None else float(linear_time_std_fraction)
    )
    normalized_log_std = None if log_time_std_fraction is None else float(log_time_std_fraction)
    normalized_mean = float(time_mean)
    normalized_margin = float(deadline_margin_std_ratio)
    normalized_linear_margin = (
        None
        if linear_deadline_margin_std_ratio is None
        else float(linear_deadline_margin_std_ratio)
    )
    normalized_log_margin = (
        None if log_deadline_margin_std_ratio is None else float(log_deadline_margin_std_ratio)
    )
    optional_values = (
        normalized_linear_std,
        normalized_log_std,
        normalized_linear_margin,
        normalized_log_margin,
    )
    if not all(
        math.isfinite(value)
        for value in (normalized_std, normalized_mean, normalized_margin)
    ) or not all(
        value is None or math.isfinite(value) for value in optional_values
    ):
        raise ValueError("Gaussian time-noise parameters must be finite")
    if any(
        value is not None and value < 0.0
        for value in (normalized_std, normalized_linear_std, normalized_log_std)
    ):
        raise ValueError("time standard-deviation fractions must be non-negative")
    if any(
        value is not None and value < 0.0
        for value in (
            normalized_margin,
            normalized_linear_margin,
            normalized_log_margin,
        )
    ):
        raise ValueError("deadline margins must be non-negative")

    # Keep the original seed and draw order on the first device. Further devices
    # receive independent streams; one replica still owns all of their states.
    generator = None
    generators: dict[str, torch.Generator] = {}
    if enabled:
        requested_devices = device if isinstance(device, tuple) else (device,)
        if not requested_devices:
            raise ValueError("Gaussian time noise requires at least one device")
        for index, requested in enumerate(requested_devices):
            normalized_device = torch.device(requested)
            if normalized_device.type == "cuda" and normalized_device.index is None:
                normalized_device = torch.device("cuda", torch.cuda.current_device())
            key = str(normalized_device)
            if key in generators:
                raise ValueError("Gaussian time noise devices must be distinct")
            selected = torch.Generator(device=normalized_device)
            device_seed = seed if index == 0 else (seed + index * 1_000_003) % (2**63 - 1)
            selected.manual_seed(device_seed)
            generators[key] = selected
        generator = next(iter(generators.values()))

    # Build the full replacement before touching shared state. Consequently, any
    # validation, device, or seed failure preserves both the old replica and counts.
    new_config = GaussianTimeNoiseConfig(
        enabled=enabled,
        time_std_fraction=normalized_std,
        linear_time_std_fraction=normalized_linear_std,
        log_time_std_fraction=normalized_log_std,
        time_mean=normalized_mean,
        deadline_margin_std_ratio=normalized_margin,
        linear_deadline_margin_std_ratio=normalized_linear_margin,
        log_deadline_margin_std_ratio=normalized_log_margin,
        seed=seed,
        generator=generator,
        generators=generators,
        explicit_scope_required=explicit_scope_required,
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
        domain: Fixed nominal TTFS encoding interval.

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

    # TimeBounds declares the nominal code interval. The sampler may derive a later
    # receiver cutoff from its upper endpoint and a validated observation margin.
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
    deadline_margin: float = 0.0,
) -> Tensor:
    """Return the analytic probability that a Gaussian event misses its deadline.

    This function evaluates the probability that
    ``t_nominal + N(time_mean, time_std)`` exceeds ``domain.max + deadline_margin``
    without drawing a random sample. Equality with that receiver cutoff is delivered,
    matching the inclusive rule used by the event sampler.

    Args:
        nominal_time: Deterministic encoder output inside ``domain``.
        time_std: Scalar or broadcastable absolute Gaussian standard deviation.
        domain: Fixed nominal TTFS encoding interval.
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

    normalized_margin = float(deadline_margin)
    if not math.isfinite(normalized_margin) or normalized_margin < 0.0:
        raise ValueError("deadline_margin must be finite and non-negative")

    # The receiver may accept events for a fixed grace interval after the nominal
    # code endpoint. This changes delivery classification without widening the code
    # rail exposed to downstream operators.
    deadline = nominal.new_tensor(float(domain.max) + normalized_margin)

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
    deadline_margin: float = 0.0,
) -> SpikeSample:
    """Sample one Gaussian timing error and classify event delivery.

    A single sampled timestamp determines both the delivered time and whether the
    event misses the receiver deadline. Delivered events retain that raw timestamp,
    including values outside the nominal code interval. Misses retain the receiver
    deadline as finite tensor storage while ``SpikeSample.fired`` preserves the
    physical distinction between a miss and an event delivered exactly at cutoff.

    Args:
        nominal_time: Deterministic encoder output inside ``domain``.
        time_std: Scalar or broadcastable absolute Gaussian standard deviation.
        domain: Fixed nominal TTFS encoding interval.
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

    normalized_margin = float(deadline_margin)
    if not math.isfinite(normalized_margin) or normalized_margin < 0.0:
        raise ValueError("deadline_margin must be finite and non-negative")

    # The receiver cutoff is inclusive. Keep the code interval separate because a
    # positive margin changes when the receiver stops waiting, not the encoder map.
    receiver_deadline_value = float(domain.max) + normalized_margin
    receiver_deadline = nominal.new_tensor(
        receiver_deadline_value
    )
    fired = raw_time <= receiver_deadline

    # Preserve every delivered raw timestamp. Only misses receive a finite carrier,
    # and their false delivery mask prevents consumers from treating it as an event.
    stored_time = torch.where(fired, raw_time, receiver_deadline)
    return SpikeSample(
        time=stored_time,
        domain=domain,
        fired=fired,
        observation_deadline=receiver_deadline_value,
    )
# ---------------------------------------------------------------------------
# Gaussian encoder injection boundary
# ---------------------------------------------------------------------------

def inject_spike_time_noise[**P, OutT: ClosedBounds](
    *,
    encoding: Literal["linear", "log"],
) -> Callable[
    [Callable[P, tuple[Tensor, OutT]]],
    Callable[P, tuple[Tensor, OutT] | SpikeSample],
]:
    """Decorate a deterministic encoder with event-aware Gaussian time noise.

    Ordinary calls preserve the encoder's ``(time, bounds)`` contract and never
    sample. A physical consumer opts into noise with
    ``return_spike_sample=True``; that request uses the process-wide Gaussian
    configuration to return one finite timestamp and its deadline-delivery mask.

    Keeping sampling at this boundary makes linear and logarithmic encoders share
    one advancing generator per configured device, one inclusive deadline rule, and one statistics schema.
    Their absolute standard deviations and deadline margins may be overridden in
    the same configuration for measured marginal-noise sensitivity experiments.

    Args:
        encoding: Encoding family selecting the optional configuration override.

    Returns:
        A wrapped encoder supporting deterministic tuples and explicit event-aware
        ``SpikeSample`` results.

    Raises:
        RuntimeError: If an event-aware result is requested while Gaussian noise is
            disabled or its enabled configuration has no generator.
        TypeError: If an event-aware encoder does not declare ``TimeBounds``.
    """

    if encoding not in ("linear", "log"):
        raise ValueError("encoding must be 'linear' or 'log'")

    def decorator(
        func: Callable[P, tuple[Tensor, OutT]],
    ) -> Callable[P, tuple[Tensor, OutT] | SpikeSample]:
        @wraps(func)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> tuple[Tensor, OutT] | SpikeSample:
            # Snapshot the process-wide configuration once so the parameters and
            # generator belong to the same replica for the complete encoder call.
            gaussian_cfg = get_gaussian_time_noise()
            return_spike_sample = bool(kwargs.get("return_spike_sample", False))

            # Run the deterministic encoder exactly once before deciding whether a
            # sample is needed. Timing error acts on the bounded nominal timestamp.
            output, out_domain = func(*args, **kwargs)

            # Clock-driven execution and continuous timing noise are separate axes.
            clock_cfg = get_clock_driven()
            if clock_cfg.enabled:
                if gaussian_cfg.enabled or return_spike_sample:
                    raise RuntimeError(
                        "clock-driven execution cannot be combined with Gaussian "
                        "spike-time noise"
                    )
                if not isinstance(out_domain, TimeBounds):
                    raise TypeError(
                        "clock-driven spike encoders must return TimeBounds"
                    )
                site = kwargs.get("noise_site", func.__name__)
                return quantize_encoder_output(output, out_domain, site=site)

            # Tensor-only consumers cannot represent a missed event, so they retain
            # the deterministic tuple even while a Gaussian replica is active.
            if not return_spike_sample:
                return out_domain.clamp(output), out_domain

            if not gaussian_cfg.enabled:
                raise RuntimeError(
                    "return_spike_sample requires enabled Gaussian time noise"
                )
            if not isinstance(out_domain, TimeBounds):
                raise TypeError("sampled spike encoders must return TimeBounds")
            generator = gaussian_cfg.generators.get(str(output.device))
            if not isinstance(generator, torch.Generator):
                raise RuntimeError(
                    "Gaussian timing noise has no generator for the encoder device"
                )

            # Explicitly unselected model regions retain the SpikeSample return type
            # expected by their physical consumers, but execute deterministically and
            # leave both the shared generator and statistics mapping untouched.
            if not gaussian_time_noise_is_active():
                nominal_time = out_domain.clamp(output)
                return SpikeSample(
                    time=nominal_time,
                    domain=out_domain,
                    fired=torch.ones_like(nominal_time, dtype=torch.bool),
                    observation_deadline=float(out_domain.max),
                )

            # An omitted override preserves the historical shared-parameter path.
            if encoding == "linear":
                time_std_fraction = gaussian_cfg.linear_time_std_fraction
                deadline_margin_std_ratio = gaussian_cfg.linear_deadline_margin_std_ratio
            else:
                time_std_fraction = gaussian_cfg.log_time_std_fraction
                deadline_margin_std_ratio = gaussian_cfg.log_deadline_margin_std_ratio
            if time_std_fraction is None:
                time_std_fraction = gaussian_cfg.time_std_fraction
            if deadline_margin_std_ratio is None:
                deadline_margin_std_ratio = gaussian_cfg.deadline_margin_std_ratio

            window_length = float(out_domain.max) - float(out_domain.min)
            if not math.isfinite(window_length) or window_length < 0.0:
                raise ValueError("Gaussian encoder window must be finite and non-negative")
            time_std = float(time_std_fraction) * window_length
            deadline_margin = float(deadline_margin_std_ratio) * time_std

            nominal_time = out_domain.clamp(output)
            sample = _sample_gaussian_spike_time(
                nominal_time,
                time_std=time_std,
                domain=out_domain,
                generator=generator,
                time_mean=gaussian_cfg.time_mean,
                deadline_margin=deadline_margin,
            )

            # Attribute sampled events to the physical values consumed downstream.
            site = kwargs.get("noise_site", func.__name__)
            counts = _stats_for(_scoped_statistics_site(site))
            deadline = nominal_time.new_tensor(float(out_domain.max))
            mask = _statistics_mask_for(sample.time)
            misses, endpoints = ~sample.fired, nominal_time == deadline
            if mask is None:
                total = sample.time.new_tensor(sample.time.numel(), dtype=torch.int64)
            else:
                total = mask.sum()
                misses, endpoints = misses & mask, endpoints & mask
            miss_and_endpoint_counts = torch.stack(
                (total, misses.sum(), endpoints.sum())
            ).to(device="cpu")
            counts["events"] += int(miss_and_endpoint_counts[0].item())
            counts["misses"] += int(miss_and_endpoint_counts[1].item())
            counts["deadline_events"] += int(miss_and_endpoint_counts[2].item())
            cpu_deadline = torch.tensor(
                float(out_domain.max),
                dtype=nominal_time.dtype,
                device="cpu",
            )
            deadline_ulp = float(
                (
                    torch.nextafter(
                        cpu_deadline, cpu_deadline.new_tensor(math.inf)
                    )
                    - cpu_deadline
                ).item()
            )
            counts["deadline_ulp_min"] = min(
                counts["deadline_ulp_min"], deadline_ulp
            )
            counts["deadline_ulp_max"] = max(
                counts["deadline_ulp_max"], deadline_ulp
            )
            return sample

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# C — static device mismatch (per-neuron frozen threshold offset)
# ---------------------------------------------------------------------------

def _range_mismatch_pre_hook(module, args):
    """Add one frozen normalized offset scaled by the module's input range."""
    pot: Potential = args[0]
    radius = 0.5 * (float(pot.domain.max) - float(pot.domain.min))
    offset = module._range_mismatch_unit_offset * radius
    return (Potential(pot.value + offset, pot.domain),) + tuple(args[1:])


def install_range_mismatch(
    model,
    range_std_fraction: float,
    enabled: bool = True,
    *,
    seed: int = 0,
):
    """Attach static range-relative offsets to every spiking encoder module.

    A unit Gaussian offset is sampled once per supported module and multiplied by
    half of that module's declared input-range width. The normalized offset is a
    frozen non-persistent buffer, is not resampled per forward, and remains
    independent of the caller's global RNG stream.

    Returns the list of hook handles (for optional removal); empty when disabled.
    """
    if not enabled or range_std_fraction <= 0.0:
        return []
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("mismatch seed must be an integer")
    if seed < 0:
        raise ValueError("mismatch seed must be non-negative")

    # Lazy import to avoid a package import cycle (spiking_ops imports utils.transforms).
    from utils.transformers.models.spiking_ops import (
        SpikingLayerNorm, SpikingLinear, SpikingConv2d,
    )

    targets = []
    for _, m in model.named_modules():
        if isinstance(m, SpikingLayerNorm):
            shape = tuple(m.normalized_shape)          # broadcasts over the trailing feature dim
        elif isinstance(m, SpikingConv2d):
            shape = (1, m.in_channels, 1, 1)           # broadcasts over the channel dim of [B,C,H,W]
        elif isinstance(m, SpikingLinear):
            shape = (m.in_features,)                   # broadcasts over the trailing feature dim
        else:
            continue

        targets.append((m, shape))

    if not targets:
        return []

    devices = {m.weight.device for m, _ in targets}
    if len(devices) != 1:
        raise ValueError("static mismatch requires all target modules on one device")
    generator = torch.Generator(device=next(iter(devices)))
    generator.manual_seed(seed)

    handles = []
    for m, shape in targets:

        w = m.weight
        unit_offset = torch.randn(
            *shape,
            device=w.device,
            dtype=w.dtype,
            generator=generator,
        ) * float(range_std_fraction)
        m.register_buffer("_range_mismatch_unit_offset", unit_offset, persistent=False)
        handles.append(m.register_forward_pre_hook(_range_mismatch_pre_hook))

    return handles
