"""Layer-wise activation calibration contracts for fixed-bound inference."""

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import torch
from torch import Tensor


CALIBRATION_FORMAT_VERSION = 1


class CalibrationMode(StrEnum):
    """Select one phase of the layer-wise calibration workflow.

    String-valued members are suitable for command-line choices and JSON metadata
    without a second translation table. Collection measures deterministic layer
    activations, validation applies a previously fixed calibration table while
    reporting clipping, and inference applies the same immutable ranges during the
    evaluated model run.

    No member combines measurement and clamping in one forward pass. Keeping those
    phases distinct prevents a layer from defining its range from the activation it
    is currently supposed to validate or constrain.
    """

    # Collection is the only phase allowed to update min-max observers. Gaussian
    # timing noise and static perturbations will be rejected by later configuration.
    COLLECT = "collect"

    # Validation consumes frozen ranges and reports clipping without widening them.
    VALIDATE = "validate"

    # Inference shares validation's fixed-range behavior for the measured run.
    INFERENCE = "inference"


class CalibrationPass(StrEnum):
    """Identify the active deterministic collection pass.

    The min-max pass discovers only signed endpoints and element counts. The
    histogram pass then replays the same dataset against bins fixed from those
    endpoints. Keeping this phase explicit prevents a histogram from changing its
    own edges while it is collecting values.
    """

    # The first pass retains no activation tensors or distribution samples.
    MIN_MAX = "min_max"

    # The second pass reuses, but never updates, the completed first-pass extrema.
    HISTOGRAM = "histogram"


class CalibrationRangePolicy(StrEnum):
    """Select how statistical cutoffs and analytic endpoints form a fixed range.

    Signed symmetric sites calibrate both distribution tails and use their larger
    magnitude for a zero-centered physical rail. Lower-bounded and upper-bounded
    sites preserve one finite analytic endpoint and calibrate only the unbounded
    direction. Fully analytic operators do not need a calibration policy or record.
    """

    # LayerNorm, affine, and residual boundaries can use one symmetric magnitude even
    # when their observed distribution is not statistically symmetric.
    SIGNED_SYMMETRIC = "signed_symmetric"

    # Attention scores use a calibrated symmetric magnitude but cannot exceed the
    # analytic radius representable by their dtype, temporal scale, and reduction
    # capacity. Both fixed endpoints persist that symmetric execution ceiling.
    SIGNED_SYMMETRIC_CEILING = "signed_symmetric_ceiling"

    # ReLU and GELU-like outputs keep a known lower endpoint and calibrate the upper
    # tail that would otherwise leave the physical range unbounded.
    LOWER_BOUNDED = "lower_bounded"

    # This is the exact mirror for an operator with a finite analytic upper endpoint.
    UPPER_BOUNDED = "upper_bounded"


@dataclass(frozen=True)
class LayerCalibrationSpec:
    """Configure range selection for one explicitly named calibration site.

    Optional fields are policy-dependent: an ordinary symmetric site requires both
    quantiles and no fixed endpoint, while a ceiling-constrained symmetric site also
    requires a pair of equal-magnitude fixed endpoints. A lower-bounded site requires
    ``fixed_min`` and only an upper quantile, and an upper-bounded site requires
    ``fixed_max`` and only a lower quantile. Margin expands calibrated sides before a
    symmetric ceiling is applied; one-sided analytic endpoints remain fixed.
    """

    module_name: str
    tensor_name: str
    range_policy: CalibrationRangePolicy
    lower_quantile: float | None
    upper_quantile: float | None
    margin_fraction: float
    fixed_min: float | None = None
    fixed_max: float | None = None


@dataclass(frozen=True)
class CalibrationRange:
    """Store one immutable lower and upper activation bound.

    The range is the final result of layer-wise distribution calibration after the
    selected histogram cutoff and margin are applied. It remains separate from the
    raw extrema: forward execution consumes these fixed endpoints, while reports
    retain the raw
    extrema that produced them.

    Endpoint validation belongs to the range-construction function added in a later
    step. Keeping this declaration free of construction policy lets persistence load
    plain data before the same centralized validation is applied.
    """

    # These names match PotentialBounds so adapters can convert the persisted range
    # without translating lower/upper terminology at every layer boundary.
    min: float
    max: float


@dataclass
class MinMaxObserverState:
    """Hold mutable measurements collected for one layer activation.

    The first collection pass updates this state from unclamped, noise-free
    activations. Positive and negative values remain signed; no absolute-value
    reduction is used,
    because asymmetric activation distributions require independent lower and upper
    bounds.

    The empty state uses opposing infinities so the first finite observation can
    replace both endpoints with ordinary minimum and maximum operations. Update and
    validation behavior will be implemented as separate reviewed functions.
    """

    # The observer is intentionally mutable only during CalibrationMode.COLLECT.
    observed_min: float = math.inf
    observed_max: float = -math.inf

    # Counts refer to tensor elements, not batches, so calibration does not depend
    # on how the same examples are partitioned into evaluation batches.
    num_values: int = 0


def update_min_max_observer(
    state: MinMaxObserverState,
    value: Tensor,
) -> None:
    """Accumulate one real activation tensor into a min-max observer.

    The function measures signed extrema from the raw tensor supplied by a layer
    calibration hook and adds the number of tensor elements to the observer. Minima,
    maxima, and element counts are associative, so collecting the same examples in a
    different order or batch partition produces the same state.

    Args:
        state: Mutable observer owned by one named layer activation.
        value: Non-empty, finite, real floating-point activation tensor.

    Raises:
        TypeError: If either argument has the wrong type or the tensor is not a real
            floating-point activation.
        ValueError: If the observer is inconsistent, the tensor is empty, or its
            reduced extrema contain NaN or infinity.

    The tensor is detached before reduction so calibration never extends an autograd
    graph. Only Python scalar extrema and a count remain in the observer; no tensor,
    device allocation, or batch-shaped data is retained between calls.
    """
    # Validate the state before reading the tensor. A corrupted or externally
    # replaced observer must fail without partially changing its existing counts.
    if not isinstance(state, MinMaxObserverState):
        raise TypeError("state must be a MinMaxObserverState")
    if isinstance(state.num_values, bool) or not isinstance(state.num_values, int):
        raise TypeError("observer num_values must be an integer")
    if state.num_values < 0:
        raise ValueError("observer num_values must be non-negative")

    # The empty sentinel and populated representation are deliberately disjoint.
    # This catches manual endpoint edits before they can contaminate later batches.
    if state.num_values == 0:
        if state.observed_min != math.inf or state.observed_max != -math.inf:
            raise ValueError("empty observer must retain its initial extrema")
    elif (
        not math.isfinite(state.observed_min)
        or not math.isfinite(state.observed_max)
        or state.observed_min > state.observed_max
    ):
        raise ValueError("populated observer must contain ordered finite extrema")

    # Calibration targets analog activations. Reject empty, integral, Boolean, and
    # complex tensors because their ordering is absent or outside this range model.
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError("calibration activation must not be empty")
    if not value.is_floating_point():
        raise TypeError("calibration activation must be real floating point")

    # Detach before the device-side reduction so the observer never owns an autograd
    # edge. aminmax performs one signed reduction and preserves asymmetric extrema.
    batch_min_tensor, batch_max_tensor = value.detach().aminmax()
    batch_min = float(batch_min_tensor.item())
    batch_max = float(batch_max_tensor.item())

    # Raw activations are not guaranteed finite merely because their intended range
    # is finite. Reject numerical failures before mutating any part of the observer.
    if not math.isfinite(batch_min) or not math.isfinite(batch_max):
        raise ValueError("calibration activation must contain only finite values")

    # Commit all three associative measurements only after validation succeeds.
    # Python integers avoid overflow in long collection runs, while scalar extrema
    # keep the state independent of the source tensor's dtype and device.
    state.observed_min = min(state.observed_min, batch_min)
    state.observed_max = max(state.observed_max, batch_max)
    state.num_values += value.numel()


@dataclass
class HistogramObserverState:
    """Hold the mutable fixed-bin histogram collected in calibration pass two.

    bounds comes from the completed min-max pass and fixes uniformly spaced bin
    edges before the calibration dataset is replayed. bin_counts is a small
    one-dimensional integer tensor rather than stored activations, so memory usage is
    independent of dataset size.

    Tail counters retain values outside the first-pass range instead of silently
    assigning them to an edge bin. With deterministic evaluation and the same input
    set those counters should remain zero, but keeping them makes pass inconsistency
    observable.
    """

    # The immutable bounds and fixed-length count tensor define the histogram layout.
    # create_histogram_observer fixes the rank, integer dtype, and bin count.
    bounds: CalibrationRange
    bin_counts: Tensor

    # Counts use tensor elements as their denominator, matching the first min-max
    # pass regardless of batch partitioning.
    num_values: int = 0
    underflows: int = 0
    overflows: int = 0


def create_histogram_observer(
    state: MinMaxObserverState,
    *,
    bins: int,
    device: torch.device | str,
) -> HistogramObserverState:
    """Create the fixed histogram layout for calibration collection pass two.

    The completed min-max observer supplies immutable signed endpoints, while bins
    selects the resolution of uniformly spaced intervals between them. The returned
    integer counter tensor starts empty on the requested device so a later
    accumulation function can update counts without retaining raw activations.

    Args:
        state: Populated first-pass observer containing finite ordered extrema.
        bins: Number of histogram intervals; at least two are required to retain
            information about distribution shape.
        device: Device that will own the mutable counter tensor during collection.

    Returns:
        A second-pass observer with copied bounds, zeroed int64 bin counts, and
        independent zero-valued totals and tail counters.

    Raises:
        TypeError: If state is not a min-max observer, its count is not an integer,
            or bins is not an integer.
        ValueError: If the first pass is empty, its extrema are non-finite or
            unordered, or fewer than two bins are requested.
        RuntimeError: If PyTorch cannot allocate the counters on device.

    Equal first-pass endpoints are accepted for constant activations. Their special
    bin assignment is part of the later histogram accumulation function; this
    constructor does not widen an observed range with an arbitrary epsilon.
    """
    # Require a completed first-pass state before allocating any second-pass memory.
    # This prevents an empty sentinel range from becoming a persisted histogram.
    if not isinstance(state, MinMaxObserverState):
        raise TypeError("state must be a MinMaxObserverState")
    if isinstance(state.num_values, bool) or not isinstance(state.num_values, int):
        raise TypeError("observer num_values must be an integer")
    if state.num_values <= 0:
        raise ValueError("min-max observer must contain at least one value")

    # Histogram endpoints must be finite and ordered. Equality remains meaningful
    # for a layer whose complete calibration distribution is one constant value.
    if (
        not math.isfinite(state.observed_min)
        or not math.isfinite(state.observed_max)
        or state.observed_min > state.observed_max
    ):
        raise ValueError("min-max observer must contain ordered finite extrema")

    # Reject Boolean and fractional resolutions explicitly instead of allowing
    # torch.zeros to coerce them into an ambiguous counter shape.
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins < 2:
        raise ValueError("bins must be at least two")

    # Allocate only fixed-size counters on the selected collection device. The core
    # function has no implicit bin-count or device policy; evaluators provide both.
    bin_counts = torch.zeros(bins, dtype=torch.int64, device=device)

    # Copy Python scalar endpoints into a new frozen range and reset all second-pass
    # measurements. The first-pass element count is intentionally not carried over.
    bounds = CalibrationRange(
        min=float(state.observed_min),
        max=float(state.observed_max),
    )
    return HistogramObserverState(bounds=bounds, bin_counts=bin_counts)


def update_histogram_observer(
    state: HistogramObserverState,
    value: Tensor,
) -> None:
    """Accumulate one activation tensor into a fixed second-pass histogram.

    Every call uses the immutable endpoints and bin count created before pass two, so
    integer counts are invariant to evaluation order and batch partitioning. Values
    below or above the first-pass range are counted as explicit tails; values exactly
    on either endpoint remain in range, with the upper endpoint assigned to the last
    bin.

    Args:
        state: Mutable fixed-bin observer created by create_histogram_observer.
        value: Non-empty, finite, real floating-point activation tensor on the same
            device as the observer counter tensor.

    Raises:
        TypeError: If the observer layout, counters, or activation has an invalid
            type.
        ValueError: If the observer is inconsistent, the activation is empty or
            non-finite, or its device differs from the histogram counter device.

    Uniform interior boundaries follow half-open intervals [e_i, e_{i+1}) and the
    final interval includes the upper endpoint. A constant first-pass range has no
    nonzero-width intervals, so equal values are recorded in the center bin while
    strict deviations remain underflow or overflow observations.
    """
    # Validate the fixed observer layout before reducing the activation. The updater
    # never repairs malformed state because changing bins would invalidate prior data.
    if not isinstance(state, HistogramObserverState):
        raise TypeError("state must be a HistogramObserverState")
    if not isinstance(state.bounds, CalibrationRange):
        raise TypeError("histogram bounds must be a CalibrationRange")
    if not isinstance(state.bin_counts, Tensor):
        raise TypeError("histogram bin_counts must be a torch.Tensor")
    if state.bin_counts.ndim != 1 or state.bin_counts.numel() < 2:
        raise ValueError(
            "histogram bin_counts must be one-dimensional with at least two bins"
        )
    if state.bin_counts.dtype != torch.int64:
        raise TypeError("histogram bin_counts must use torch.int64")

    # Python counters are checked without synchronizing the device-resident bin
    # tensor. Tail counts cannot exceed the number of observed activation elements.
    for name in ("num_values", "underflows", "overflows"):
        count = getattr(state, name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"histogram {name} must be an integer")
        if count < 0:
            raise ValueError(f"histogram {name} must be non-negative")
    if state.underflows + state.overflows > state.num_values:
        raise ValueError("histogram tail counts exceed num_values")

    # Fixed endpoints must remain finite and ordered. A finite width is additionally
    # required for uniform interpolation unless the activation is exactly constant.
    lower = float(state.bounds.min)
    upper = float(state.bounds.max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("histogram bounds must be ordered and finite")
    width = upper - lower
    if lower != upper and not math.isfinite(width):
        raise ValueError("histogram width must be finite")

    # Calibration histograms describe analog activations and intentionally avoid
    # hidden device transfers during every layer hook invocation.
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError("calibration activation must not be empty")
    if not value.is_floating_point():
        raise TypeError("calibration activation must be real floating point")
    if value.device != state.bin_counts.device:
        raise ValueError("activation and histogram counters must share a device")

    # Detach before any reduction, then reject NaN or infinity using the reduced
    # extrema. No raw activation or autograd edge is retained after this call.
    detached = value.detach()
    batch_min_tensor, batch_max_tensor = detached.aminmax()
    batch_min = float(batch_min_tensor.item())
    batch_max = float(batch_max_tensor.item())
    if not math.isfinite(batch_min) or not math.isfinite(batch_max):
        raise ValueError("calibration activation must contain only finite values")

    # Classify strict tails before binning. Equality at both first-pass endpoints is
    # representable, and two scalar tail counts are transferred to Python together.
    underflow_mask = detached < lower
    overflow_mask = detached > upper
    in_range_values = detached[~(underflow_mask | overflow_mask)]
    tail_counts = torch.stack(
        (underflow_mask.count_nonzero(), overflow_mask.count_nonzero())
    ).tolist()
    batch_underflows = int(tail_counts[0])
    batch_overflows = int(tail_counts[1])

    # Build batch-local integer counts without mutating shared state. Constant
    # distributions use the center bin because every reconstructed edge is equal.
    bins = state.bin_counts.numel()
    if lower == upper:
        batch_counts = torch.zeros_like(state.bin_counts)
        batch_counts[bins // 2] = in_range_values.numel()
    else:
        # Float16 and bfloat16 inputs are promoted for supported, stable boundary
        # construction; unusually wide float32 ranges fall back to float64.
        working_dtype = (
            torch.float64 if detached.dtype == torch.float64 else torch.float32
        )
        if width > torch.finfo(working_dtype).max:
            working_dtype = torch.float64
        working_values = in_range_values.to(dtype=working_dtype)
        edges = torch.linspace(
            lower,
            upper,
            bins + 1,
            dtype=working_dtype,
            device=detached.device,
        )

        # right=True implements [e_i, e_{i+1}) for interior edges. Removing the two
        # outer edges keeps the exact lower endpoint in bin zero and upper in last.
        bin_indices = torch.bucketize(
            working_values,
            edges[1:-1],
            right=True,
        )
        batch_counts = torch.bincount(bin_indices, minlength=bins)

    # Commit integer bins and scalar totals only after every validation and temporary
    # calculation succeeds, so rejected batches leave the observer unchanged.
    state.bin_counts.add_(batch_counts)
    state.num_values += value.numel()
    state.underflows += batch_underflows
    state.overflows += batch_overflows


@dataclass(frozen=True)
class CalibrationHistogram:
    """Store an immutable, JSON-compatible layer activation histogram.

    Uniform bin edges are reconstructed from bounds and the length of bin_counts.
    Persisting integer counts rather than raw activations permits later percentile
    and margin studies without dataset-sized calibration artifacts.

    Underflow and overflow counts are stored separately because folding second-pass
    excursions into the edge bins would distort tail percentiles and hide a mismatch
    between the two deterministic calibration passes.
    """

    # Tuples make the persisted distribution transitively immutable and serialize
    # through dataclasses.asdict without a custom tensor encoder.
    bounds: CalibrationRange
    bin_counts: tuple[int, ...]

    # The total includes both in-range bins and explicit tail counts, allowing a
    # loader to verify that no observations were lost during persistence.
    num_values: int
    underflows: int
    overflows: int


def finalize_histogram_observer(
    state: HistogramObserverState,
) -> CalibrationHistogram:
    """Convert a completed mutable observer into immutable persistence data.

    Finalization copies device-resident integer bins to a CPU-backed tuple and
    verifies that in-range bins plus explicit tails exactly equal num_values. The
    returned object therefore contains no tensor storage, device dependency, or
    shared mutable state and can be serialized through dataclasses.asdict.

    Args:
        state: Populated second-pass histogram observer to finalize.

    Returns:
        An immutable histogram containing copied bounds, integer bin counts, the
        total number of observed elements, and separate tail counts.

    Raises:
        TypeError: If the observer layout, tensor dtype, or scalar counters have an
            invalid type.
        ValueError: If the observer is empty, has invalid bounds or negative counts,
            or its bins and tails do not sum to num_values.

    Finalization does not clear or otherwise mutate state. Collection code may retain
    the observer for diagnostics, while later changes to it cannot alter the immutable
    snapshot returned by this function.
    """
    # Validate the structural contract before transferring any tensor to the CPU.
    # A malformed layout cannot be repaired without changing histogram semantics.
    if not isinstance(state, HistogramObserverState):
        raise TypeError("state must be a HistogramObserverState")
    if not isinstance(state.bounds, CalibrationRange):
        raise TypeError("histogram bounds must be a CalibrationRange")
    if not isinstance(state.bin_counts, Tensor):
        raise TypeError("histogram bin_counts must be a torch.Tensor")
    if state.bin_counts.ndim != 1 or state.bin_counts.numel() < 2:
        raise ValueError(
            "histogram bin_counts must be one-dimensional with at least two bins"
        )
    if state.bin_counts.dtype != torch.int64:
        raise TypeError("histogram bin_counts must use torch.int64")

    # Persisted endpoints must be finite and ordered. Equal endpoints remain valid
    # for the explicitly supported constant-activation histogram representation.
    lower = float(state.bounds.min)
    upper = float(state.bounds.max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("histogram bounds must be ordered and finite")
    if lower != upper and not math.isfinite(upper - lower):
        raise ValueError("histogram width must be finite")

    # Scalar totals remain ordinary Python integers in the serialized schema. Reject
    # Boolean aliases and negative values before checking the complete count identity.
    for name in ("num_values", "underflows", "overflows"):
        count = getattr(state, name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"histogram {name} must be an integer")
        if count < 0:
            raise ValueError(f"histogram {name} must be non-negative")
    if state.num_values == 0:
        raise ValueError("histogram observer must contain at least one value")

    # One explicit device-to-CPU copy detaches the persistent representation from
    # accelerator memory. Validate every bin before constructing the immutable tuple.
    cpu_counts = state.bin_counts.detach().to(device="cpu")
    if bool((cpu_counts < 0).any().item()):
        raise ValueError("histogram bin counts must be non-negative")
    bin_counts = tuple(int(count) for count in cpu_counts.tolist())

    # Exact integer accounting detects lost values, double counting, or a tail that
    # was incorrectly folded into an edge bin before the table is written to disk.
    counted_values = sum(bin_counts) + state.underflows + state.overflows
    if counted_values != state.num_values:
        raise ValueError("histogram bins and tails must sum exactly to num_values")

    # Copy the frozen endpoints as plain Python floats and return a transitively
    # immutable snapshot whose contents cannot follow later observer mutations.
    bounds = CalibrationRange(min=lower, max=upper)
    return CalibrationHistogram(
        bounds=bounds,
        bin_counts=bin_counts,
        num_values=state.num_values,
        underflows=state.underflows,
        overflows=state.overflows,
    )


def select_histogram_quantile_range(
    histogram: CalibrationHistogram,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> CalibrationRange:
    """Select an outward-rounded range from a fixed activation histogram.

    Quantiles use the same closed ``[0, 1]`` convention as ``torch.quantile``.
    Because a histogram retains counts rather than values inside each bin, the exact
    sample quantile is unavailable. The lower endpoint is therefore rounded to the
    selected bin's left edge and the upper endpoint to its right edge. This preserves
    the complete boundary bins instead of claiming unsupported within-bin precision.

    Args:
        histogram: Immutable fixed-bin activation distribution from calibration.
        lower_quantile: Lower cumulative probability in the closed interval [0, 1].
        upper_quantile: Upper cumulative probability in the closed interval [0, 1].

    Returns:
        A finite ordered range aligned to the histogram's uniform bin edges.

    Raises:
        TypeError: If the histogram or quantile arguments have invalid types.
        ValueError: If the histogram is inconsistent, the quantiles are invalid, or
            a requested quantile falls in an explicit tail whose values were not
            retained by the histogram.

    This function applies no calibration margin. Keeping quantile selection and
    margin expansion separate makes the stored cutoff auditable and lets later margin
    studies reuse the same immutable histogram without collecting activations again.
    """
    # Validate the persisted representation even though the dataclass is frozen.
    # JSON loading or direct construction can still create structurally invalid data.
    if not isinstance(histogram, CalibrationHistogram):
        raise TypeError("histogram must be a CalibrationHistogram")
    if not isinstance(histogram.bounds, CalibrationRange):
        raise TypeError("histogram bounds must be a CalibrationRange")
    if not isinstance(histogram.bin_counts, tuple):
        raise TypeError("histogram bin_counts must be a tuple")
    if len(histogram.bin_counts) < 2:
        raise ValueError("histogram must contain at least two bins")

    # Quantiles are probabilities rather than percentages. Reject Boolean aliases,
    # non-scalar objects, NaN, infinity, reversed cutoffs, and values outside [0, 1].
    for name, quantile in (
        ("lower_quantile", lower_quantile),
        ("upper_quantile", upper_quantile),
    ):
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if lower_quantile > upper_quantile:
        raise ValueError("lower_quantile must not exceed upper_quantile")

    # Validate all integer accounting before locating ranks. Exact equality ensures
    # the cumulative distribution represents every recorded activation exactly once.
    scalar_counts = (
        ("num_values", histogram.num_values),
        ("underflows", histogram.underflows),
        ("overflows", histogram.overflows),
    )
    for name, count in scalar_counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"histogram {name} must be an integer")
        if count < 0:
            raise ValueError(f"histogram {name} must be non-negative")
    if histogram.num_values == 0:
        raise ValueError("histogram must contain at least one value")
    for count in histogram.bin_counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("histogram bin counts must be integers")
        if count < 0:
            raise ValueError("histogram bin counts must be non-negative")
    if (
        sum(histogram.bin_counts)
        + histogram.underflows
        + histogram.overflows
        != histogram.num_values
    ):
        raise ValueError("histogram bins and tails must sum exactly to num_values")

    # Floor the lower zero-based rank and ceil the upper rank so linear sample
    # quantiles are enclosed even when their positions lie between two observations.
    last_rank = histogram.num_values - 1
    lower_rank = math.floor(float(lower_quantile) * last_rank)
    upper_rank = math.ceil(float(upper_quantile) * last_rank)
    first_known_rank = histogram.underflows
    last_known_rank = histogram.num_values - histogram.overflows - 1
    if lower_rank < first_known_rank:
        raise ValueError("lower quantile falls inside the unrecorded underflow tail")
    if upper_rank > last_known_rank:
        raise ValueError("upper quantile falls inside the unrecorded overflow tail")

    # Convert global ranks to positions within the stored bins, then find the first
    # cumulative bin count strictly greater than each zero-based position.
    lower_position = lower_rank - histogram.underflows
    upper_position = upper_rank - histogram.underflows
    lower_bin = -1
    upper_bin = -1
    cumulative = 0
    for index, count in enumerate(histogram.bin_counts):
        cumulative += count
        if lower_bin < 0 and lower_position < cumulative:
            lower_bin = index
        if upper_position < cumulative:
            upper_bin = index
            break
    if lower_bin < 0 or upper_bin < 0:
        raise ValueError("quantile ranks are not represented by histogram bins")

    # Reconstruct uniform edges from the original signed extrema. Constant
    # activations retain their exact singleton range without artificial widening.
    observed_min = float(histogram.bounds.min)
    observed_max = float(histogram.bounds.max)
    if (
        not math.isfinite(observed_min)
        or not math.isfinite(observed_max)
        or observed_min > observed_max
    ):
        raise ValueError("histogram bounds must be ordered and finite")
    if observed_min == observed_max:
        return CalibrationRange(min=observed_min, max=observed_max)
    width = observed_max - observed_min
    if not math.isfinite(width):
        raise ValueError("histogram width must be finite")

    # Outward bin edges avoid inventing within-bin precision. Clamp the arithmetic
    # endpoints to the observed range to absorb only floating-point interpolation
    # error; this does not inspect or adapt to inference activations.
    bins = len(histogram.bin_counts)
    selected_min = observed_min + width * (lower_bin / bins)
    selected_max = observed_min + width * ((upper_bin + 1) / bins)
    return CalibrationRange(
        min=max(observed_min, selected_min),
        max=min(observed_max, selected_max),
    )


def apply_calibration_margin(
    bounds: CalibrationRange,
    *,
    margin_fraction: float,
) -> CalibrationRange:
    """Expand a selected calibration range by a fraction of its signed span.

    If the selected interval is ``[lower, upper]`` with width ``w``, this function
    returns ``[lower - margin_fraction * w, upper + margin_fraction * w]``. Applying
    the same additive distance to both sides preserves the interval center and avoids
    making the margin depend on an arbitrary activation offset from zero.

    Args:
        bounds: Finite ordered range selected from a calibration histogram.
        margin_fraction: Non-negative per-side fraction of the selected range width.

    Returns:
        A new immutable range with the requested symmetric span expansion.

    Raises:
        TypeError: If bounds or margin_fraction has an invalid type.
        ValueError: If the input range or margin is non-finite, the range is reversed,
            the margin is negative, or expansion would produce a non-finite endpoint.

    A constant range has zero width and therefore remains unchanged. Introducing an
    absolute epsilon here would silently depend on activation units; a genuinely
    variable site must instead supply a representative histogram or an explicit
    operator-derived interval.
    """
    # Validate the immutable input explicitly because direct construction and JSON
    # loading can bypass the functions that normally create calibration ranges.
    if not isinstance(bounds, CalibrationRange):
        raise TypeError("bounds must be a CalibrationRange")
    lower = float(bounds.min)
    upper = float(bounds.max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("calibration bounds must be ordered and finite")

    # The margin is a dimensionless scalar. Boolean values are rejected even though
    # Python treats them as integers, and negative fractions may not shrink a range.
    if isinstance(margin_fraction, bool) or not isinstance(
        margin_fraction, (int, float)
    ):
        raise TypeError("margin_fraction must be a real number")
    margin = float(margin_fraction)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_fraction must be finite and non-negative")

    # Compute the signed span once. A non-finite subtraction can occur even when both
    # endpoints are individually finite, so it is checked before multiplication.
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("calibration range width must be finite")
    expansion = width * margin
    if not math.isfinite(expansion):
        raise ValueError("calibration margin expansion must be finite")

    # Apply the same physical-distance margin to both endpoints, preserving the
    # selected interval center rather than scaling endpoints about zero.
    expanded_min = lower - expansion
    expanded_max = upper + expansion
    if not math.isfinite(expanded_min) or not math.isfinite(expanded_max):
        raise ValueError("expanded calibration bounds must be finite")

    # Return fresh immutable data and never modify the selected quantile range.
    # Zero margins and constant ranges naturally reproduce the original endpoints.
    return CalibrationRange(min=expanded_min, max=expanded_max)


def _normalize_layer_calibration_spec(
    spec: LayerCalibrationSpec,
) -> tuple[float | None, float | None, float, float | None, float | None]:
    """Validate one policy specification and return built-in numeric scalars."""
    # Stable identities mirror torch named_modules and distinguish multiple tensor
    # boundaries without allowing whitespace aliases.
    if not isinstance(spec, LayerCalibrationSpec):
        raise TypeError("spec must be a LayerCalibrationSpec")
    if not isinstance(spec.module_name, str):
        raise TypeError("module_name must be a string")
    if spec.module_name != spec.module_name.strip():
        raise ValueError("module_name must not contain surrounding whitespace")
    if not isinstance(spec.tensor_name, str):
        raise TypeError("tensor_name must be a string")
    if not spec.tensor_name or spec.tensor_name != spec.tensor_name.strip():
        raise ValueError(
            "tensor_name must be non-empty without surrounding whitespace"
        )
    if not isinstance(spec.range_policy, CalibrationRangePolicy):
        raise TypeError("range_policy must be a CalibrationRangePolicy")

    # Quantiles are optional only because one-sided policies intentionally omit the
    # cutoff corresponding to their fixed analytic endpoint.
    quantiles: list[float | None] = []
    for name, value in (
        ("lower_quantile", spec.lower_quantile),
        ("upper_quantile", spec.upper_quantile),
    ):
        if value is None:
            quantiles.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number or None")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be finite and lie in [0, 1]")
        quantiles.append(numeric)

    # Margin remains dimensionless and non-negative for every calibrated side.
    if isinstance(spec.margin_fraction, bool) or not isinstance(
        spec.margin_fraction, (int, float)
    ):
        raise TypeError("margin_fraction must be a real number")
    margin = float(spec.margin_fraction)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_fraction must be finite and non-negative")

    # Analytic endpoints, when present, must be ordinary finite physical values.
    fixed_endpoints: list[float | None] = []
    for name, value in (("fixed_min", spec.fixed_min), ("fixed_max", spec.fixed_max)):
        if value is None:
            fixed_endpoints.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number or None")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        fixed_endpoints.append(numeric)

    lower_quantile, upper_quantile = quantiles
    fixed_min, fixed_max = fixed_endpoints

    # Enforce an exact field shape for each policy so persistence cannot contain an
    # ignored quantile or endpoint that misrepresents how the range was obtained.
    if spec.range_policy is CalibrationRangePolicy.SIGNED_SYMMETRIC:
        if lower_quantile is None or upper_quantile is None:
            raise ValueError("signed symmetric policy requires both quantiles")
        if lower_quantile > upper_quantile:
            raise ValueError("lower_quantile must not exceed upper_quantile")
        if fixed_min is not None or fixed_max is not None:
            raise ValueError("signed symmetric policy does not accept fixed endpoints")
    elif spec.range_policy is CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING:
        if lower_quantile is None or upper_quantile is None:
            raise ValueError(
                "signed symmetric ceiling policy requires both quantiles"
            )
        if lower_quantile > upper_quantile:
            raise ValueError("lower_quantile must not exceed upper_quantile")
        if fixed_min is None or fixed_max is None:
            raise ValueError(
                "signed symmetric ceiling policy requires both fixed endpoints"
            )
        if fixed_min >= 0.0 or fixed_max <= 0.0 or fixed_min != -fixed_max:
            raise ValueError(
                "signed symmetric ceiling endpoints must be nonzero and symmetric"
            )
    elif spec.range_policy is CalibrationRangePolicy.LOWER_BOUNDED:
        if fixed_min is None or fixed_max is not None:
            raise ValueError(
                "lower-bounded policy requires fixed_min and forbids fixed_max"
            )
        if lower_quantile is not None or upper_quantile is None:
            raise ValueError(
                "lower-bounded policy requires only an upper quantile"
            )
    elif spec.range_policy is CalibrationRangePolicy.UPPER_BOUNDED:
        if fixed_max is None or fixed_min is not None:
            raise ValueError(
                "upper-bounded policy requires fixed_max and forbids fixed_min"
            )
        if lower_quantile is None or upper_quantile is not None:
            raise ValueError(
                "upper-bounded policy requires only a lower quantile"
            )
    else:
        raise ValueError(f"unsupported calibration range policy {spec.range_policy!r}")

    return lower_quantile, upper_quantile, margin, fixed_min, fixed_max


def select_calibration_policy_range(
    histogram: CalibrationHistogram,
    spec: LayerCalibrationSpec,
) -> CalibrationRange:
    """Build one fixed range using a site's statistical and analytic policy.

    Signed symmetric policy selects both histogram tails, encloses them in a
    zero-centered interval, and expands both calibrated endpoints. Its ceiling form
    then intersects that statistical interval with a symmetric analytic limit.
    Lower-bounded and upper-bounded policies retain their finite analytic endpoint
    exactly and apply margin only toward the calibrated unbounded side.

    Args:
        histogram: Completed signed activation histogram from deterministic replay.
        spec: Immutable site identity and policy-specific range controls.

    Returns:
        A finite fixed range ready for persistence and later runtime clipping.

    Raises:
        TypeError: If the specification, policy, optional endpoints, or quantiles have
            invalid types.
        ValueError: If policy fields are inconsistent, an analytic endpoint does not
            bound the observed distribution, or selection and expansion are invalid.

    Fully bounded operators intentionally bypass this function and retain their
    analytic range. They are not calibration sites and do not consume histogram or
    quantile configuration.
    """
    # Normalize all optional policy fields once. The same helper is used at collector
    # setup so malformed configuration fails before dataset execution begins.
    (
        lower_quantile,
        upper_quantile,
        margin,
        fixed_min,
        fixed_max,
    ) = _normalize_layer_calibration_spec(spec)

    # A symmetric rail calibrates both tails. Taking the larger selected magnitude
    # enforces exact zero-centered symmetry even for a biased activation distribution.
    if spec.range_policy is CalibrationRangePolicy.SIGNED_SYMMETRIC:
        selected = select_histogram_quantile_range(
            histogram,
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
        )
        radius = max(abs(float(selected.min)), abs(float(selected.max)))
        if not math.isfinite(radius):
            raise ValueError("symmetric calibration radius must be finite")
        return apply_calibration_margin(
            CalibrationRange(min=-radius, max=radius),
            margin_fraction=margin,
        )

    # Attention score calibration first applies the ordinary symmetric quantile and
    # margin policy, then intersects that statistical rail with a separately derived
    # representability ceiling. Calibration observations may exceed the ceiling: such
    # tails are exactly the approximation excursions frozen runtime must later count.
    if spec.range_policy is CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING:
        selected = select_histogram_quantile_range(
            histogram,
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
        )
        selected_radius = max(
            abs(float(selected.min)),
            abs(float(selected.max)),
        )
        expanded = apply_calibration_margin(
            CalibrationRange(min=-selected_radius, max=selected_radius),
            margin_fraction=margin,
        )
        ceiling_radius = float(fixed_max)
        final_radius = min(float(expanded.max), ceiling_radius)
        if not math.isfinite(final_radius) or final_radius <= 0.0:
            raise ValueError(
                "signed symmetric ceiling produced a non-positive final radius"
            )
        return CalibrationRange(min=-final_radius, max=final_radius)

    # A lower-bounded site preserves its analytic endpoint and selects only an upper
    # cutoff. The fixed endpoint must cover the complete observed calibration support,
    # not merely the chosen quantile, or it is not a valid analytic bound.
    if spec.range_policy is CalibrationRangePolicy.LOWER_BOUNDED:
        fixed_lower = float(fixed_min)
        selected = select_histogram_quantile_range(
            histogram,
            lower_quantile=0.0,
            upper_quantile=float(upper_quantile),
        )
        if fixed_lower > float(histogram.bounds.min):
            raise ValueError("fixed_min does not bound the observed activation minimum")
        base = CalibrationRange(min=fixed_lower, max=float(selected.max))
        width = base.max - base.min
        expansion = margin * width
        final_max = base.max + expansion
        if not math.isfinite(width) or width < 0.0 or not math.isfinite(final_max):
            raise ValueError("lower-bounded calibration expansion must be finite")
        return CalibrationRange(min=fixed_lower, max=final_max)

    # The upper-bounded policy mirrors the preceding construction. Its analytic
    # maximum stays exact while the selected lower tail alone receives margin.
    if spec.range_policy is CalibrationRangePolicy.UPPER_BOUNDED:
        fixed_upper = float(fixed_max)
        selected = select_histogram_quantile_range(
            histogram,
            lower_quantile=float(lower_quantile),
            upper_quantile=1.0,
        )
        if fixed_upper < float(histogram.bounds.max):
            raise ValueError("fixed_max does not bound the observed activation maximum")
        base = CalibrationRange(min=float(selected.min), max=fixed_upper)
        width = base.max - base.min
        expansion = margin * width
        final_min = base.min - expansion
        if not math.isfinite(width) or width < 0.0 or not math.isfinite(final_min):
            raise ValueError("upper-bounded calibration expansion must be finite")
        return CalibrationRange(min=final_min, max=fixed_upper)

    # StrEnum validation above makes this unreachable for current members, but an
    # explicit failure protects readers if a future enum member lacks implementation.
    raise ValueError(f"unsupported calibration range policy {spec.range_policy!r}")


@dataclass
class CalibrationClippingCounts:
    """Accumulate pre-clamp underflow and overflow counts for one layer.

    Validation and inference compare raw activations with the same frozen calibration
    range before clamping. Separate lower- and upper-tail counters make asymmetric
    range failures visible, and num_values supplies their common denominator.

    This type is mutable measurement state rather than calibration data. Updating
    these counters must never modify the immutable range or min-max observations from
    which that range was created.
    """

    # All observed elements contribute to the denominator, including values exactly
    # on a bound; equality is representable and is not counted as clipping.
    num_values: int = 0

    # Tail counts remain separate so reports can distinguish an offset error from a
    # range that is simply too narrow on both sides.
    underflows: int = 0
    overflows: int = 0


def clamp_with_calibration(
    value: Tensor,
    bounds: CalibrationRange,
    counts: CalibrationClippingCounts,
) -> Tensor:
    """Record strict range excursions and clamp an activation to frozen bounds.

    Validation and inference call this function before constructing the next
    ``Potential``. Values strictly below or above the immutable calibration range are
    added to separate counters, while values exactly on either endpoint remain valid.
    The returned tensor is clamped with ordinary PyTorch operations and therefore
    preserves the input dtype, device, shape, and autograd connectivity.

    Args:
        value: Non-empty, finite, real floating-point activation tensor.
        bounds: Frozen calibration range selected before this forward invocation.
        counts: Mutable clipping counters owned by the same layer activation.

    Returns:
        The activation clamped to the dtype-representable calibration endpoints.

    Raises:
        TypeError: If the tensor, range, counters, or stored counts have invalid
            types.
        ValueError: If the tensor is empty or non-finite, the range is invalid in
            Python or the tensor dtype, or the existing counters are negative.

    Every validation and device-side reduction completes before the counters mutate.
    A rejected activation therefore leaves prior clipping statistics unchanged.
    """
    # Validate the mutable accounting state first so an externally corrupted counter
    # cannot be carried forward or partially repaired by a successful tensor clamp.
    if not isinstance(counts, CalibrationClippingCounts):
        raise TypeError("counts must be CalibrationClippingCounts")
    for field_name in ("num_values", "underflows", "overflows"):
        count = getattr(counts, field_name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"clipping {field_name} must be an integer")
        if count < 0:
            raise ValueError(f"clipping {field_name} must be non-negative")
    if counts.underflows + counts.overflows > counts.num_values:
        raise ValueError("clipping tail counts exceed num_values")

    # Bounds loaded from persistence still require runtime validation. Equality is
    # allowed for an activation that calibration found to be exactly constant.
    if not isinstance(bounds, CalibrationRange):
        raise TypeError("bounds must be a CalibrationRange")
    lower = float(bounds.min)
    upper = float(bounds.max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("calibration bounds must be ordered and finite")

    # Calibration applies only to analog activation tensors. Reject invalid payloads
    # before allocating endpoint tensors or synchronizing device-side statistics.
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError("calibrated activation must not be empty")
    if not value.is_floating_point():
        raise TypeError("calibrated activation must be real floating point")
    detached = value.detach()
    batch_min_tensor, batch_max_tensor = detached.aminmax()
    if not bool(torch.isfinite(batch_min_tensor) and torch.isfinite(batch_max_tensor)):
        raise ValueError("calibrated activation must contain only finite values")

    # Materialize both rails in the activation dtype. Finite Python floats may
    # overflow or collapse when converted to float16 or bfloat16, so the physical
    # clamp must reject a representation that disagrees with persisted metadata.
    endpoints = value.new_tensor([lower, upper])
    if not bool(
        torch.isfinite(endpoints).all()
        and endpoints[0] <= endpoints[1]
    ):
        raise ValueError(
            "calibration bounds must remain finite and ordered in the tensor dtype"
        )

    # Count strict excursions on the source device before clamping. Endpoint equality
    # is representable and intentionally contributes only to the common denominator.
    batch_underflows = int((detached < endpoints[0]).sum().item())
    batch_overflows = int((detached > endpoints[1]).sum().item())
    clamped = value.clamp(min=endpoints[0], max=endpoints[1])

    # Commit scalar counts only after the output tensor has been produced. Python
    # integers avoid accumulator overflow across long validation or inference runs.
    counts.num_values += value.numel()
    counts.underflows += batch_underflows
    counts.overflows += batch_overflows
    return clamped


def calibration_clipping_rates(
    counts: CalibrationClippingCounts,
) -> tuple[float, float]:
    """Return lower- and upper-tail clipping fractions for one activation.

    The rates use tensor elements rather than batches as their denominator, matching
    min-max collection, histogram accumulation, and runtime clipping counts. Keeping
    the two tails separate preserves information about distribution shift direction.

    Args:
        counts: Populated mutable clipping counters from validation or inference.

    Returns:
        ``(underflow_rate, overflow_rate)`` as ordinary Python floats.

    Raises:
        TypeError: If the state or one of its counters has an invalid type.
        ValueError: If the state is empty, negative, or internally inconsistent.
    """
    # Validate all scalar accounting rather than allowing division to conceal a
    # malformed state behind a plausible-looking floating-point rate.
    if not isinstance(counts, CalibrationClippingCounts):
        raise TypeError("counts must be CalibrationClippingCounts")
    for field_name in ("num_values", "underflows", "overflows"):
        count = getattr(counts, field_name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"clipping {field_name} must be an integer")
        if count < 0:
            raise ValueError(f"clipping {field_name} must be non-negative")
    if counts.num_values == 0:
        raise ValueError("clipping counts must contain at least one value")
    if counts.underflows + counts.overflows > counts.num_values:
        raise ValueError("clipping tail counts exceed num_values")

    # Perform one explicit normalization after validation so both reported tails
    # share exactly the same element-count denominator.
    denominator = float(counts.num_values)
    return counts.underflows / denominator, counts.overflows / denominator


@dataclass(frozen=True)
class LayerCalibration:
    """Describe the frozen calibration result for one named layer activation.

    module_name is the stable name returned by torch.nn.Module.named_modules;
    tensor_name distinguishes boundaries such as input and output within that
    module. Together they form the lookup identity used by model adapters.

    The final range, raw signed extrema, fixed-bin histogram, policy-specific optional
    quantiles and analytic endpoints, element count, and applied margin are stored
    together so a result remains auditable after serialization. Runtime clipping
    counts remain separate from the immutable calibration table.
    """

    # Explicit module and tensor names avoid positional identities that change when
    # hooks are registered in a different order.
    module_name: str
    tensor_name: str

    # The fixed range is consumed during forward execution; the remaining fields
    # preserve how two-pass layer-wise calibration produced those endpoints.
    bounds: CalibrationRange
    observed_min: float
    observed_max: float
    num_values: int
    histogram: CalibrationHistogram

    # Policy inputs remain separate from the final bounds so persistence can reproduce
    # which tails were calibrated and which analytic endpoint stayed fixed.
    range_policy: CalibrationRangePolicy
    lower_quantile: float | None
    upper_quantile: float | None
    margin_fraction: float
    fixed_min: float | None
    fixed_max: float | None


def create_layer_calibration(
    spec: LayerCalibrationSpec,
    min_max: MinMaxObserverState,
    histogram: CalibrationHistogram,
) -> LayerCalibration:
    """Create one immutable layer record from two deterministic collection passes.

    The first-pass signed extrema must exactly match the fixed endpoints retained by
    the second-pass histogram, and both passes must observe the same number of tensor
    elements. Any histogram tail is rejected here: replaying the same calibration
    data through a deterministic model should not escape first-pass extrema, so a
    nonzero tail identifies an inconsistent pass rather than useful distribution data.

    Args:
        spec: Stable site identity and one policy-specific set of statistical and
            analytic range controls.
        min_max: Completed first-pass signed extrema observer.
        histogram: Finalized immutable second-pass histogram.

    Returns:
        A self-contained immutable calibration record with auditable selection data.

    Raises:
        TypeError: If names, observers, histogram fields, or numeric controls have
            invalid types.
        ValueError: If names are malformed, either pass is empty or inconsistent,
            histogram tails are nonzero, or range selection and expansion fail.

    This function produces statistical calibration data only. Operator-specific
    encoding constraints are applied when a later model adapter consumes the record.
    """
    # Validate the immutable policy before observer state. This fails malformed site
    # identities and policy field combinations consistently for direct callers and
    # records reconstructed from persistence.
    if not isinstance(spec, LayerCalibrationSpec):
        raise TypeError("spec must be a LayerCalibrationSpec")

    # Validate the completed first pass independently of the histogram. The empty
    # sentinel and manually corrupted extrema must never become persistent metadata.
    if not isinstance(min_max, MinMaxObserverState):
        raise TypeError("min_max must be a MinMaxObserverState")
    if isinstance(min_max.num_values, bool) or not isinstance(min_max.num_values, int):
        raise TypeError("min-max num_values must be an integer")
    if min_max.num_values <= 0:
        raise ValueError("min-max observer must contain at least one value")
    observed_min = float(min_max.observed_min)
    observed_max = float(min_max.observed_max)
    if (
        not math.isfinite(observed_min)
        or not math.isfinite(observed_max)
        or observed_min > observed_max
    ):
        raise ValueError("min-max observer must contain ordered finite extrema")

    # The immutable histogram carries the complete second pass. Strictly require the
    # same sample population and endpoints before any quantile policy is evaluated.
    if not isinstance(histogram, CalibrationHistogram):
        raise TypeError("histogram must be a CalibrationHistogram")
    if histogram.num_values != min_max.num_values:
        raise ValueError("calibration passes must observe the same number of values")
    if histogram.underflows != 0 or histogram.overflows != 0:
        raise ValueError("deterministic histogram replay must not contain tail values")
    if not isinstance(histogram.bounds, CalibrationRange):
        raise TypeError("histogram bounds must be a CalibrationRange")
    if (
        float(histogram.bounds.min) != observed_min
        or float(histogram.bounds.max) != observed_max
    ):
        raise ValueError("histogram bounds must exactly match first-pass extrema")

    # Policy selection is the single source of truth for symmetric calibration and
    # one-sided analytic endpoints. It also validates all optional policy fields.
    final_bounds = select_calibration_policy_range(histogram, spec)

    # Copy all scalar controls into ordinary Python representations so the frozen
    # dataclass remains independent of caller subclasses and JSON serialization policy.
    return LayerCalibration(
        module_name=spec.module_name,
        tensor_name=spec.tensor_name,
        bounds=final_bounds,
        observed_min=observed_min,
        observed_max=observed_max,
        num_values=min_max.num_values,
        histogram=histogram,
        range_policy=spec.range_policy,
        lower_quantile=(
            None if spec.lower_quantile is None else float(spec.lower_quantile)
        ),
        upper_quantile=(
            None if spec.upper_quantile is None else float(spec.upper_quantile)
        ),
        margin_fraction=float(spec.margin_fraction),
        fixed_min=None if spec.fixed_min is None else float(spec.fixed_min),
        fixed_max=None if spec.fixed_max is None else float(spec.fixed_max),
    )


@dataclass(frozen=True)
class CalibrationMetadata:
    """Identify the clean model and input configuration used for calibration.

    A calibration table is reusable only when the model family, checkpoint, dataset,
    preprocessing, numerical configuration, input capacity, and active model path
    match. model_options stores sorted immutable key-value pairs for model-family
    choices such as LayerNorm stages, attention backend, and activation function.

    Gaussian timing noise, static threshold mismatch, and weight or bias perturbation
    are not table identities. Calibration is collected from the clean deterministic
    model, then the same frozen table is reused while those robustness axes are
    measured through clipping and output statistics.
    """

    # These fields identify the pretrained model and representative input data.
    model_family: str
    model_id: str
    dataset_id: str
    dataset_split: str
    preprocessing: str

    # Numerical and TTFS parameters affect representable activations and therefore
    # must match before a persisted table can be reused.
    dtype: str
    theta: float
    tau_s: float
    tau_m: float
    clip_margin: float

    # Capacity metadata prevents a table collected for a shorter sequence or a
    # different image geometry from silently constraining a larger input.
    max_sequence_length: int | None
    input_shape: tuple[int, ...] | None

    # Tuples keep the frozen dataclass transitively immutable. Values are restricted
    # to JSON scalar types so persistence needs no repository-specific encoder.
    model_options: tuple[
        tuple[str, str | int | float | bool | None], ...
    ]


@dataclass(frozen=True)
class CalibrationTable:
    """Bundle metadata and all frozen layer-wise calibration results.

    The table is the persisted artifact loaded by validation and inference. Entries
    are stored as a tuple to prevent runtime insertion, deletion, or replacement;
    a later lookup function may build a private dictionary without changing this
    serialized source of truth.

    format_version makes incompatible persistence changes explicit. It describes
    only the file schema and is independent of the model checkpoint or experiment
    version recorded by CalibrationMetadata.
    """

    # The schema version is required rather than silently defaulted so every reader
    # must decide which persisted representation it supports.
    format_version: int

    # Metadata validates table reuse, while the immutable tuple contains one signed
    # activation range per stable module-and-tensor identity.
    metadata: CalibrationMetadata
    layers: tuple[LayerCalibration, ...]


def _validate_calibration_metadata(metadata: CalibrationMetadata) -> None:
    """Validate one immutable calibration-run identity without changing it."""
    # Dataclass construction does not enforce annotations at runtime, so every text
    # identity is checked before it participates in compatibility comparisons.
    if not isinstance(metadata, CalibrationMetadata):
        raise TypeError("metadata must be CalibrationMetadata")
    text_fields = (
        "model_family",
        "model_id",
        "dataset_id",
        "dataset_split",
        "preprocessing",
        "dtype",
    )
    for field_name in text_fields:
        value = getattr(metadata, field_name)
        if not isinstance(value, str):
            raise TypeError(f"metadata {field_name} must be a string")
        if not value or value != value.strip():
            raise ValueError(
                f"metadata {field_name} must be non-empty without surrounding whitespace"
            )

    # Numerical settings identify the physical and floating-point configuration.
    # Booleans are excluded even though Python treats them as integer subclasses.
    positive_fields = ("theta", "tau_s", "tau_m")
    for field_name in positive_fields:
        value = getattr(metadata, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metadata {field_name} must be a real number")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"metadata {field_name} must be finite and positive")
    if isinstance(metadata.clip_margin, bool) or not isinstance(
        metadata.clip_margin, (int, float)
    ):
        raise TypeError("metadata clip_margin must be a real number")
    if (
        not math.isfinite(metadata.clip_margin)
        or metadata.clip_margin < 0.0
        or metadata.clip_margin >= metadata.theta
    ):
        raise ValueError(
            "metadata clip_margin must be finite, non-negative, and below theta"
        )

    # Capacity fields are optional across model families, but supplied dimensions
    # must be strictly positive integers and remain transitively immutable tuples.
    if metadata.max_sequence_length is not None:
        if isinstance(metadata.max_sequence_length, bool) or not isinstance(
            metadata.max_sequence_length, int
        ):
            raise TypeError("metadata max_sequence_length must be an integer or None")
        if metadata.max_sequence_length <= 0:
            raise ValueError("metadata max_sequence_length must be positive")
    if metadata.input_shape is not None:
        if not isinstance(metadata.input_shape, tuple):
            raise TypeError("metadata input_shape must be a tuple or None")
        if not metadata.input_shape:
            raise ValueError("metadata input_shape must not be empty")
        for dimension in metadata.input_shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError("metadata input_shape dimensions must be integers")
            if dimension <= 0:
                raise ValueError("metadata input_shape dimensions must be positive")

    # Model options are sorted immutable pairs so equivalent configurations serialize
    # identically. Values remain JSON scalars and finite when represented as floats.
    if not isinstance(metadata.model_options, tuple):
        raise TypeError("metadata model_options must be a tuple")
    option_names: list[str] = []
    for option in metadata.model_options:
        if not isinstance(option, tuple) or len(option) != 2:
            raise TypeError("metadata model_options entries must be two-item tuples")
        option_name, option_value = option
        if not isinstance(option_name, str):
            raise TypeError("metadata model option names must be strings")
        if not option_name or option_name != option_name.strip():
            raise ValueError(
                "metadata model option names must be non-empty without surrounding whitespace"
            )
        if option_value is not None and not isinstance(
            option_value, (str, int, float, bool)
        ):
            raise TypeError("metadata model option values must be JSON scalars")
        if isinstance(option_value, float) and not math.isfinite(option_value):
            raise ValueError("metadata floating model options must be finite")
        option_names.append(option_name)
    if option_names != sorted(option_names) or len(option_names) != len(
        set(option_names)
    ):
        raise ValueError("metadata model_options must have sorted unique names")


def _validate_layer_calibration(layer: LayerCalibration) -> None:
    """Reconstruct and compare one layer record to enforce its complete contract."""
    # Reuse the public constructor as the single source of truth for pass identity,
    # zero-tail replay, quantile selection, and margin expansion.
    if not isinstance(layer, LayerCalibration):
        raise TypeError("calibration layers must be LayerCalibration instances")
    canonical = create_layer_calibration(
        LayerCalibrationSpec(
            module_name=layer.module_name,
            tensor_name=layer.tensor_name,
            range_policy=layer.range_policy,
            lower_quantile=layer.lower_quantile,
            upper_quantile=layer.upper_quantile,
            margin_fraction=layer.margin_fraction,
            fixed_min=layer.fixed_min,
            fixed_max=layer.fixed_max,
        ),
        MinMaxObserverState(
            observed_min=layer.observed_min,
            observed_max=layer.observed_max,
            num_values=layer.num_values,
        ),
        layer.histogram,
    )

    # Exact dataclass equality is appropriate because all endpoints were persisted as
    # JSON numbers and the canonical arithmetic is deterministic in Python float64.
    if layer != canonical:
        raise ValueError(
            "layer calibration bounds do not match its histogram, quantiles, and margin"
        )


def create_calibration_table(
    metadata: CalibrationMetadata,
    layers: Iterable[LayerCalibration],
    *,
    format_version: int = CALIBRATION_FORMAT_VERSION,
) -> CalibrationTable:
    """Validate and canonicalize a complete immutable calibration table.

    Layer records are sorted by their stable ``(module_name, tensor_name)`` identity
    so collection-hook order cannot change the serialized artifact. Duplicate names
    are rejected because runtime lookup must resolve every activation unambiguously.

    Args:
        metadata: Immutable identity of the model, data, and numerical configuration.
        layers: Finite iterable of completed immutable layer calibration records.
        format_version: Persistence schema version supported by this implementation.

    Returns:
        A validated table with a deterministic layer order.

    Raises:
        TypeError: If the version, metadata, iterable, or any layer has an invalid
            type.
        ValueError: If the version is unsupported, metadata or a layer is invalid,
            the table is empty, or a layer identity is duplicated.
    """
    # Refuse booleans and unknown schema versions before consuming the layer iterable.
    # Readers and writers therefore agree on one explicit representation.
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise TypeError("format_version must be an integer")
    if format_version != CALIBRATION_FORMAT_VERSION:
        raise ValueError(
            f"unsupported calibration format_version {format_version}; "
            f"expected {CALIBRATION_FORMAT_VERSION}"
        )

    # Metadata validation is independent of layer collection order and runs before
    # any iterator is consumed, giving configuration errors deterministic priority.
    _validate_calibration_metadata(metadata)
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Iterable):
        raise TypeError("layers must be an iterable of LayerCalibration records")
    layer_tuple = tuple(layers)
    if not layer_tuple:
        raise ValueError("calibration table must contain at least one layer")

    # Validate each complete record, then reject duplicate activation identities.
    # The key is composed only from standard module and tensor boundary names.
    seen_keys: set[tuple[str, str]] = set()
    for layer in layer_tuple:
        _validate_layer_calibration(layer)
        key = (layer.module_name, layer.tensor_name)
        if key in seen_keys:
            raise ValueError(
                "duplicate layer calibration for "
                f"module={layer.module_name!r}, tensor={layer.tensor_name!r}"
            )
        seen_keys.add(key)

    # Sorting makes byte-for-byte JSON output independent of hook registration and
    # distributed result-gather order without changing immutable record contents.
    canonical_layers = tuple(
        sorted(layer_tuple, key=lambda layer: (layer.module_name, layer.tensor_name))
    )
    return CalibrationTable(
        format_version=format_version,
        metadata=metadata,
        layers=canonical_layers,
    )


def validate_calibration_metadata(
    actual: CalibrationMetadata,
    expected: CalibrationMetadata,
) -> None:
    """Require an exact calibration identity match and report differing fields.

    Validation and inference call this before installing any layer ranges. An exact
    match prevents a table collected from another checkpoint, dataset split, input
    capacity, dtype, or ablation path from being silently reused.

    Args:
        actual: Metadata loaded from the persisted calibration table.
        expected: Metadata constructed from the requested evaluation configuration.

    Raises:
        TypeError: If either metadata object or one of its fields is invalid.
        ValueError: If one or more validated metadata fields differ.
    """
    # Validate both sides first so a malformed object is not reported merely as an
    # ordinary configuration mismatch.
    _validate_calibration_metadata(actual)
    _validate_calibration_metadata(expected)

    # Dataclass field order supplies a stable diagnostic while repr preserves enough
    # detail to distinguish paths, numerical values, shapes, and model options.
    mismatches = [
        field.name
        for field in fields(CalibrationMetadata)
        if getattr(actual, field.name) != getattr(expected, field.name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: table={getattr(actual, name)!r}, expected={getattr(expected, name)!r}"
            for name in mismatches
        )
        raise ValueError(f"calibration metadata mismatch: {details}")


def get_layer_calibration(
    table: CalibrationTable,
    module_name: str,
    tensor_name: str,
) -> LayerCalibration:
    """Resolve one required layer activation from a validated immutable table.

    Lookup never synthesizes a missing range from the current tensor. This function is
    intended for setup or hook installation; model adapters can retain the returned
    immutable record rather than scanning the table during every forward invocation.

    Args:
        table: Calibration table created or loaded through this module.
        module_name: Exact stable module name, including the empty root name if used.
        tensor_name: Exact non-empty activation boundary name.

    Returns:
        The unique immutable layer calibration with the requested identity.

    Raises:
        TypeError: If the table or lookup names have invalid types.
        ValueError: If the table or lookup names are malformed.
        KeyError: If the requested layer activation is absent.
    """
    # Canonical reconstruction validates direct dataclass construction, layer order,
    # duplicates, and every nested record before lookup trusts the table.
    if not isinstance(table, CalibrationTable):
        raise TypeError("table must be a CalibrationTable")
    canonical = create_calibration_table(
        table.metadata,
        table.layers,
        format_version=table.format_version,
    )
    if table != canonical:
        raise ValueError("calibration table is not in canonical layer order")

    # Apply the same name contract used during record creation. Root module lookup
    # deliberately accepts an empty module name, while tensor names remain explicit.
    if not isinstance(module_name, str):
        raise TypeError("module_name must be a string")
    if module_name != module_name.strip():
        raise ValueError("module_name must not contain surrounding whitespace")
    if not isinstance(tensor_name, str):
        raise TypeError("tensor_name must be a string")
    if not tensor_name or tensor_name != tensor_name.strip():
        raise ValueError("tensor_name must be non-empty without surrounding whitespace")

    # Linear search is paid only during setup and keeps the persisted table free of a
    # second mutable index. Callers retain the immutable result for forward execution.
    for layer in table.layers:
        if layer.module_name == module_name and layer.tensor_name == tensor_name:
            return layer
    raise KeyError(
        "missing layer calibration for "
        f"module={module_name!r}, tensor={tensor_name!r}"
    )


def calibration_table_to_dict(table: CalibrationTable) -> dict[str, object]:
    """Convert a validated calibration table to its canonical JSON object tree.

    The returned dictionary contains only strings, finite JSON numbers, booleans,
    nulls, lists or tuples, and nested dictionaries. Validation occurs first so no
    malformed directly constructed dataclass can be emitted as an apparently valid
    artifact.
    """
    # Rebuild the table to validate every nested invariant and canonical ordering.
    if not isinstance(table, CalibrationTable):
        raise TypeError("table must be a CalibrationTable")
    canonical = create_calibration_table(
        table.metadata,
        table.layers,
        format_version=table.format_version,
    )
    if table != canonical:
        raise ValueError("calibration table is not in canonical layer order")

    # dataclasses.asdict recursively copies frozen dataclasses, ensuring callers
    # cannot mutate the table through the serialization object returned here.
    return asdict(canonical)


def calibration_table_from_dict(payload: object) -> CalibrationTable:
    """Parse and validate one exact calibration-table JSON object tree.

    Unknown or missing fields are rejected at every nesting level. Numeric strings,
    Boolean integers, non-finite floats, mutable schema shortcuts, and malformed
    option pairs are not coerced. The result is passed through the same canonical
    constructors used for in-process calibration.

    Args:
        payload: Object produced by ``json.loads`` or an equivalent JSON decoder.

    Returns:
        A validated immutable calibration table in canonical layer order.

    Raises:
        TypeError: If an object has the wrong JSON-compatible type.
        ValueError: If keys, scalar values, nested records, or table invariants are
            invalid.
    """
    # Small local readers keep schema errors attached to their precise JSON path and
    # avoid permissive Python constructor coercions such as float("1.0").
    def require_mapping(
        value: object,
        required_keys: set[str],
        context: str,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a JSON object")
        keys = set(value.keys())
        if not all(isinstance(key, str) for key in keys):
            raise TypeError(f"{context} keys must be strings")
        if keys != required_keys:
            missing = sorted(required_keys - keys)
            unknown = sorted(keys - required_keys)
            raise ValueError(
                f"{context} fields differ from schema; missing={missing}, unknown={unknown}"
            )
        return value

    def require_list(value: object, context: str) -> list[object] | tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{context} must be a JSON array")
        return value

    def require_string(value: object, context: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{context} must be a string")
        return value

    def require_integer(value: object, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{context} must be an integer")
        return value

    def require_number(value: object, context: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{context} must be finite")
        return numeric

    def require_optional_number(value: object, context: str) -> float | None:
        if value is None:
            return None
        return require_number(value, context)

    def parse_range(value: object, context: str) -> CalibrationRange:
        range_payload = require_mapping(value, {"min", "max"}, context)
        return CalibrationRange(
            min=require_number(range_payload["min"], f"{context}.min"),
            max=require_number(range_payload["max"], f"{context}.max"),
        )

    # Parse the top-level schema and metadata before allocating nested layer records.
    table_payload = require_mapping(
        payload,
        {"format_version", "metadata", "layers"},
        "calibration table",
    )
    metadata_keys = {field.name for field in fields(CalibrationMetadata)}
    metadata_payload = require_mapping(
        table_payload["metadata"], metadata_keys, "calibration metadata"
    )

    # Convert optional capacity arrays and immutable sorted model-option pairs without
    # accepting arbitrary nested JSON values.
    input_shape_payload = metadata_payload["input_shape"]
    input_shape: tuple[int, ...] | None
    if input_shape_payload is None:
        input_shape = None
    else:
        input_shape = tuple(
            require_integer(dimension, "calibration metadata.input_shape[]")
            for dimension in require_list(
                input_shape_payload, "calibration metadata.input_shape"
            )
        )

    max_sequence_payload = metadata_payload["max_sequence_length"]
    max_sequence_length = (
        None
        if max_sequence_payload is None
        else require_integer(
            max_sequence_payload, "calibration metadata.max_sequence_length"
        )
    )
    model_options: list[tuple[str, str | int | float | bool | None]] = []
    for index, option_payload in enumerate(
        require_list(metadata_payload["model_options"], "calibration metadata.model_options")
    ):
        pair = require_list(
            option_payload, f"calibration metadata.model_options[{index}]"
        )
        if len(pair) != 2:
            raise ValueError("calibration metadata model option pairs must have length two")
        option_name = require_string(
            pair[0], f"calibration metadata.model_options[{index}][0]"
        )
        option_value = pair[1]
        if option_value is not None and not isinstance(
            option_value, (str, int, float, bool)
        ):
            raise TypeError("calibration metadata model option values must be JSON scalars")
        if isinstance(option_value, float) and not math.isfinite(option_value):
            raise ValueError("calibration metadata floating model options must be finite")
        model_options.append((option_name, option_value))

    metadata = CalibrationMetadata(
        model_family=require_string(metadata_payload["model_family"], "metadata.model_family"),
        model_id=require_string(metadata_payload["model_id"], "metadata.model_id"),
        dataset_id=require_string(metadata_payload["dataset_id"], "metadata.dataset_id"),
        dataset_split=require_string(
            metadata_payload["dataset_split"], "metadata.dataset_split"
        ),
        preprocessing=require_string(
            metadata_payload["preprocessing"], "metadata.preprocessing"
        ),
        dtype=require_string(metadata_payload["dtype"], "metadata.dtype"),
        theta=require_number(metadata_payload["theta"], "metadata.theta"),
        tau_s=require_number(metadata_payload["tau_s"], "metadata.tau_s"),
        tau_m=require_number(metadata_payload["tau_m"], "metadata.tau_m"),
        clip_margin=require_number(
            metadata_payload["clip_margin"], "metadata.clip_margin"
        ),
        max_sequence_length=max_sequence_length,
        input_shape=input_shape,
        model_options=tuple(model_options),
    )

    # Parse each complete layer with exact dataclass field sets. Nested histogram and
    # range objects are reconstructed explicitly before canonical validation.
    layer_keys = {field.name for field in fields(LayerCalibration)}
    histogram_keys = {field.name for field in fields(CalibrationHistogram)}
    layers: list[LayerCalibration] = []
    for index, layer_payload_object in enumerate(
        require_list(table_payload["layers"], "calibration table.layers")
    ):
        layer_context = f"calibration table.layers[{index}]"
        layer_payload = require_mapping(
            layer_payload_object, layer_keys, layer_context
        )
        histogram_context = f"{layer_context}.histogram"
        histogram_payload = require_mapping(
            layer_payload["histogram"], histogram_keys, histogram_context
        )
        bin_counts = tuple(
            require_integer(count, f"{histogram_context}.bin_counts[]")
            for count in require_list(
                histogram_payload["bin_counts"], f"{histogram_context}.bin_counts"
            )
        )
        histogram = CalibrationHistogram(
            bounds=parse_range(histogram_payload["bounds"], f"{histogram_context}.bounds"),
            bin_counts=bin_counts,
            num_values=require_integer(
                histogram_payload["num_values"], f"{histogram_context}.num_values"
            ),
            underflows=require_integer(
                histogram_payload["underflows"], f"{histogram_context}.underflows"
            ),
            overflows=require_integer(
                histogram_payload["overflows"], f"{histogram_context}.overflows"
            ),
        )
        layers.append(
            LayerCalibration(
                module_name=require_string(
                    layer_payload["module_name"], f"{layer_context}.module_name"
                ),
                tensor_name=require_string(
                    layer_payload["tensor_name"], f"{layer_context}.tensor_name"
                ),
                bounds=parse_range(layer_payload["bounds"], f"{layer_context}.bounds"),
                observed_min=require_number(
                    layer_payload["observed_min"], f"{layer_context}.observed_min"
                ),
                observed_max=require_number(
                    layer_payload["observed_max"], f"{layer_context}.observed_max"
                ),
                num_values=require_integer(
                    layer_payload["num_values"], f"{layer_context}.num_values"
                ),
                histogram=histogram,
                range_policy=CalibrationRangePolicy(
                    require_string(
                        layer_payload["range_policy"],
                        f"{layer_context}.range_policy",
                    )
                ),
                lower_quantile=require_optional_number(
                    layer_payload["lower_quantile"], f"{layer_context}.lower_quantile"
                ),
                upper_quantile=require_optional_number(
                    layer_payload["upper_quantile"], f"{layer_context}.upper_quantile"
                ),
                margin_fraction=require_number(
                    layer_payload["margin_fraction"], f"{layer_context}.margin_fraction"
                ),
                fixed_min=require_optional_number(
                    layer_payload["fixed_min"], f"{layer_context}.fixed_min"
                ),
                fixed_max=require_optional_number(
                    layer_payload["fixed_max"], f"{layer_context}.fixed_max"
                ),
            )
        )

    # The canonical constructor performs semantic validation, duplicate detection,
    # and deterministic sorting after the exact JSON shape has been reconstructed.
    return create_calibration_table(
        metadata,
        layers,
        format_version=require_integer(
            table_payload["format_version"], "calibration table.format_version"
        ),
    )


def save_calibration_table(
    table: CalibrationTable,
    path: str | Path,
) -> None:
    """Atomically write a validated calibration table as deterministic JSON.

    The target directory is created when absent. JSON keys and layer records use
    canonical order, non-finite numbers are forbidden, and a temporary file in the
    same directory is flushed before ``os.replace`` publishes the complete artifact.

    Args:
        table: Validated immutable calibration table to persist.
        path: Destination JSON path.

    Raises:
        TypeError: If the table or path has an invalid type.
        ValueError: If the table is invalid or path names no file.
        OSError: If directory creation, writing, flushing, or replacement fails.
    """
    # Validate and serialize before touching the filesystem. allow_nan=False provides
    # a final JSON-layer guard even though nested numerical fields are already checked.
    payload = calibration_table_to_dict(table)
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    # Accept the documented path forms without treating Boolean or arbitrary objects
    # as filesystem names. A final component is required for atomic replacement.
    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    target = Path(path)
    if not target.name:
        raise ValueError("calibration path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)

    # A same-directory temporary file keeps os.replace atomic on ordinary local
    # filesystems. Clean it on every failure without touching an existing target.
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_calibration_table(path: str | Path) -> CalibrationTable:
    """Load strict JSON and reconstruct a validated immutable calibration table.

    The decoder rejects JavaScript-style NaN and infinity constants before schema
    parsing. Missing files and I/O failures propagate normally, while malformed JSON
    or calibration records fail without producing a partial table.

    Args:
        path: Existing calibration JSON file.

    Returns:
        A canonical immutable table ready for metadata validation and setup lookup.

    Raises:
        TypeError: If path or a decoded field has an invalid type.
        ValueError: If JSON syntax, constants, schema, or calibration invariants are
            invalid.
        OSError: If the file cannot be read.
    """
    # Validate the path independently of file access for stable caller diagnostics.
    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    source = Path(path)
    if not source.name:
        raise ValueError("calibration path must name a file")

    # parse_constant blocks the non-standard NaN and Infinity tokens accepted by
    # Python's JSON decoder by default, preserving the finite-range schema contract.
    def reject_constant(constant: str) -> object:
        raise ValueError(f"non-finite JSON constant is not allowed: {constant}")

    payload = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    return calibration_table_from_dict(payload)


@dataclass
class CalibrationCollectorState:
    """Own mutable state for one complete two-pass calibration collection.

    The collector is passed explicitly to instrumentation rather than stored in a
    process-wide singleton. Each stable layer identity owns one min-max observer and,
    after the phase transition, one fixed-bin histogram observer. Policy
    specifications are validated at creation and exposed through a read-only mapping.

    ``finalized`` closes the collector after a table has been produced. This prevents
    later batches from silently changing observer state behind an already persisted
    calibration artifact.
    """

    metadata: CalibrationMetadata
    bin_count: int
    site_specs: Mapping[tuple[str, str], LayerCalibrationSpec]
    active_pass: CalibrationPass = CalibrationPass.MIN_MAX
    min_max_states: dict[tuple[str, str], MinMaxObserverState] = field(
        default_factory=dict
    )
    histogram_states: dict[tuple[str, str], HistogramObserverState] = field(
        default_factory=dict
    )
    finalized: bool = False


@dataclass
class CalibrationRuntimeState:
    """Apply one frozen calibration table during validation or inference.

    Runtime state contains no observers and therefore cannot derive or widen a range.
    The table is immutable; only per-layer clipping counters change as raw activation
    values are compared with their precomputed bounds.
    """

    mode: CalibrationMode
    table: CalibrationTable
    clipping_counts: dict[
        tuple[str, str], CalibrationClippingCounts
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerCalibrationClipping:
    """Snapshot clipping measurements for one stable layer activation.

    Counts and rates are copied from mutable runtime state so a report remains stable
    even if evaluation continues. Rates use tensor elements as their denominator and
    are zero before the layer has observed any value.
    """

    module_name: str
    tensor_name: str
    num_values: int
    underflows: int
    overflows: int
    underflow_rate: float
    overflow_rate: float


def _calibration_site_key(
    module_name: str,
    tensor_name: str,
) -> tuple[str, str]:
    """Validate and return one stable module-and-tensor calibration identity."""
    # Preserve the empty root-module name used by ``named_modules`` while rejecting
    # aliases that differ only through invisible surrounding whitespace.
    if not isinstance(module_name, str):
        raise TypeError("module_name must be a string")
    if module_name != module_name.strip():
        raise ValueError("module_name must not contain surrounding whitespace")

    # A tensor boundary must always be named because a module may expose more than
    # one activation requiring independent ranges.
    if not isinstance(tensor_name, str):
        raise TypeError("tensor_name must be a string")
    if not tensor_name or tensor_name != tensor_name.strip():
        raise ValueError(
            "tensor_name must be non-empty without surrounding whitespace"
        )
    return module_name, tensor_name


def create_calibration_collector(
    metadata: CalibrationMetadata,
    site_specs: Iterable[LayerCalibrationSpec],
    *,
    bin_count: int,
) -> CalibrationCollectorState:
    """Create an empty collector configured for deterministic two-pass calibration.

    Args:
        metadata: Complete model, data, numerical, and capacity identity.
        site_specs: Finite unique set of explicitly selected calibration boundaries
            and their layer-specific range policies.
        bin_count: Positive number of fixed-width bins created after pass one.

    Returns:
        A mutable collector initially accepting only min-max observations.

    Raises:
        TypeError: If metadata or policy controls have invalid types.
        ValueError: If metadata is incomplete or policy controls are out of range.
    """
    # Validate metadata at setup so an expensive dataset pass never starts with an
    # identity that cannot later be persisted or compared during evaluation.
    _validate_calibration_metadata(metadata)
    if isinstance(bin_count, bool) or not isinstance(bin_count, int):
        raise TypeError("bin_count must be an integer")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")

    # Materialize and validate the declared site set before any activation is seen.
    # Fully bounded operators are intentionally absent and continue to use analytic
    # propagation; collection cannot silently add an undeclared boundary later.
    try:
        specifications = tuple(site_specs)
    except TypeError as error:
        raise TypeError("site_specs must be an iterable of LayerCalibrationSpec") from error
    if not specifications:
        raise ValueError("site_specs must contain at least one calibration site")
    specifications_by_key: dict[tuple[str, str], LayerCalibrationSpec] = {}
    for spec in specifications:
        _normalize_layer_calibration_spec(spec)
        key = (spec.module_name, spec.tensor_name)
        if key in specifications_by_key:
            raise ValueError(
                "duplicate calibration site specification for "
                f"{spec.module_name!r}/{spec.tensor_name!r}"
            )
        specifications_by_key[key] = spec

    # The mutable collector owns its lookup mapping, while each contained policy is
    # frozen. No runtime output may create or replace a specification.
    return CalibrationCollectorState(
        metadata=metadata,
        bin_count=bin_count,
        site_specs=MappingProxyType(specifications_by_key),
    )


def observe_calibration_activation(
    state: CalibrationCollectorState,
    module_name: str,
    tensor_name: str,
    value: Tensor,
) -> None:
    """Accumulate one named raw activation in the collector's active pass.

    The first pass updates only signed extrema. During the second pass, histogram
    edges are constructed lazily on the activation device from the completed first
    pass and then remain fixed. A site absent from pass one is rejected rather than
    receiving bounds derived from its current second-pass value.

    Raises:
        TypeError: If the collector, identity, or activation has an invalid type.
        ValueError: If collection is closed, a site appears only in pass two, or the
            underlying observer rejects the activation.
    """
    # Validate ownership and lifecycle before resolving or mutating any site state.
    if not isinstance(state, CalibrationCollectorState):
        raise TypeError("state must be a CalibrationCollectorState")
    if state.finalized:
        raise ValueError("calibration collector is already finalized")
    key = _calibration_site_key(module_name, tensor_name)
    if key not in state.site_specs:
        raise ValueError(
            "activation is not a declared calibration site: "
            f"{module_name!r}/{tensor_name!r}"
        )

    # Pass one creates a small scalar observer per stable identity. The observer
    # function validates the tensor and commits only after all reductions succeed.
    if state.active_pass is CalibrationPass.MIN_MAX:
        observer = state.min_max_states.get(key)
        if observer is None:
            observer = MinMaxObserverState()
            update_min_max_observer(observer, value)
            state.min_max_states[key] = observer
        else:
            update_min_max_observer(observer, value)
        return

    # Pass two must be a replay of the known site set. This check prevents a late
    # conditional branch from calibrating itself from the tensor being observed.
    if state.active_pass is not CalibrationPass.HISTOGRAM:
        raise ValueError(f"unsupported calibration pass {state.active_pass!r}")
    min_max = state.min_max_states.get(key)
    if min_max is None:
        raise ValueError(
            "histogram pass encountered a site absent from the min-max pass: "
            f"{module_name!r}/{tensor_name!r}"
        )

    # Construct a new histogram locally and publish it only after the first update
    # succeeds. A non-finite or otherwise invalid tensor therefore leaves no partial
    # observer registered for this site.
    histogram = state.histogram_states.get(key)
    if histogram is None:
        candidate = create_histogram_observer(
            min_max,
            bins=state.bin_count,
            device=value.device,
        )
        update_histogram_observer(candidate, value)
        state.histogram_states[key] = candidate
    else:
        update_histogram_observer(histogram, value)


def start_histogram_calibration_pass(state: CalibrationCollectorState) -> None:
    """Close min-max collection and begin fixed-bin histogram replay.

    The transition is one-way and requires at least one completed first-pass site.
    Histogram observers remain lazy so every site's counters are allocated on the
    device of its first replayed activation without retaining calibration tensors.
    """
    # Reject repeated or post-finalization transitions before checking observer data.
    if not isinstance(state, CalibrationCollectorState):
        raise TypeError("state must be a CalibrationCollectorState")
    if state.finalized:
        raise ValueError("calibration collector is already finalized")
    if state.active_pass is not CalibrationPass.MIN_MAX:
        raise ValueError("histogram calibration pass has already started")

    # A zero-site pass usually indicates missing instrumentation and must not produce
    # an apparently valid empty artifact. Each observer is checked before transition.
    expected_keys = set(state.site_specs)
    observed_keys = set(state.min_max_states)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            "min-max calibration site mismatch: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    for observer in state.min_max_states.values():
        if observer.num_values <= 0:
            raise ValueError("min-max calibration contains an empty site observer")

    # No histogram may exist before the transition; preserving this invariant makes
    # it impossible to reuse stale second-pass state after a failed or repeated run.
    if state.histogram_states:
        raise ValueError("histogram state exists before the histogram pass")
    state.active_pass = CalibrationPass.HISTOGRAM


def finalize_calibration_collection(
    state: CalibrationCollectorState,
) -> CalibrationTable:
    """Finalize every replayed site and close the collector as an immutable table.

    All first-pass sites must appear in pass two, and no second-pass site can exist
    without a first-pass observer. Each layer constructor verifies population equality,
    zero replay tails, quantile selection, and margin expansion before the table is
    canonicalized by stable identity.
    """
    # Finalization is valid only once, after the explicit phase transition. The state
    # remains open if any validation below fails so callers can inspect the problem.
    if not isinstance(state, CalibrationCollectorState):
        raise TypeError("state must be a CalibrationCollectorState")
    if state.finalized:
        raise ValueError("calibration collector is already finalized")
    if state.active_pass is not CalibrationPass.HISTOGRAM:
        raise ValueError("histogram calibration pass has not started")

    # Compare complete identity sets before finalizing individual counters. This gives
    # a direct missing-site diagnostic instead of a later population mismatch.
    min_max_keys = set(state.min_max_states)
    histogram_keys = set(state.histogram_states)
    specification_keys = set(state.site_specs)
    if min_max_keys != histogram_keys or min_max_keys != specification_keys:
        missing = sorted(min_max_keys - histogram_keys)
        unexpected = sorted(histogram_keys - min_max_keys)
        raise ValueError(
            "calibration pass site mismatch: "
            f"missing_histograms={missing!r}, unexpected_histograms={unexpected!r}, "
            f"specifications_match={specification_keys == min_max_keys}"
        )

    # Build immutable layer records in stable order. Device counters are copied to
    # Python integers by histogram finalization and no activation tensor is retained.
    layers: list[LayerCalibration] = []
    for module_name, tensor_name in sorted(min_max_keys):
        histogram = finalize_histogram_observer(
            state.histogram_states[(module_name, tensor_name)]
        )
        layers.append(
            create_layer_calibration(
                state.site_specs[(module_name, tensor_name)],
                state.min_max_states[(module_name, tensor_name)],
                histogram,
            )
        )

    # Canonical table validation is the final transaction. Mark the collector closed
    # only after every site and metadata field has produced a valid immutable result.
    table = create_calibration_table(
        state.metadata,
        layers,
        format_version=CALIBRATION_FORMAT_VERSION,
    )
    state.finalized = True
    return table


def create_calibration_runtime(
    mode: CalibrationMode,
    table: CalibrationTable,
    *,
    expected_metadata: CalibrationMetadata,
) -> CalibrationRuntimeState:
    """Create frozen validation or inference state from a compatible table.

    Collection is deliberately unsupported because runtime state contains no observer
    capable of measuring a bound. The table must already be canonical, and its full
    metadata identity must exactly match the requested evaluation configuration.
    """
    # Require enum members rather than accepting arbitrary strings at this internal
    # boundary; command-line conversion belongs to evaluator argument parsing.
    if not isinstance(mode, CalibrationMode):
        raise TypeError("mode must be a CalibrationMode")
    if mode is CalibrationMode.COLLECT:
        raise ValueError("collection requires CalibrationCollectorState")

    # Rebuild the canonical form to validate every nested record, then reject a caller
    # that supplies the same records in a noncanonical order.
    if not isinstance(table, CalibrationTable):
        raise TypeError("table must be a CalibrationTable")
    canonical = create_calibration_table(
        table.metadata,
        table.layers,
        format_version=table.format_version,
    )
    if canonical != table:
        raise ValueError("calibration table is not in canonical layer order")
    validate_calibration_metadata(table.metadata, expected_metadata)

    # Prepopulate counters for exactly the frozen table identities. Runtime cannot
    # insert an unknown site or infer a fallback bound from its activation.
    clipping_counts = {
        (layer.module_name, layer.tensor_name): CalibrationClippingCounts()
        for layer in table.layers
    }
    return CalibrationRuntimeState(
        mode=mode,
        table=table,
        clipping_counts=clipping_counts,
    )


def apply_calibrated_activation(
    state: CalibrationRuntimeState,
    module_name: str,
    tensor_name: str,
    value: Tensor,
) -> Tensor:
    """Clamp one raw activation to its frozen layer range and record excursions.

    The lookup identity must already exist in the runtime table. Underflow and
    overflow are counted before clamping, while the returned tensor preserves shape,
    dtype, device, and autograd connectivity through ``Tensor.clamp``.
    """
    # Runtime state is valid only for the non-collecting phases. Check lifecycle before
    # looking up the activation so a malformed state cannot mutate counters.
    if not isinstance(state, CalibrationRuntimeState):
        raise TypeError("state must be a CalibrationRuntimeState")
    if state.mode not in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
        raise ValueError("calibration runtime must be validation or inference")
    key = _calibration_site_key(module_name, tensor_name)

    # Use the strict table lookup rather than constructing a range from the current
    # tensor. A missing entry is a configuration failure, never an adaptive fallback.
    layer = get_layer_calibration(state.table, module_name, tensor_name)
    counts = state.clipping_counts.get(key)
    if counts is None:
        raise ValueError(
            "calibration runtime has no clipping counter for "
            f"{module_name!r}/{tensor_name!r}"
        )

    # The low-level clamp performs finite-tensor and dtype endpoint checks atomically,
    # then records strict excursions before applying the immutable endpoints.
    return clamp_with_calibration(value, layer.bounds, counts)


def get_calibration_clipping_report(
    state: CalibrationRuntimeState,
) -> tuple[LayerCalibrationClipping, ...]:
    """Snapshot per-layer clipping counts and rates in canonical table order.

    The result is immutable and detached from continuing runtime updates. Sites not
    yet executed are retained with zero counts and rates so reports have a stable
    schema across early exits and limited-batch smoke evaluations.
    """
    # Validate phase and exact counter keys before reading any mutable count. This
    # detects accidental insertion, deletion, or table/runtime mismatches explicitly.
    if not isinstance(state, CalibrationRuntimeState):
        raise TypeError("state must be a CalibrationRuntimeState")
    if state.mode not in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
        raise ValueError("calibration runtime must be validation or inference")
    expected_keys = {
        (layer.module_name, layer.tensor_name) for layer in state.table.layers
    }
    actual_keys = set(state.clipping_counts)
    if actual_keys != expected_keys:
        raise ValueError("calibration runtime clipping sites do not match its table")

    # Copy scalar values in table order. Reuse centralized rate validation once a site
    # has observations, with an explicit all-zero representation before first use.
    report: list[LayerCalibrationClipping] = []
    for layer in state.table.layers:
        counts = state.clipping_counts[(layer.module_name, layer.tensor_name)]
        if counts.num_values == 0:
            if counts.underflows != 0 or counts.overflows != 0:
                raise ValueError("empty clipping counts must have zero tails")
            underflow_rate = 0.0
            overflow_rate = 0.0
        else:
            underflow_rate, overflow_rate = calibration_clipping_rates(counts)
        report.append(
            LayerCalibrationClipping(
                module_name=layer.module_name,
                tensor_name=layer.tensor_name,
                num_values=counts.num_values,
                underflows=counts.underflows,
                overflows=counts.overflows,
                underflow_rate=underflow_rate,
                overflow_rate=overflow_rate,
            )
        )

    # Tuple output prevents a caller from changing ordering or membership and mistaking
    # the edited report for the live runtime measurement state.
    return tuple(report)
