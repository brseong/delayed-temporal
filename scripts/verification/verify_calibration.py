#!/usr/bin/env python3
"""Verify layer-wise calibration collection, freezing, and persistence contracts."""

from __future__ import annotations

import ast
import copy
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transforms.calibration import (  # noqa: E402
    CALIBRATION_FORMAT_VERSION,
    CalibrationClippingCounts,
    CalibrationHistogram,
    CalibrationMetadata,
    CalibrationMode,
    CalibrationPass,
    CalibrationRange,
    CalibrationRangePolicy,
    HistogramObserverState,
    LayerCalibrationSpec,
    MinMaxObserverState,
    apply_calibrated_activation,
    apply_calibration_margin,
    calibration_clipping_rates,
    calibration_table_from_dict,
    calibration_table_to_dict,
    clamp_with_calibration,
    create_calibration_collector,
    create_calibration_runtime,
    create_calibration_table,
    create_histogram_observer,
    create_layer_calibration,
    finalize_calibration_collection,
    finalize_histogram_observer,
    get_calibration_clipping_report,
    get_layer_calibration,
    load_calibration_table,
    observe_calibration_activation,
    save_calibration_table,
    select_calibration_policy_range,
    select_histogram_quantile_range,
    start_histogram_calibration_pass,
    update_histogram_observer,
    update_min_max_observer,
    validate_calibration_metadata,
)
from utils.transforms.types import Potential, PotentialBounds  # noqa: E402
from utils.transformers.calibration import (  # noqa: E402
    bind_model_calibration,
    calibrated_potential,
    clear_model_calibration,
    model_calibration_is_bound,
    select_calibration_subset,
)
from utils.transformers.models.spiking_vit.calibration import (  # noqa: E402
    build_vit_calibration_metadata,
    collect_vit_calibration_table,
    image_processor_pixel_bounds,
    vit_calibration_specs,
    vit_residual_calibration_specs,
)
from utils.transformers.models.spiking_gpt2.calibration import (  # noqa: E402
    build_gpt2_calibration_metadata,
    collect_gpt2_calibration_table,
    gpt2_calibration_specs,
)


def _expect_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str | None = None,
) -> None:
    """Require one operation to fail with the requested diagnostic category."""
    try:
        operation()
    except exception_type as error:
        if message_fragment is not None:
            assert message_fragment in str(error), (
                f"expected {message_fragment!r} in {str(error)!r}"
            )
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def _make_layer(module_name: str, counts: tuple[int, ...]):
    """Construct a compact valid layer record for persistence checks."""
    num_values = sum(counts)
    min_max = MinMaxObserverState(-2.0, 2.0, num_values)
    histogram = CalibrationHistogram(
        bounds=CalibrationRange(-2.0, 2.0),
        bin_counts=counts,
        num_values=num_values,
        underflows=0,
        overflows=0,
    )
    return create_layer_calibration(
        LayerCalibrationSpec(
            module_name=module_name,
            tensor_name="output",
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=0.0,
            upper_quantile=1.0,
            margin_fraction=0.05,
        ),
        min_max,
        histogram,
    )


def _make_lower_bounded_layer(module_name: str):
    """Construct a lower-bounded record with optional policy fields for JSON tests."""
    histogram = CalibrationHistogram(
        bounds=CalibrationRange(0.0, 4.0),
        bin_counts=(1, 1, 1, 1),
        num_values=4,
        underflows=0,
        overflows=0,
    )
    return create_layer_calibration(
        LayerCalibrationSpec(
            module_name=module_name,
            tensor_name="output",
            range_policy=CalibrationRangePolicy.LOWER_BOUNDED,
            lower_quantile=None,
            upper_quantile=0.5,
            margin_fraction=0.1,
            fixed_min=0.0,
        ),
        MinMaxObserverState(0.0, 4.0, 4),
        histogram,
    )


def _make_metadata() -> CalibrationMetadata:
    """Construct one complete calibration identity reused across test groups."""
    return CalibrationMetadata(
        model_family="vit",
        model_id="checkpoint",
        dataset_id="imagenet-1k",
        dataset_split="train",
        preprocessing="resize=224",
        dtype="float32",
        theta=2000.0,
        tau_s=1.0,
        tau_m=1.0,
        clip_margin=1.0e-5,
        max_sequence_length=None,
        input_shape=(3, 224, 224),
        model_options=(
            ("spiking_attention", True),
            ("spiking_layernorm", True),
        ),
    )


# @lat: [[calibration#Layer-wise Calibration#Two-pass Collection#Observer and Histogram Invariants]]
def verify_observer_and_histogram_invariants() -> None:
    """Exercise deterministic extrema, fixed bins, tails, and error atomicity."""
    values = torch.tensor([-2.0, -1.0, -0.25, 0.5, 1.0, 2.0])

    # Associative min, max, and element counts must not depend on batch partitioning
    # or the order in which the same activation elements reach the observer.
    whole = MinMaxObserverState()
    update_min_max_observer(whole, values)
    partitioned = MinMaxObserverState()
    update_min_max_observer(partitioned, values[3:])
    update_min_max_observer(partitioned, values[:3])
    assert partitioned == whole == MinMaxObserverState(-2.0, 2.0, 6)

    # A non-finite batch must fail before any scalar field changes, ensuring a later
    # valid batch cannot inherit a partially committed calibration state.
    before = copy.deepcopy(partitioned)
    _expect_raises(
        ValueError,
        lambda: update_min_max_observer(
            partitioned, torch.tensor([0.0, float("nan")])
        ),
        "finite",
    )
    assert partitioned == before

    # Histogram edges come from pass one. Replaying the same elements in different
    # partitions must produce identical int64 bins and total element counts.
    histogram_whole = create_histogram_observer(whole, bins=4, device="cpu")
    update_histogram_observer(histogram_whole, values)
    histogram_partitioned = create_histogram_observer(whole, bins=4, device="cpu")
    update_histogram_observer(histogram_partitioned, values[3:])
    update_histogram_observer(histogram_partitioned, values[:3])
    assert torch.equal(
        histogram_partitioned.bin_counts, histogram_whole.bin_counts
    )
    assert histogram_partitioned.num_values == histogram_whole.num_values == 6

    # Strict excursions are stored outside edge bins while equality with either
    # endpoint remains representable in the first or final interval.
    tail_state = create_histogram_observer(whole, bins=4, device="cpu")
    update_histogram_observer(
        tail_state, torch.tensor([-3.0, -2.0, 2.0, 3.0])
    )
    frozen_tail = finalize_histogram_observer(tail_state)
    assert frozen_tail.underflows == 1 and frozen_tail.overflows == 1
    assert sum(frozen_tail.bin_counts) == 2
    assert sum(frozen_tail.bin_counts) + 2 == frozen_tail.num_values

    # Constant activation ranges stay exact and place all equal values in the center
    # bin instead of inventing a scale-dependent epsilon width.
    constant_min_max = MinMaxObserverState(3.0, 3.0, 3)
    constant = create_histogram_observer(constant_min_max, bins=3, device="cpu")
    update_histogram_observer(constant, torch.full((3,), 3.0))
    assert finalize_histogram_observer(constant).bin_counts == (0, 3, 0)


# @lat: [[calibration#Layer-wise Calibration#Two-pass Collection#Quantile and Margin Policy]]
def verify_quantile_and_margin_policy() -> None:
    """Check outward histogram quantiles and independent width-based margins."""
    histogram = CalibrationHistogram(
        bounds=CalibrationRange(-4.0, 4.0),
        bin_counts=(1, 1, 1, 1, 1, 1, 1, 1),
        num_values=8,
        underflows=0,
        overflows=0,
    )

    # Floor the lower rank and ceil the upper rank, then retain complete boundary
    # bins. The 25%-75% selection therefore resolves to [-3, 3] at this resolution.
    selected = select_histogram_quantile_range(
        histogram,
        lower_quantile=0.25,
        upper_quantile=0.75,
    )
    assert selected == CalibrationRange(-3.0, 3.0)
    assert select_histogram_quantile_range(
        histogram,
        lower_quantile=0.0,
        upper_quantile=1.0,
    ) == CalibrationRange(-4.0, 4.0)

    # A per-side 10% margin on width six adds 0.6 to both endpoints and preserves
    # the selected center. An offset range receives the same width-based distance.
    expanded = apply_calibration_margin(selected, margin_fraction=0.1)
    assert math.isclose(expanded.min, -3.6)
    assert math.isclose(expanded.max, 3.6)
    offset = apply_calibration_margin(
        CalibrationRange(100.0, 101.0), margin_fraction=0.1
    )
    assert math.isclose(offset.min, 99.9)
    assert math.isclose(offset.max, 101.1)

    # Signed calibration encloses independently selected tails in one exact symmetric
    # rail. Margin then expands both calibrated endpoints by the symmetric width.
    symmetric = select_calibration_policy_range(
        histogram,
        LayerCalibrationSpec(
            module_name="signed",
            tensor_name="output",
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=0.25,
            upper_quantile=0.75,
            margin_fraction=0.1,
        ),
    )
    assert math.isclose(symmetric.min, -3.6)
    assert math.isclose(symmetric.max, 3.6)

    # One-sided policies preserve their analytic endpoint exactly and apply the same
    # width-based margin only toward the statistically calibrated unbounded side.
    positive = CalibrationHistogram(
        CalibrationRange(0.0, 4.0), (1, 1, 1, 1), 4, 0, 0
    )
    lower_bounded = select_calibration_policy_range(
        positive,
        LayerCalibrationSpec(
            module_name="relu",
            tensor_name="output",
            range_policy=CalibrationRangePolicy.LOWER_BOUNDED,
            lower_quantile=None,
            upper_quantile=0.5,
            margin_fraction=0.1,
            fixed_min=0.0,
        ),
    )
    assert lower_bounded.min == 0.0
    assert math.isclose(lower_bounded.max, 3.3)

    negative = CalibrationHistogram(
        CalibrationRange(-4.0, 0.0), (1, 1, 1, 1), 4, 0, 0
    )
    upper_bounded = select_calibration_policy_range(
        negative,
        LayerCalibrationSpec(
            module_name="negative",
            tensor_name="output",
            range_policy=CalibrationRangePolicy.UPPER_BOUNDED,
            lower_quantile=0.5,
            upper_quantile=None,
            margin_fraction=0.1,
            fixed_max=0.0,
        ),
    )
    assert math.isclose(upper_bounded.min, -3.3)
    assert upper_bounded.max == 0.0

    # A claimed analytic endpoint must enclose the complete observed support. Policy
    # fields that are irrelevant to a one-sided range are rejected, not ignored.
    _expect_raises(
        ValueError,
        lambda: select_calibration_policy_range(
            positive,
            LayerCalibrationSpec(
                module_name="relu",
                tensor_name="output",
                range_policy=CalibrationRangePolicy.LOWER_BOUNDED,
                lower_quantile=None,
                upper_quantile=0.5,
                margin_fraction=0.0,
                fixed_min=0.5,
            ),
        ),
        "does not bound",
    )

    # Constant ranges remain constant, while quantiles whose ranks enter an explicit
    # unrecorded tail fail instead of fabricating unavailable activation values.
    assert apply_calibration_margin(
        CalibrationRange(3.0, 3.0), margin_fraction=10.0
    ) == CalibrationRange(3.0, 3.0)
    tailed = CalibrationHistogram(
        CalibrationRange(0.0, 4.0), (2, 2, 2, 2), 10, 1, 1
    )
    _expect_raises(
        ValueError,
        lambda: select_histogram_quantile_range(
            tailed, lower_quantile=0.0, upper_quantile=0.8
        ),
        "underflow tail",
    )
    _expect_raises(
        ValueError,
        lambda: apply_calibration_margin(selected, margin_fraction=-0.01),
        "non-negative",
    )


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Layer Record and Clipping]]
def verify_layer_record_and_clipping() -> None:
    """Check pass consistency, strict excursion counts, clamp, and gradients."""
    min_max = MinMaxObserverState(-4.0, 4.0, 8)
    histogram = CalibrationHistogram(
        CalibrationRange(-4.0, 4.0),
        (1, 1, 1, 1, 1, 1, 1, 1),
        8,
        0,
        0,
    )
    spec = LayerCalibrationSpec(
        module_name="encoder.layer.0",
        tensor_name="output",
        range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
        lower_quantile=0.25,
        upper_quantile=0.75,
        margin_fraction=0.1,
    )

    # The frozen record combines both completed passes with auditable quantile and
    # margin settings, but retains the raw extrema and full histogram unchanged.
    layer = create_layer_calibration(
        spec,
        min_max,
        histogram,
    )
    assert math.isclose(layer.bounds.min, -3.6)
    assert math.isclose(layer.bounds.max, 3.6)
    assert layer.histogram is histogram

    # Deterministic replay is a strict invariant: count mismatch and nonzero tails
    # both stop record creation rather than being hidden by a permissive quantile.
    _expect_raises(
        ValueError,
        lambda: create_layer_calibration(
            spec,
            MinMaxObserverState(-4.0, 4.0, 9),
            histogram,
        ),
        "same number",
    )
    tailed = CalibrationHistogram(
        CalibrationRange(-4.0, 4.0), (1, 1, 1, 1, 1, 1), 8, 1, 1
    )
    _expect_raises(
        ValueError,
        lambda: create_layer_calibration(
            spec,
            min_max,
            tailed,
        ),
        "tail",
    )

    # Strict endpoint comparisons count only genuine excursions. The clamped tensor
    # retains dtype, device, shape, and a valid autograd path to the source tensor.
    counts = CalibrationClippingCounts()
    value = torch.tensor([-4.0, -3.6, 0.0, 3.6, 5.0], requires_grad=True)
    clamped = clamp_with_calibration(value, layer.bounds, counts)
    assert clamped.dtype == value.dtype
    assert clamped.device == value.device and clamped.shape == value.shape
    assert torch.allclose(clamped, torch.tensor([-3.6, -3.6, 0.0, 3.6, 3.6]))
    assert (counts.num_values, counts.underflows, counts.overflows) == (5, 1, 1)
    assert calibration_clipping_rates(counts) == (0.2, 0.2)
    clamped.sum().backward()
    assert value.grad is not None

    # Non-finite input rejection is atomic: previously accumulated validation counts
    # remain unchanged when a later batch cannot be calibrated safely.
    snapshot = copy.deepcopy(counts)
    _expect_raises(
        ValueError,
        lambda: clamp_with_calibration(
            torch.tensor([0.0, float("nan")]), layer.bounds, counts
        ),
        "finite",
    )
    assert counts == snapshot


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Collection and Runtime Phase Separation]]
def verify_collection_and_runtime_phase_separation() -> None:
    """Check two-pass orchestration and frozen runtime lookup behavior."""
    metadata = _make_metadata()
    site_specs = tuple(
        LayerCalibrationSpec(
            module_name=f"encoder.layer.{index}",
            tensor_name="output",
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=0.0,
            upper_quantile=1.0,
            margin_fraction=0.0,
        )
        for index in (0, 1)
    )
    collector = create_calibration_collector(
        metadata,
        site_specs,
        bin_count=4,
    )
    assert collector.active_pass is CalibrationPass.MIN_MAX

    # Pass one receives two named distributions in deliberately split batches. No
    # histogram state may exist until the explicit one-way phase transition.
    observe_calibration_activation(
        collector, "encoder.layer.1", "output", torch.tensor([-1.0, 0.0])
    )
    observe_calibration_activation(
        collector, "encoder.layer.1", "output", torch.tensor([1.0, 2.0])
    )
    observe_calibration_activation(
        collector, "encoder.layer.0", "output", torch.tensor([-2.0, 2.0])
    )
    assert not collector.histogram_states
    start_histogram_calibration_pass(collector)
    assert collector.active_pass is CalibrationPass.HISTOGRAM

    # Replay the identical site populations with a different partition. A site that
    # was not present in pass one must fail before it can create histogram state.
    observe_calibration_activation(
        collector,
        "encoder.layer.1",
        "output",
        torch.tensor([-1.0, 0.0, 1.0, 2.0]),
    )
    observe_calibration_activation(
        collector, "encoder.layer.0", "output", torch.tensor([2.0, -2.0])
    )
    _expect_raises(
        ValueError,
        lambda: observe_calibration_activation(
            collector, "encoder.layer.2", "output", torch.tensor([0.0])
        ),
        "not a declared calibration site",
    )

    # Finalization sorts identities and permanently closes collection. Continuing to
    # observe after table creation cannot mutate the persisted calibration result.
    table = finalize_calibration_collection(collector)
    assert [layer.module_name for layer in table.layers] == [
        "encoder.layer.0",
        "encoder.layer.1",
    ]
    _expect_raises(
        ValueError,
        lambda: observe_calibration_activation(
            collector, "encoder.layer.0", "output", torch.tensor([0.0])
        ),
        "already finalized",
    )

    # Runtime setup accepts only frozen validation or inference. Applying a value
    # records strict excursions against the table and never adds an unknown site.
    _expect_raises(
        ValueError,
        lambda: create_calibration_runtime(
            CalibrationMode.COLLECT, table, expected_metadata=metadata
        ),
        "CalibrationCollectorState",
    )
    runtime = create_calibration_runtime(
        CalibrationMode.VALIDATE,
        table,
        expected_metadata=metadata,
    )
    raw = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
    clamped = apply_calibrated_activation(
        runtime, "encoder.layer.0", "output", raw
    )
    assert torch.equal(clamped, torch.tensor([-2.0, 0.0, 2.0]))
    clamped.sum().backward()
    assert raw.grad is not None
    _expect_raises(
        KeyError,
        lambda: apply_calibrated_activation(
            runtime, "encoder.missing", "output", torch.tensor([0.0])
        ),
        "missing layer calibration",
    )

    # Reports preserve canonical membership, including zero-valued rows for sites not
    # reached in a limited run, while copying live counters into immutable snapshots.
    report = get_calibration_clipping_report(runtime)
    assert len(report) == 2
    assert report[0].module_name == "encoder.layer.0"
    assert (report[0].num_values, report[0].underflows, report[0].overflows) == (
        3,
        1,
        1,
    )
    assert report[0].underflow_rate == report[0].overflow_rate == 1.0 / 3.0
    assert report[1].num_values == 0
    assert report[1].underflow_rate == report[1].overflow_rate == 0.0


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Model Binding and Potential Boundary]]
def verify_model_binding_and_potential_boundary() -> None:
    """Check explicit module binding, analytic collection rails, and frozen clamps."""

    class CalibrationModel(nn.Module):
        """Expose one stable named module for calibration integration checks."""

        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Identity()

    metadata = _make_metadata()
    spec = LayerCalibrationSpec(
        module_name="block",
        tensor_name="output",
        range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    collector = create_calibration_collector(metadata, (spec,), bin_count=4)
    model = CalibrationModel()
    assert not model_calibration_is_bound(model.block)

    # Binding resolves every declared name before mutation and rejects replicated or
    # already-bound models rather than sharing mutable counters ambiguously.
    assert bind_model_calibration(model, collector) == 1
    assert model_calibration_is_bound(model.block)
    _expect_raises(
        ValueError,
        lambda: bind_model_calibration(model, collector),
        "already has calibration state",
    )
    _expect_raises(
        RuntimeError,
        lambda: bind_model_calibration(nn.DataParallel(CalibrationModel()), collector),
        "DataParallel",
    )

    # Collection retains raw tensors on a static analytic safety rail. An escaped
    # value fails before observer mutation, and undeclared tensor names cannot appear.
    safety_bounds = PotentialBounds(-3.0, 3.0)
    raw = torch.tensor([-2.0, 0.0, 2.0])
    potential = calibrated_potential(
        model.block,
        "output",
        raw,
        collection_bounds=safety_bounds,
    )
    assert potential.value is raw and potential.domain is safety_bounds
    observer_snapshot = copy.deepcopy(collector.min_max_states[("block", "output")])
    _expect_raises(
        ValueError,
        lambda: calibrated_potential(
            model.block,
            "output",
            torch.tensor([4.0]),
            collection_bounds=safety_bounds,
        ),
        "escaped",
    )
    assert collector.min_max_states[("block", "output")] == observer_snapshot
    _expect_raises(
        ValueError,
        lambda: calibrated_potential(
            model.block,
            "unknown",
            torch.tensor([0.0]),
            collection_bounds=safety_bounds,
        ),
        "not a declared calibration site",
    )

    # Replay the same raw population against fixed histogram bins, finalize the table,
    # and remove only adapter attributes while retaining the completed collector data.
    start_histogram_calibration_pass(collector)
    calibrated_potential(
        model.block,
        "output",
        raw.flip(0),
        collection_bounds=safety_bounds,
    )
    table = finalize_calibration_collection(collector)
    assert clear_model_calibration(model, expected_state=collector) == 1
    assert not model_calibration_is_bound(model.block)
    _expect_raises(
        RuntimeError,
        lambda: calibrated_potential(model.block, "output", raw),
        "no complete calibration binding",
    )

    # Frozen validation ignores the collection safety rail, clamps against persisted
    # bounds, and attaches those exact endpoints to the returned Potential metadata.
    runtime = create_calibration_runtime(
        CalibrationMode.VALIDATE,
        table,
        expected_metadata=metadata,
    )
    assert bind_model_calibration(model, runtime) == 1
    evaluated = calibrated_potential(
        model.block,
        "output",
        torch.tensor([-4.0, 0.0, 4.0]),
    )
    assert torch.equal(evaluated.value, torch.tensor([-2.0, 0.0, 2.0]))
    assert evaluated.domain == PotentialBounds(-2.0, 2.0)
    report = get_calibration_clipping_report(runtime)
    assert (report[0].num_values, report[0].underflows, report[0].overflows) == (
        3,
        1,
        1,
    )

    # Cleanup uses an optional identity guard and does not erase the runtime report.
    wrong_state = create_calibration_runtime(
        CalibrationMode.INFERENCE,
        table,
        expected_metadata=metadata,
    )
    _expect_raises(
        ValueError,
        lambda: clear_model_calibration(model, expected_state=wrong_state),
        "does not match expected_state",
    )
    assert clear_model_calibration(model, expected_state=runtime) == 1
    assert get_calibration_clipping_report(runtime) == report


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Preprocessing-Derived Image Range]]
def verify_preprocessing_derived_image_range() -> None:
    """Verify fixed ViT pixel rails from processor rescaling and normalization."""
    # Three distinct channel statistics must reduce to one scalar range without
    # depending on an observed image batch. Floating-point endpoint arithmetic is
    # compared with tolerance because 1/255 cannot be represented exactly.
    processor = SimpleNamespace(
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=(0.5, 0.4, 0.3),
        image_std=(0.5, 0.2, 0.1),
    )
    bounds = image_processor_pixel_bounds(processor, num_channels=3)
    assert math.isclose(bounds.min, -3.0)
    assert math.isclose(bounds.max, 7.0)

    # A scalar field broadcasts over every configured channel. If normalization
    # produces an entirely positive interval, the returned PWM rail widens only to
    # zero so the shared reference event remains representable.
    positive_processor = SimpleNamespace(
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=-1.0,
        image_std=1.0,
    )
    assert image_processor_pixel_bounds(
        positive_processor,
        num_channels=3,
    ) == PotentialBounds(0.0, 2.0)

    # Disabled preprocessing stages retain the original uint8 range. Invalid
    # channel counts, metadata lengths, non-positive scales, and non-finite values
    # must fail during setup instead of creating batch-dependent fallback bounds.
    raw_processor = SimpleNamespace(do_rescale=False, do_normalize=False)
    assert image_processor_pixel_bounds(
        raw_processor,
        num_channels=1,
    ) == PotentialBounds(0.0, 255.0)
    _expect_raises(
        TypeError,
        lambda: image_processor_pixel_bounds(raw_processor, num_channels=True),
        "integer",
    )
    _expect_raises(
        ValueError,
        lambda: image_processor_pixel_bounds(raw_processor, num_channels=0),
        "positive",
    )
    _expect_raises(
        ValueError,
        lambda: image_processor_pixel_bounds(
            SimpleNamespace(
                do_rescale=True,
                rescale_factor=(1.0, 1.0),
                do_normalize=False,
            ),
            num_channels=3,
        ),
        "one or 3",
    )
    _expect_raises(
        ValueError,
        lambda: image_processor_pixel_bounds(
            SimpleNamespace(
                do_rescale=False,
                do_normalize=True,
                image_mean=0.0,
                image_std=0.0,
            ),
            num_channels=3,
        ),
        "positive",
    )
    _expect_raises(
        ValueError,
        lambda: image_processor_pixel_bounds(
            SimpleNamespace(
                do_rescale=True,
                rescale_factor=float("inf"),
                do_normalize=False,
            ),
            num_channels=3,
        ),
        "finite",
    )


# @lat: [[calibration#Layer-wise Calibration#Two-pass Collection#Deterministic Training Subset]]
def verify_deterministic_training_subset() -> None:
    """Verify seeded subset identity, metadata, and automatic two-pass replay."""
    import random

    from torch.utils.data import DataLoader, Dataset
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.calibration import model_calibration_is_bound

    class FingerprintedDataset:
        """Implement the small Hugging Face dataset protocol used by selection."""

        def __init__(self, rows, fingerprint: str) -> None:
            self.rows = tuple(rows)
            self._fingerprint = fingerprint

        def __len__(self) -> int:
            return len(self.rows)

        def shuffle(self, *, seed: int):
            indices = list(range(len(self.rows)))
            random.Random(seed).shuffle(indices)
            return FingerprintedDataset(
                (self.rows[index] for index in indices),
                f"{self._fingerprint}:shuffle:{seed}",
            )

        def select(self, indices):
            selected_indices = tuple(indices)
            return FingerprintedDataset(
                (self.rows[index] for index in selected_indices),
                f"{self._fingerprint}:select:{selected_indices}",
            )

    # The same source, seed, and prefix length must reconstruct the same ordered
    # population and fingerprint; changing the seed changes its artifact identity.
    source = FingerprintedDataset(range(10), "source-v1")
    first = select_calibration_subset(source, sample_count=4, seed=7)
    replay = select_calibration_subset(source, sample_count=4, seed=7)
    different = select_calibration_subset(source, sample_count=4, seed=8)
    assert first.rows == replay.rows
    assert first._fingerprint == replay._fingerprint
    assert different._fingerprint != first._fingerprint
    _expect_raises(
        ValueError,
        lambda: select_calibration_subset(source, sample_count=11, seed=7),
        "exceeds split size",
    )

    # Canonical metadata stores subset selection alongside every processor and model
    # option that affects the measured residual distributions.
    processor = SimpleNamespace(
        do_resize=True,
        size={"height": 4, "width": 4},
        do_center_crop=False,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    config = SimpleNamespace(
        image_size=4,
        num_channels=3,
        hidden_act="gelu",
        theta=4.0,
        tau_s=1.0,
        tau_m=1.0,
        use_spiking_layernorm=True,
        spiking_ln_mul=True,
        spiking_ln_log=True,
        spiking_ln_expdiff=True,
        use_spiking_mlp=True,
        spiking_mlp_exact_gelu=False,
    )
    metadata = build_vit_calibration_metadata(
        model_id="checkpoint",
        dataset_id="images",
        calibration_split="train",
        calibration_dataset_fingerprint=first._fingerprint,
        calibration_samples=4,
        calibration_seed=7,
        processor=processor,
        config=config,
        dtype="float32",
        attention_implementation="eager",
    )
    assert metadata.dataset_split == "train"
    assert metadata.input_shape == (3, 4, 4)
    assert first._fingerprint in metadata.preprocessing

    class CalibrationDataset(Dataset):
        """Return deterministic preprocessed tensors to the two-pass driver."""

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            return {
                "pixel_values": torch.tensor(
                    [float(index) - 1.5, float(index) - 1.0],
                    dtype=torch.float32,
                )
            }

    class CalibrationBlock(nn.Module):
        """Expose one analytic-or-calibrated activation boundary."""

        def forward(self, value: torch.Tensor) -> Potential:
            bounds = PotentialBounds(-4.0, 4.0)
            if model_calibration_is_bound(self):
                return calibrated_potential(
                    self,
                    "output",
                    value,
                    collection_bounds=bounds,
                )
            return Potential(value, bounds)

    class CalibrationDriverModel(nn.Module):
        """Accept evaluator-style pixel batches and execute the bound site."""

        def __init__(self) -> None:
            super().__init__()
            self.block = CalibrationBlock()

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.block(pixel_values).value

    # The driver owns binding cleanup and moves each sequential batch to the requested
    # dtype/device. Its finalized table must contain exactly the replayed population.
    driver_model = CalibrationDriverModel().eval()
    driver_spec = LayerCalibrationSpec(
        module_name="block",
        tensor_name="output",
        range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    collector = create_calibration_collector(
        metadata,
        (driver_spec,),
        bin_count=4,
    )
    loader = DataLoader(CalibrationDataset(), batch_size=2, shuffle=False)
    set_gaussian_time_noise(enabled=False)
    table = collect_vit_calibration_table(
        driver_model,
        loader,
        collector,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expected_samples=4,
    )
    assert table.layers[0].num_values == 8
    assert not model_calibration_is_bound(driver_model.block)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT Evaluator Artifact Lifecycle]]
def verify_vit_evaluator_artifact_lifecycle() -> None:
    """Verify ViT calibration CLI conversion and clean-collection restrictions."""
    from unittest.mock import patch

    from scripts.evaluation.error_analysis_vit import (
        parse_arguments,
        validate_vit_calibration_arguments,
    )

    # Parsing exposes every artifact and statistical control without aliases to the
    # legacy quantile diagnostic. The active string converts to the shared enum only
    # after backend, path, population, and range-policy validation succeeds.
    with patch(
        "sys.argv",
        [
            "error_analysis_vit.py",
            "--experiment_name",
            "calibration-smoke",
            "--model_backend",
            "spiking",
            "--calibration-mode",
            "collect",
            "--calibration-path",
            "artifacts/calibration/vit-smoke.json",
            "--calibration-samples",
            "32",
            "--calibration-seed",
            "17",
            "--calibration-bins",
            "64",
            "--calibration-lower-quantile",
            "0.01",
            "--calibration-upper-quantile",
            "0.99",
            "--calibration-margin-fraction",
            "0.1",
        ],
    ):
        args = parse_arguments()
    assert validate_vit_calibration_arguments(args) is CalibrationMode.COLLECT
    assert args.calibration_samples == 32
    assert args.calibration_seed == 17
    assert args.calibration_bins == 64

    # Collection must remain clean, whereas frozen validation may combine the table
    # with independent Gaussian timing noise. Disabled calibration needs no path.
    _expect_raises(
        ValueError,
        lambda: validate_vit_calibration_arguments(
            replace(args, gaussian_time_noise=True)
        ),
        "perturbations to be disabled",
    )
    frozen_args = replace(
        args,
        calibration_mode="validate",
        gaussian_time_noise=True,
    )
    assert (
        validate_vit_calibration_arguments(frozen_args)
        is CalibrationMode.VALIDATE
    )
    assert validate_vit_calibration_arguments(
        replace(args, calibration_mode="none", calibration_path="")
    ) is None

    # Invalid active paths and statistical controls must fail before dataset loading.
    _expect_raises(
        ValueError,
        lambda: validate_vit_calibration_arguments(
            replace(args, calibration_path="")
        ),
        "calibration_path",
    )
    _expect_raises(
        ValueError,
        lambda: validate_vit_calibration_arguments(
            replace(args, calibration_lower_quantile=0.9999)
        ),
        "ordered",
    )


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT Fixed Activation Ranges]]
def verify_vit_fixed_activation_ranges() -> None:
    """Verify direct ViT MLP activations propagate input-derived fixed ranges."""
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import (
        ViTConfig,
    )
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
        ViTIntermediate,
    )

    # Every case receives different activation values on the same declared input
    # range. Output domains must remain identical across those values, proving that
    # direct activation metadata no longer uses current-tensor extrema.
    set_gaussian_time_noise(enabled=False)
    input_domain = PotentialBounds(-2.0, 2.0)
    first_value = torch.tensor(
        [[[-1.5, -0.25, 0.5, 1.25]]],
        dtype=torch.float32,
    )
    second_value = torch.tensor(
        [[[1.5, 0.25, -0.5, -1.25]]],
        dtype=torch.float32,
    )
    cases = (
        (True, True, "gelu"),
        (False, False, "gelu"),
        (False, False, "relu"),
        (False, False, "tanh"),
    )
    for index, (use_spiking_mlp, exact_gelu, hidden_act) in enumerate(cases):
        torch.manual_seed(1800 + index)
        config = ViTConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            theta=4.0,
            use_spiking_mlp=use_spiking_mlp,
            spiking_mlp_exact_gelu=exact_gelu,
            hidden_act=hidden_act,
        )
        module = ViTIntermediate(config).eval()
        first = module(Potential(first_value, input_domain))
        second = module(Potential(second_value, input_domain))
        assert first.domain == second.domain
        assert first.domain.min <= 0.0 <= first.domain.max


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#BERT Fixed Range Flow]]
def verify_bert_fixed_range_flow() -> None:
    """Verify BERT embedding, activation, encoder, and pooler range propagation."""
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_bert.configuration_bert import BertConfig
    from utils.transformers.models.spiking_bert.modeling_spiking_bert import (
        BertEmbeddings,
        BertEncoder,
        BertIntermediate,
        BertPooler,
    )

    def make_config(*, use_spiking_mlp: bool) -> BertConfig:
        """Construct one small deterministic BERT range-flow configuration."""
        config = BertConfig(
            vocab_size=16,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=8,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            use_spiking_layernorm=False,
            use_spiking_mlp=use_spiking_mlp,
            hidden_act="gelu",
            theta=4.0,
        )
        config._attn_implementation = "eager"
        return config

    # Different token batches use the same table-derived normalized range. Public
    # embedding calls remain tensors, while the internal opt-in carries Potential.
    set_gaussian_time_noise(enabled=False)
    torch.manual_seed(1900)
    embeddings = BertEmbeddings(make_config(use_spiking_mlp=True)).eval()
    first_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    second_ids = torch.tensor([[4, 3, 2, 1]], dtype=torch.long)
    public_output = embeddings(input_ids=first_ids)
    first_output = embeddings(input_ids=first_ids, return_potential=True)
    second_output = embeddings(input_ids=second_ids, return_potential=True)
    assert isinstance(public_output, torch.Tensor)
    assert isinstance(first_output, Potential)
    assert isinstance(second_output, Potential)
    assert first_output.domain == second_output.domain

    # Frozen table intervals are immutable cache results. A standard parameter update
    # invalidates them until an explicit refresh establishes a new parameter regime.
    first_bounds = embeddings.freeze_parameter_bounds()
    assert embeddings.freeze_parameter_bounds() is first_bounds
    with torch.no_grad():
        embeddings.word_embeddings.weight.add_(0.01)
    _expect_raises(
        RuntimeError,
        embeddings.freeze_parameter_bounds,
        "changed after bounds were frozen",
    )
    refreshed_bounds = embeddings.freeze_parameter_bounds(refresh=True)
    assert refreshed_bounds != first_bounds

    # A plain custom tensor may reuse the word-table rail only while it stays inside
    # that rail. An explicit Potential supplies a separately established fixed range.
    word_bounds = refreshed_bounds[0]
    compatible_custom = torch.full(
        (1, 4, 4),
        (float(word_bounds.min) + float(word_bounds.max)) / 2.0,
    )
    compatible = embeddings(
        inputs_embeds=compatible_custom,
        return_potential=True,
    )
    assert isinstance(compatible, Potential)
    explicit = embeddings(
        inputs_embeds=Potential(
            torch.zeros_like(compatible_custom),
            PotentialBounds(-1.0, 1.0),
        ),
        return_potential=True,
    )
    assert isinstance(explicit, Potential)
    _expect_raises(
        ValueError,
        lambda: embeddings(
            inputs_embeds=torch.full_like(
                compatible_custom,
                float(word_bounds.max) + 1.0,
            ),
            return_potential=True,
        ),
        "escaped its declared fixed range",
    )

    # Encoder entry, direct activation, and first-token pooling all retain declared
    # ranges across different tensor values instead of measuring either batch.
    input_domain = PotentialBounds(-2.0, 2.0)
    first_value = torch.tensor(
        [[[-1.5, -0.5, 0.5, 1.5], [0.25, -0.25, 0.75, -0.75]]],
        dtype=torch.float32,
    )
    second_value = -first_value
    for index, use_spiking_mlp in enumerate((False, True)):
        torch.manual_seed(1910 + index)
        config = make_config(use_spiking_mlp=use_spiking_mlp)
        encoder = BertEncoder(config).eval()
        intermediate = BertIntermediate(config).eval()
        pooler = BertPooler(config).eval()
        first_potential = Potential(first_value, input_domain)
        second_potential = Potential(second_value, input_domain)
        encoded_first = encoder(first_potential)
        encoded_second = encoder(second_potential)
        assert encoded_first.domain == encoded_second.domain
        activated_first = intermediate(first_potential)
        activated_second = intermediate(second_potential)
        assert activated_first.domain == activated_second.domain
        assert pooler(encoded_first).shape == pooler(encoded_second).shape == (1, 4)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#RoBERTa Fixed Range Flow]]
def verify_roberta_fixed_range_flow() -> None:
    """Verify RoBERTa fixed ranges through dense/spiking blocks and task heads."""
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_roberta.configuration_roberta import (
        RobertaConfig,
    )
    from utils.transformers.models.spiking_roberta.modeling_spiking_roberta import (
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        RobertaModel,
    )

    def make_config(*, use_spiking_mlp: bool) -> RobertaConfig:
        """Construct a small evaluation-only RoBERTa configuration."""
        config = RobertaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            type_vocab_size=1,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            use_spiking_layernorm=False,
            use_spiking_mlp=use_spiking_mlp,
            hidden_act="gelu",
            theta=8.0,
            num_labels=2,
            pad_token_id=1,
        )
        config._attn_implementation = "eager"
        return config

    # Two token populations with the same shapes must retain identical embedding and
    # final encoder domains for both dense and spiking MLP implementations. Public
    # model calls retain Hugging Face outputs; only local wrappers request Potential.
    set_gaussian_time_noise(enabled=False)
    first_ids = torch.tensor([[0, 4, 5, 2], [0, 7, 8, 2]])
    second_ids = torch.tensor([[0, 9, 10, 2], [0, 11, 12, 2]])
    attention_mask = torch.ones_like(first_ids)
    for index, use_spiking_mlp in enumerate((False, True)):
        torch.manual_seed(2000 + index)
        config = make_config(use_spiking_mlp=use_spiking_mlp)
        model = RobertaModel(config).eval()
        public_embeddings = model.embeddings(input_ids=first_ids)
        first_embeddings = model.embeddings(
            input_ids=first_ids,
            return_potential=True,
        )
        second_embeddings = model.embeddings(
            input_ids=second_ids,
            return_potential=True,
        )
        assert isinstance(public_embeddings, torch.Tensor)
        assert isinstance(first_embeddings, Potential)
        assert isinstance(second_embeddings, Potential)
        assert first_embeddings.domain == second_embeddings.domain

        public_output = model(
            input_ids=first_ids,
            attention_mask=attention_mask,
        )
        internal_output, first_potential = model(
            input_ids=first_ids,
            attention_mask=attention_mask,
            return_potential=True,
        )
        _, second_potential = model(
            input_ids=second_ids,
            attention_mask=attention_mask,
            return_potential=True,
        )
        assert public_output.last_hidden_state.shape == (2, 4, 8)
        assert internal_output.last_hidden_state.shape == (2, 4, 8)
        assert first_potential.domain == second_potential.domain

        # Local wrappers consume the private Potential path but retain their standard
        # task output classes and shapes at the external API boundary.
        classifier = RobertaForSequenceClassification(config).eval()
        classified = classifier(
            input_ids=first_ids,
            attention_mask=attention_mask,
        )
        assert classified.logits.shape == (2, 2)
        masked_lm = RobertaForMaskedLM(config).eval()
        predicted = masked_lm(
            input_ids=first_ids,
            attention_mask=attention_mask,
        )
        assert predicted.logits.shape == (2, 4, 32)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#GPT-2 Evaluator Artifact Lifecycle]]
def verify_gpt2_evaluator_artifact_lifecycle() -> None:
    """Verify GPT-2 CLI controls, metadata identity, and two-pass token replay."""
    from unittest.mock import patch

    from torch.utils.data import DataLoader, Dataset

    from scripts.evaluation.error_analysis_gpt2 import (
        parse_arguments,
        validate_gpt2_calibration_arguments,
    )
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import GPT2Model

    # The CLI exposes collection identity and histogram controls independently from
    # the older diagnostic quantile hook. Validation runs before datasets or model
    # checkpoints are loaded and restricts collection to a clean spiking backend.
    with patch(
        "sys.argv",
        [
            "error_analysis_gpt2.py",
            "--model_backend",
            "spiking",
            "--calibration-mode",
            "collect",
            "--calibration-path",
            "artifacts/calibration/gpt2-smoke.json",
            "--calibration-samples",
            "4",
            "--calibration-seed",
            "19",
            "--calibration-bins",
            "16",
            "--calibration-lower-quantile",
            "0.01",
            "--calibration-upper-quantile",
            "0.99",
            "--calibration-margin-fraction",
            "0.1",
        ],
    ):
        args = parse_arguments()
    assert validate_gpt2_calibration_arguments(args) is CalibrationMode.COLLECT
    assert args.calibration_samples == 4
    assert args.calibration_seed == 19
    assert args.calibration_bins == 16
    _expect_raises(
        ValueError,
        lambda: validate_gpt2_calibration_arguments(
            replace(args, gaussian_time_noise=True)
        ),
        "timing noise off",
    )
    assert validate_gpt2_calibration_arguments(
        replace(
            args,
            calibration_mode="validate",
            gaussian_time_noise=True,
        )
    ) is CalibrationMode.VALIDATE

    # Metadata includes the selected text fingerprint, tokenizer ID mapping controls,
    # padded length, model path, and TTFS configuration. Robustness noise is omitted
    # so the clean table can be reused after exact compatibility validation.
    config = GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        use_spiking_mlp=True,
        use_spiking_layernorm=False,
        activation_function="gelu_new",
        theta=8.0,
        tau_s=1.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    tokenizer = SimpleNamespace(
        name_or_path="tiny-tokenizer",
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        padding_side="right",
        truncation_side="right",
    )
    metadata = build_gpt2_calibration_metadata(
        model_id="tiny-gpt2",
        dataset_id="wikitext:wikitext-2-raw-v1",
        calibration_split="train",
        calibration_dataset_fingerprint="filtered-selected-v1",
        calibration_samples=4,
        calibration_seed=19,
        tokenizer=tokenizer,
        config=config,
        max_length=4,
        attention_implementation="eager",
    )
    assert metadata.model_family == "gpt2"
    assert metadata.max_sequence_length == 4
    assert "filtered-selected-v1" in metadata.preprocessing
    assert metadata.input_shape == (4,)

    class TokenDataset(Dataset):
        """Return fixed padded token batches through default dictionary collation."""

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            tokens = torch.tensor(
                [1, 3 + index, 4 + index, 2],
                dtype=torch.long,
            )
            return {
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "labels": tokens.clone(),
            }

    # The production driver binds one collector across two sequential passes, ignores
    # labels, disables cache, finalizes all entry/residual records, and unbinds state.
    torch.manual_seed(2120)
    model = GPT2Model(config).eval()
    specs = gpt2_calibration_specs(
        model,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    collector = create_calibration_collector(metadata, specs, bin_count=16)
    loader = DataLoader(TokenDataset(), batch_size=2, shuffle=False)
    set_gaussian_time_noise(enabled=False)
    table = collect_gpt2_calibration_table(
        model,
        loader,
        collector,
        device=torch.device("cpu"),
        expected_samples=4,
    )
    assert len(table.layers) == 3
    assert all(layer.num_values > 0 for layer in table.layers)
    assert not model_calibration_is_bound(model)
    assert not model_calibration_is_bound(model.h[0])


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#GPT-2 Fixed Range Flow]]
def verify_gpt2_fixed_range_flow() -> None:
    """Verify GPT-2 embedding, MLP, and pre-norm residual fixed-range flow."""
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
        GPT2MLP,
        GPT2Model,
    )

    def make_config(
        *,
        use_spiking_mlp: bool,
        activation_function: str = "gelu_new",
    ) -> GPT2Config:
        """Construct a small cache-free GPT-2 verification configuration."""
        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            use_spiking_mlp=use_spiking_mlp,
            use_spiking_layernorm=False,
            activation_function=activation_function,
            theta=8.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        return config

    # Every maintained activation maps the same declared input range identically
    # across different tensor values, for both dense and spiking Conv1D execution.
    set_gaussian_time_noise(enabled=False)
    input_domain = PotentialBounds(-2.0, 2.0)
    first_value = torch.tensor(
        [[[-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5]]]
    )
    second_value = -first_value
    for use_spiking_mlp in (False, True):
        for activation_name in ("gelu_new", "relu", "silu", "tanh"):
            torch.manual_seed(2100)
            config = make_config(
                use_spiking_mlp=use_spiking_mlp,
                activation_function=activation_name,
            )
            mlp = GPT2MLP(16, config).eval()
            first = mlp(Potential(first_value, input_domain))
            second = mlp(Potential(second_value, input_domain))
            assert first.domain == second.domain

    # Architecture discovery declares one root model entry and two residual sites per
    # block. Two identical passes collect a complete immutable table from token IDs.
    torch.manual_seed(2110)
    model = GPT2Model(make_config(use_spiking_mlp=True)).eval()
    specs = gpt2_calibration_specs(
        model,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    assert tuple((spec.module_name, spec.tensor_name) for spec in specs) == (
        ("", "input"),
        ("h.0", "attention_residual"),
        ("h.0", "output"),
    )
    metadata = replace(
        _make_metadata(),
        model_family="gpt2",
        model_id="tiny-gpt2",
        max_sequence_length=4,
        input_shape=(4,),
        model_options=(("use_spiking_mlp", True),),
    )
    collector = create_calibration_collector(metadata, specs, bin_count=16)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    assert bind_model_calibration(model, collector) == 2
    first_pass = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    start_histogram_calibration_pass(collector)
    second_pass = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    assert torch.equal(first_pass.last_hidden_state, second_pass.last_hidden_state)
    table = finalize_calibration_collection(collector)
    assert clear_model_calibration(model, expected_state=collector) == 2

    # Frozen validation binds exactly the same root and block modules. Every site is
    # exercised without updating its persisted range, and reports a positive element
    # denominator even when this in-population replay has no excursions.
    runtime = create_calibration_runtime(
        CalibrationMode.VALIDATE,
        table,
        expected_metadata=metadata,
    )
    assert bind_model_calibration(model, runtime) == 2
    frozen = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    assert frozen.last_hidden_state.shape == (2, 4, 8)
    report = get_calibration_clipping_report(runtime)
    assert {
        (item.module_name, item.tensor_name) for item in report
    } == {
        ("", "input"),
        ("h.0", "attention_residual"),
        ("h.0", "output"),
    }
    assert all(item.num_values > 0 for item in report)
    assert clear_model_calibration(model, expected_state=runtime) == 2


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT Residual Range Reset]]
def verify_vit_residual_range_reset() -> None:
    """Verify ViT block residuals collect raw values and consume frozen ranges."""
    # Import the model adapter only for this integration group so the common observer
    # checks remain independent of Hugging Face model registration side effects.
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import (
        ViTConfig,
    )
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
        ViTEncoder,
        ViTLayer,
    )

    class ResidualModel(nn.Module):
        """Expose one ViT block under the stable name used by calibration records."""

        def __init__(self) -> None:
            super().__init__()
            config = ViTConfig(
                hidden_size=4,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=8,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                theta=4.0,
                use_spiking_layernorm=False,
                use_spiking_mlp=True,
                spiking_mlp_exact_gelu=False,
            )
            config._attn_implementation = "eager"
            self.block = ViTLayer(config)

    torch.manual_seed(1701)
    model = ResidualModel().eval()

    # Both residual boundaries use signed symmetric calibration because their dense
    # distributions cross zero and must provide valid zero-reference affine rails.
    specs = vit_residual_calibration_specs(
        model,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    assert tuple((spec.module_name, spec.tensor_name) for spec in specs) == (
        ("block", "attention_residual"),
        ("block", "output"),
    )
    metadata = replace(
        _make_metadata(),
        input_shape=(2, 4),
        model_options=(("residual_test", True),),
    )
    collector = create_calibration_collector(metadata, specs, bin_count=8)
    set_gaussian_time_noise(enabled=False)
    calibration_input = Potential(
        torch.tensor(
            [[[-0.2, -0.1, 0.1, 0.2], [0.15, -0.05, 0.05, -0.15]]],
            dtype=torch.float32,
        ),
        PotentialBounds(-2.0, 2.0),
    )

    # The two deterministic collection passes must return the same raw block output.
    # Their analytic safety rails remain available while observers record each named
    # residual distribution without clamping it against its own measurements.
    assert bind_model_calibration(model, collector) == 1
    first_pass = model.block(calibration_input)
    start_histogram_calibration_pass(collector)
    second_pass = model.block(calibration_input)
    assert torch.equal(first_pass.value, second_pass.value)
    table = finalize_calibration_collection(collector)
    assert clear_model_calibration(model, expected_state=collector) == 1

    # Calibration-free execution retains the wider analytic interval. Installing the
    # frozen runtime replaces the second residual metadata with the persisted block
    # range and keeps the same in-range activation numerically unchanged.
    analytic = model.block(calibration_input)
    runtime = create_calibration_runtime(
        CalibrationMode.VALIDATE,
        table,
        expected_metadata=metadata,
    )
    assert bind_model_calibration(model, runtime) == 1
    frozen = model.block(calibration_input)
    frozen_record = get_layer_calibration(table, "block", "output")
    assert frozen.domain == PotentialBounds(
        frozen_record.bounds.min,
        frozen_record.bounds.max,
    )
    assert frozen.domain != analytic.domain
    assert torch.equal(frozen.value, analytic.value)

    # A broader input still carries the same declared upstream safety rail, but its
    # raw residuals exceed the narrow calibration population. Frozen execution must
    # count and clamp those excursions without widening either persisted range.
    evaluation_input = Potential(
        torch.tensor(
            [[[-1.8, -1.2, 1.2, 1.8], [1.5, -1.0, 1.0, -1.5]]],
            dtype=torch.float32,
        ),
        calibration_input.domain,
    )
    evaluated = model.block(evaluation_input)
    assert evaluated.domain == frozen.domain
    report = get_calibration_clipping_report(runtime)
    assert {item.tensor_name for item in report} == {
        "attention_residual",
        "output",
    }
    assert sum(item.underflows + item.overflows for item in report) > 0
    assert clear_model_calibration(model, expected_state=runtime) == 1

    # A complete encoder adds one entry site before the two residual sites. Its
    # calibration-free theta rail makes declared output domains independent of the
    # current embedding values instead of reconstructing them from batch extrema.
    class EncoderModel(nn.Module):
        """Expose a one-block encoder under a wrapper-stable module path."""

        def __init__(self) -> None:
            super().__init__()
            config = ViTConfig(
                hidden_size=4,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=8,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                theta=4.0,
                use_spiking_layernorm=False,
                use_spiking_mlp=True,
            )
            config._attn_implementation = "eager"
            self.encoder = ViTEncoder(config)

    torch.manual_seed(1702)
    encoder_model = EncoderModel().eval()
    encoder_specs = vit_calibration_specs(
        encoder_model,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.0,
    )
    assert tuple(
        (spec.module_name, spec.tensor_name) for spec in encoder_specs
    ) == (
        ("encoder", "input"),
        ("encoder.layer.0", "attention_residual"),
        ("encoder.layer.0", "output"),
    )
    small_embeddings = torch.tensor(
        [[[-0.2, -0.1, 0.1, 0.2], [0.1, -0.2, 0.2, -0.1]]],
        dtype=torch.float32,
    )
    broad_embeddings = small_embeddings * 8.0
    small_output = encoder_model.encoder(small_embeddings)
    broad_output = encoder_model.encoder(broad_embeddings)
    assert small_output.domain == broad_output.domain


# @lat: [[calibration#Layer-wise Calibration#Persistence#Canonical Table Round Trip]]
def verify_canonical_table_round_trip() -> None:
    """Check canonical ordering, identity validation, strict schema, and atomic I/O."""
    metadata = _make_metadata()
    later = _make_lower_bounded_layer("encoder.layer.1")
    earlier = _make_layer("encoder.layer.0", (2, 1, 2, 1))

    # Input order is deliberately reversed. Canonical construction sorts stable
    # identities and rejects duplicates before persistence or runtime lookup.
    table = create_calibration_table(metadata, [later, earlier])
    assert table.format_version == CALIBRATION_FORMAT_VERSION
    assert [layer.module_name for layer in table.layers] == [
        "encoder.layer.0",
        "encoder.layer.1",
    ]
    assert get_layer_calibration(table, "encoder.layer.1", "output") == later
    _expect_raises(
        KeyError,
        lambda: get_layer_calibration(table, "encoder.missing", "output"),
        "missing layer calibration",
    )
    _expect_raises(
        ValueError,
        lambda: create_calibration_table(metadata, [earlier, earlier]),
        "duplicate",
    )

    # Metadata compatibility is exact and reports the differing configuration field
    # rather than allowing a table collected at another threshold to be installed.
    validate_calibration_metadata(table.metadata, metadata)
    _expect_raises(
        ValueError,
        lambda: validate_calibration_metadata(
            table.metadata, replace(metadata, theta=1000.0)
        ),
        "theta",
    )

    # In-memory schema round-trip returns independent immutable data and catches both
    # unknown fields and bounds that no longer match histogram policy.
    payload = calibration_table_to_dict(table)
    assert calibration_table_from_dict(copy.deepcopy(payload)) == table
    unknown = copy.deepcopy(payload)
    unknown["unknown"] = True
    _expect_raises(
        ValueError,
        lambda: calibration_table_from_dict(unknown),
        "unknown",
    )
    tampered = copy.deepcopy(payload)
    tampered["layers"][0]["bounds"]["max"] += 1.0
    _expect_raises(
        ValueError,
        lambda: calibration_table_from_dict(tampered),
        "do not match",
    )

    # Filesystem persistence creates missing parents, publishes complete JSON, and is
    # byte-deterministic across repeated writes of the same canonical table.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "calibration.json"
        save_calibration_table(table, path)
        first_bytes = path.read_bytes()
        assert load_calibration_table(path) == table
        save_calibration_table(table, path)
        assert path.read_bytes() == first_bytes
        assert not list(path.parent.glob(f".{path.name}.*.tmp"))

        # Python's JSON parser accepts NaN by default; the strict loader explicitly
        # rejects it before any partial calibration table can be returned.
        invalid_path = Path(directory) / "nonfinite.json"
        invalid_path.write_text('{"format_version": NaN}', encoding="utf-8")
        _expect_raises(
            ValueError,
            lambda: load_calibration_table(invalid_path),
            "non-finite JSON constant",
        )


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Live Tensor Extrema Source Audit]]
def verify_no_live_tensor_extrema_bounds() -> None:
    """Reject tensor-reduction-derived bounds in maintained execution functions.

    The audit parses maintained transform and Transformer Python sources and detects
    both direct construction such as ``PotentialBounds(x.min(), x.max())`` and simple
    local data flow where a reduction is assigned before entering a bound constructor.
    Built-in scalar ``min`` and ``max`` remain valid interval arithmetic.

    Learned-parameter and embedding-table reductions are allowed only inside the
    explicitly named freeze methods that establish immutable cache entries. Module
    demonstration code is outside every function and therefore outside production
    execution scope.
    """
    bound_constructor_names = {"OpenBounds", "PotentialBounds", "TimeBounds"}
    tensor_reduction_names = {"min", "max", "amin", "amax", "nanmin", "nanmax"}
    allowed_freeze_functions = {
        "freeze_parameter_bounds",
        "freeze_embedding_bounds",
        "freeze_dense_layer_norm_bounds",
    }

    def call_name(node: ast.Call) -> str | None:
        """Return the terminal callable name without resolving imported aliases."""
        # Bounds are imported by their canonical class names throughout maintained
        # code. Supporting a qualified attribute also catches a future module alias.
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def contains_tensor_reduction(node: ast.AST) -> bool:
        """Return whether an expression contains a tensor-style extrema call."""
        # Method reductions and torch/numpy qualified reductions both appear as an
        # Attribute call. A plain Name call such as built-in min/max is deliberately
        # excluded because it combines already fixed scalar interval endpoints.
        return any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr in tensor_reduction_names
            for candidate in ast.walk(node)
        )

    def assigned_names(target: ast.AST) -> set[str]:
        """Collect local names written by one ordinary or unpacking assignment."""
        # Tuple/list unpacking is included so splitting a reduced tensor cannot evade
        # the simple local data-flow check. Attribute writes are intentionally not
        # treated as local immutable scalar setup.
        return {
            candidate.id
            for candidate in ast.walk(target)
            if isinstance(candidate, ast.Name)
        }

    def expression_uses_names(node: ast.AST, names: set[str]) -> bool:
        """Return whether an expression reads any locally tainted scalar name."""
        return any(
            isinstance(candidate, ast.Name) and candidate.id in names
            for candidate in ast.walk(node)
        )

    def function_violations(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[int, ...]:
        """Find bound constructors reached by direct or local extrema reductions."""
        if function.name in allowed_freeze_functions:
            return ()

        # Build a conservative local taint set. Repeating to a fixed point handles
        # aliases such as ``lo = x.min(); bound_lo = float(lo)`` without attempting
        # whole-program type inference or following calls across function boundaries.
        assignments: list[tuple[set[str], ast.AST]] = []
        for candidate in ast.walk(function):
            if isinstance(candidate, ast.Assign):
                targets = set().union(
                    *(assigned_names(target) for target in candidate.targets)
                )
                assignments.append((targets, candidate.value))
            elif isinstance(candidate, ast.AnnAssign):
                if candidate.value is not None:
                    assignments.append(
                        (assigned_names(candidate.target), candidate.value)
                    )
        tainted_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for targets, value in assignments:
                if contains_tensor_reduction(value) or expression_uses_names(
                    value,
                    tainted_names,
                ):
                    new_names = targets - tainted_names
                    if new_names:
                        tainted_names.update(new_names)
                        changed = True

        # A violation is tied to the constructor line for a precise repair location.
        # Keyword arguments are included even though current bounds use positional
        # endpoints, preventing a future named-endpoint form from escaping the audit.
        violation_lines: list[int] = []
        for candidate in ast.walk(function):
            if not isinstance(candidate, ast.Call):
                continue
            if call_name(candidate) not in bound_constructor_names:
                continue
            expressions = (*candidate.args, *(item.value for item in candidate.keywords))
            if any(
                contains_tensor_reduction(expression)
                or expression_uses_names(expression, tainted_names)
                for expression in expressions
            ):
                violation_lines.append(candidate.lineno)
        return tuple(sorted(set(violation_lines)))

    def audit_source(source: str, filename: str) -> tuple[str, ...]:
        """Parse one source string and return stable function/line diagnostics."""
        tree = ast.parse(source, filename=filename)
        diagnostics: list[str] = []
        for candidate in ast.walk(tree):
            if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for line in function_violations(candidate):
                diagnostics.append(f"{filename}:{line}:{candidate.name}")
        return tuple(sorted(set(diagnostics)))

    # Verify the verifier first: direct and aliased tensor reductions must fail, while
    # built-in min/max over fixed endpoints must remain accepted interval arithmetic.
    assert len(
        audit_source(
            "def forward(x):\n"
            "    return PotentialBounds(x.min().item(), x.max().item())\n",
            "direct.py",
        )
    ) == 1
    assert len(
        audit_source(
            "def helper(x):\n"
            "    lo = x.amin().item()\n"
            "    alias = float(lo)\n"
            "    return TimeBounds(alias, 1.0)\n",
            "indirect.py",
        )
    ) == 1
    assert audit_source(
        "def helper(domain_a, domain_b):\n"
        "    return PotentialBounds(min(domain_a.min, domain_b.min), "
        "max(domain_a.max, domain_b.max))\n",
        "interval.py",
    ) == ()

    # Ordinary LayerNorm now follows the same immutable parameter-cache contract as
    # the spiking and affine adapters. Different activation batches reuse one domain;
    # parameter mutation fails until setup explicitly refreshes the cached envelope.
    from utils.transformers.models.spiking_ops import (
        _apply_norm,
        freeze_dense_layer_norm_bounds,
    )

    dense_norm = nn.LayerNorm(4).eval()
    input_domain = PotentialBounds(-2.0, 2.0)
    first_output = _apply_norm(
        dense_norm,
        Potential(torch.tensor([[-1.0, -0.5, 0.5, 1.0]]), input_domain),
    )
    second_output = _apply_norm(
        dense_norm,
        Potential(torch.tensor([[1.5, 0.25, -0.25, -1.5]]), input_domain),
    )
    assert first_output.domain is second_output.domain
    with torch.no_grad():
        dense_norm.weight.mul_(2.0)
    _expect_raises(
        RuntimeError,
        lambda: _apply_norm(
            dense_norm,
            Potential(torch.zeros(1, 4), input_domain),
        ),
        "refresh=True",
    )
    refreshed_domain = freeze_dense_layer_norm_bounds(dense_norm, refresh=True)
    assert refreshed_domain != first_output.domain

    # Scan only maintained operator and model-adapter sources. Calibration observers
    # are included; their extrema may update statistics but cannot flow into a bound
    # constructor in the same collection invocation.
    source_roots = (
        REPOSITORY_ROOT / "utils" / "transforms",
        REPOSITORY_ROOT / "utils" / "transformers",
    )
    violations: list[str] = []
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.extend(
                audit_source(path.read_text(encoding="utf-8"), relative_path)
            )
    assert not violations, (
        "live tensor extrema reached bound constructors outside explicit parameter "
        f"freeze methods: {violations!r}"
    )


def main() -> None:
    """Run every permanent calibration contract check."""
    checks = (
        verify_observer_and_histogram_invariants,
        verify_quantile_and_margin_policy,
        verify_layer_record_and_clipping,
        verify_collection_and_runtime_phase_separation,
        verify_model_binding_and_potential_boundary,
        verify_preprocessing_derived_image_range,
        verify_deterministic_training_subset,
        verify_vit_evaluator_artifact_lifecycle,
        verify_vit_fixed_activation_ranges,
        verify_bert_fixed_range_flow,
        verify_roberta_fixed_range_flow,
        verify_gpt2_evaluator_artifact_lifecycle,
        verify_gpt2_fixed_range_flow,
        verify_vit_residual_range_reset,
        verify_canonical_table_round_trip,
        verify_no_live_tensor_extrema_bounds,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"Calibration verification passed ({len(checks)} groups).")


if __name__ == "__main__":
    main()
