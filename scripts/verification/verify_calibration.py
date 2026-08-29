#!/usr/bin/env python3
"""Verify layer-wise calibration collection, freezing, and persistence contracts."""

from __future__ import annotations

import copy
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
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
from utils.transforms.types import PotentialBounds  # noqa: E402
from utils.transformers.calibration import (  # noqa: E402
    bind_model_calibration,
    calibrated_potential,
    clear_model_calibration,
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

    # Binding resolves every declared name before mutation and rejects replicated or
    # already-bound models rather than sharing mutable counters ambiguously.
    assert bind_model_calibration(model, collector) == 1
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


def main() -> None:
    """Run every permanent calibration contract check."""
    checks = (
        verify_observer_and_histogram_invariants,
        verify_quantile_and_margin_policy,
        verify_layer_record_and_clipping,
        verify_collection_and_runtime_phase_separation,
        verify_model_binding_and_potential_boundary,
        verify_canonical_table_round_trip,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"Calibration verification passed ({len(checks)} groups).")


if __name__ == "__main__":
    main()
