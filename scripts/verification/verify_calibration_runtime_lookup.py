"""Verify frozen calibration lookup avoids repeated table validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import utils.transforms.calibration as core
from utils.transformers.calibration import (
    bind_model_calibration, calibrated_potential, model_calibration_is_bound,
)
from utils.transforms.types import PotentialBounds


def _table(site_count=1, bins=8):
    metadata = core.CalibrationMetadata(
        model_family="bert", model_id="fixture", dataset_id="fixture",
        dataset_split="train", preprocessing="none", dtype="float64",
        theta=40.0, tau_s=1.0, tau_m=1.0, clip_margin=1e-5,
        max_sequence_length=4, input_shape=None, model_options=(),
    )
    histogram = core.CalibrationHistogram(
        bounds=core.CalibrationRange(-2.0, 2.0), bin_counts=(1,) * bins,
        num_values=bins, underflows=0, overflows=0,
    )
    names = ("",) if site_count == 1 else tuple(f"site{i}" for i in range(site_count))
    layers = tuple(core.create_layer_calibration(
        core.LayerCalibrationSpec(
            name, "output", core.CalibrationRangePolicy.SIGNED_SYMMETRIC,
            0.0, 1.0, 0.05,
        ),
        core.MinMaxObserverState(-2.0, 2.0, bins),
        histogram,
    ) for name in names)
    return core.create_calibration_table(metadata, layers)


def _runtime(table):
    return core.create_calibration_runtime(
        core.CalibrationMode.INFERENCE, table, expected_metadata=table.metadata,
    )


def _reject(operation, fragment, error_type=ValueError):
    try:
        operation()
    except error_type as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {fragment}")


def _verify_table(table):
    """Test every record, complete accounting, and strict public lookup."""
    serialized = core.calibration_table_to_dict(table)
    expected_counts = {}
    with patch.object(core, "create_calibration_table",
                      wraps=core.create_calibration_table) as validate:
        state = _runtime(table)
        assert validate.call_count == 1
        started = perf_counter()
        for layer in table.layers:
            key = (layer.module_name, layer.tensor_name)
            lower, upper = layer.bounds.min, layer.bounds.max
            span = upper - lower
            values = torch.tensor(
                [lower - span, lower, 0.0, upper, upper + span],
                dtype=getattr(torch, table.metadata.dtype),
            )
            counts = core.CalibrationClippingCounts()
            expected = core.clamp_with_calibration(values, layer.bounds, counts)
            for batch in (values[:2], values[2:]):
                assert core.get_runtime_layer_calibration(state, *key) is layer
                result = core.apply_calibrated_activation(state, *key, batch)
                reference = expected[:2] if batch.numel() == 2 else expected[2:]
                assert torch.equal(result, reference)
            expected_counts[key] = counts
        elapsed = perf_counter() - started
        assert state.clipping_counts == expected_counts
        assert validate.call_count == 1, "forward reconstructed the calibration table"
        first = table.layers[0]
        assert core.get_layer_calibration(
            table, first.module_name, first.tensor_name,
        ) is first
        assert validate.call_count == 2, "public setup lookup stopped validating"
    assert core.calibration_table_to_dict(table) == serialized
    assert isinstance(state._record_index.records, MappingProxyType)
    _reject(
        lambda: state._record_index.records.__setitem__(("missing", "output"), first),
        "__setitem__", AttributeError,
    )
    return elapsed


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Runtime Record Lookup]]
def verify_runtime_lookup():
    """Verify exact values, bounds, counts, setup validation, and invalid states."""
    table = _table()
    key = ("", "output")
    state = _runtime(table)
    module = nn.Module()
    bind_model_calibration(module, state)
    values = torch.tensor([-3.0, -2.2, 0.0, 2.2, 3.0], dtype=torch.float64)
    counts = core.CalibrationClippingCounts()
    expected = core.clamp_with_calibration(values, table.layers[0].bounds, counts)
    with patch.object(core, "create_calibration_table",
                      side_effect=AssertionError("forward rebuilt table")):
        potential = calibrated_potential(module, "output", values)
    assert torch.equal(potential.value, expected)
    assert potential.domain == PotentialBounds(-2.2, 2.2)
    assert state.clipping_counts[key] == counts

    before = replace(state.clipping_counts[key])
    for bad in (float("nan"), float("inf"), -float("inf")):
        _reject(lambda: calibrated_potential(
            module, "output", torch.tensor([bad], dtype=torch.float64),
        ), "finite")
        assert state.clipping_counts[key] == before
    _reject(lambda: core.get_runtime_layer_calibration(
        state, "missing", "output",
    ), "missing layer", KeyError)
    _reject(lambda: core.get_runtime_layer_calibration(
        state, "", " output",
    ), "whitespace")
    state.clipping_counts.pop(key)
    _reject(lambda: core.apply_calibrated_activation(
        state, *key, values,
    ), "no clipping counter")
    state.clipping_counts[key] = before

    state.table = replace(table)
    _reject(lambda: calibrated_potential(module, "output", values), "table was replaced")
    assert state.clipping_counts[key] == before

    direct = core.CalibrationRuntimeState(
        core.CalibrationMode.INFERENCE, table, {key: core.CalibrationClippingCounts()},
    )
    _reject(lambda: core.apply_calibrated_activation(direct, *key, values),
            "created by create_calibration_runtime")
    unbound = nn.Module()
    _reject(lambda: bind_model_calibration(unbound, direct),
            "created by create_calibration_runtime")
    assert not model_calibration_is_bound(unbound)

    malformed = replace(table, layers=(replace(
        table.layers[0], bounds=core.CalibrationRange(-99, 99),
    ),))
    _reject(lambda: _runtime(malformed), "bounds do not match")
    _reject(lambda: core.get_layer_calibration(malformed, *key), "bounds do not match")

    for damaged, fragment in (
        (None, "created by"),
        (replace(_runtime(table)._record_index, records={}), "must be immutable"),
        (replace(_runtime(table)._record_index, records=MappingProxyType({})),
         "differs from its table"),
        (replace(_runtime(table)._record_index, records=MappingProxyType({
            key: replace(table.layers[0], tensor_name="different"),
        })), "identity differs"),
    ):
        damaged_state = _runtime(table)
        damaged_state._record_index = damaged
        _reject(lambda: core.apply_calibrated_activation(
            damaged_state, *key, values,
        ), fragment)
        assert damaged_state.clipping_counts[key] == core.CalibrationClippingCounts()

    state = _runtime(table)
    state.mode = core.CalibrationMode.COLLECT
    _reject(lambda: core.apply_calibrated_activation(state, *key, values),
            "validation or inference")
    _verify_table(table)
    _verify_table(_table(site_count=110, bins=2048))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, help="Optional real persisted table")
    args = parser.parse_args()
    verify_runtime_lookup()
    if args.table:
        table = core.load_calibration_table(args.table)
        elapsed = _verify_table(table)
        print(f"Verified {len(table.layers)} persisted sites in {elapsed:.6f} seconds")
    print("Calibration runtime lookup verification passed")


if __name__ == "__main__":
    main()
