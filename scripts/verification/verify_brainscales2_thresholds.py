#!/usr/bin/env python3
"""Local checks for threshold calibration without importing EBRAINS packages."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.brainscales2_thresholds import code_schedule
from utils.hardware.brainscales2.backend import calibration_sha256
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.neuron_weights import delivery_statistics, weight_context
from utils.hardware.brainscales2.thresholds import (
    allocate_validated_coordinates, load_threshold_selection,
    passing_threshold_neurons, refine_result, selected_coordinates, validate_coordinates,
)
from utils.hardware.brainscales2.toy_pooling import ToyPoolConfig
from utils.hardware.brainscales2.toy_artifacts import iter_intermediate_conditions


def rejects(action):
    try:
        action()
    except (ValueError, KeyError):
        return
    raise AssertionError("invalid input accepted")


# @lat: [[hardware#Toy ANN2SNN Verification#Physical threshold selection]]
def verify_threshold_selection():
    good = torch.ones(512, dtype=torch.bool)
    maps = allocate_validated_coordinates(good)
    for placement, values in maps.items():
        full = validate_coordinates(torch.tensor(values), 30, 16)
        assert full.unique().numel() == 480
        for size in (1, 2, 4, 8, 16):
            assert torch.equal(torch.tensor(values)[:, :size], full[:, :size])
        if placement == "local-pool":
            assert bool(((full // 128) == (full[:, :1] // 128)).all())
    good[0] = False
    good[128] = False
    replacement = allocate_validated_coordinates(good)
    assert 0 not in torch.tensor(replacement["local-pool"])
    assert 128 not in torch.tensor(replacement["cross-quadrant"])
    # 509 good circuits can still lack 30 whole local pools of size 16.
    good[256] = False
    rejects(lambda: allocate_validated_coordinates(good))
    rejects(lambda: validate_coordinates(torch.zeros((30, 16), dtype=torch.long), 30, 16))
    rejects(lambda: validate_coordinates(torch.arange(480).reshape(30, 16) + 100, 30, 16))

    codes = code_schedule(3, 8, "mixed", 42, 64)
    assert torch.equal(codes, code_schedule(3, 8, "mixed", 42, 64))
    assert not torch.equal(codes, code_schedule(3, 8, "mixed", 100042, 64))
    for column in range(3):
        assert torch.bincount(codes[64:, column]).tolist() == [8] * 32
    isolated = code_schedule(3, 2, "isolated", 42, 10)
    assert bool(((isolated[10:] >= 0).sum(1) == 1).all())
    for column in range(3):
        assert torch.bincount(isolated[:, column][isolated[:, column] >= 0]).tolist() == [2] * 32
    nominal = codes.double() * 1e-6 + 5e-6
    nominal[codes < 0] = torch.nan
    first = nominal + 1e-6
    count = torch.isfinite(first).long()
    stats = delivery_statistics(first, count, nominal, deadline_s=60e-6)
    assert bool(passing_threshold_neurons(stats).all())
    first[64, 0] = torch.nan
    count[64, 0] = 0
    assert not passing_threshold_neurons(delivery_statistics(first, count, nominal, deadline_s=60e-6))[0]

    base_target = SimpleNamespace(leak=80, reset=80, threshold=125, tau_mem=20, capacitance=63)
    base = SimpleNamespace(neuron_result=SimpleNamespace(target=base_target, parameters={"threshold": 999}),
                           target=SimpleNamespace(neuron_target=base_target))
    seen = []
    def refine(connection, result, *, target):
        seen.append((connection, target.threshold))
        result.target.threshold = target.threshold
        result.parameters["threshold"] = 321
    target = SimpleNamespace(threshold=90)
    candidate = refine_result(base, target, "connection", refine)
    assert seen == [("connection", 90)] and base.neuron_result.target.threshold == 125
    assert base.neuron_result.parameters["threshold"] == 999
    assert candidate.neuron_result.parameters["threshold"] == 321
    assert candidate.target.neuron_target.tau_mem == 20
    assert candidate.target.neuron_target.capacitance == 63
    assert candidate.target.neuron_target.leak == candidate.target.neuron_target.reset == 80

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        analog = root / "selected.pbin"
        analog.write_bytes(b"fake analog calibration")
        path = root / "selection.json"
        config = BrainScaleS2PoolConfig(calibration_path=analog, threshold=90)
        payload = {"schema_version": 1, "viable": True, "chip_identifier": ["test"],
                   "context": weight_context(config), "calibration_path": str(analog), "coordinates": maps}
        path.write_text(json.dumps(payload))
        config = replace(config, threshold_selection_path=str(path), threshold_selection_sha256=calibration_sha256(path))
        pool = ToyPoolConfig(logical_neurons=30, pool_size=4)
        selected = selected_coordinates(config, pool, ["test"])
        assert torch.equal(selected, torch.tensor(maps[pool.placement])[:, :4])
        rejects(lambda: selected_coordinates(config, pool, ["wrong chip"]))
        rejects(lambda: selected_coordinates(replace(config, input_fan_in=4), pool, ["test"]))
        rejects(lambda: selected_coordinates(replace(config, synaptic_weight=62), pool, ["test"]))
        assert selected_coordinates(replace(config, observation_deadline_s=100e-6), pool, ["test"]).shape == (30, 4)
        analog.write_bytes(b"changed")
        rejects(lambda: load_threshold_selection(path))


# @lat: [[hardware#Toy ANN2SNN Verification#Bounded artifact aggregation]]
def verify_shard_reader():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        tensor = torch.arange(12).reshape(3, 4)
        torch.save({"conditions": {"test": {"first_spike_s": tensor}}}, root / "condition.pt")
        torch.save({"schema_version": 2, "conditions": {}, "condition_shards": {"test": "condition.pt"}}, root / "index.pt")
        old = list(iter_intermediate_conditions(root / "condition.pt"))
        new = list(iter_intermediate_conditions(root / "index.pt"))
        assert old[0][0] == new[0][0] and torch.equal(old[0][1]["first_spike_s"], new[0][1]["first_spike_s"])


if __name__ == "__main__":
    verify_threshold_selection()
    verify_shard_reader()
    assert "hxtorch" not in sys.modules and "calix" not in sys.modules
    print("Physical threshold selection checks passed")
