#!/usr/bin/env python3
"""Pure torch checks for physical neuron weight calibration."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.backend import calibration_sha256
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.neuron_weights import (
    condition_key, delivery_statistics, load_neuron_weights,
    passing_neurons, select_neuron_weights, weight_context,
)
from utils.hardware.brainscales2.toy_pooling import (
    ToyPoolConfig, _configure_grouped_synapse_weights, resolve_grouped_physical_coordinates,
)
from scripts.evaluation.brainscales2_neuron_weights import make_inputs


def rejects(action):
    try:
        action()
    except (ValueError, KeyError):
        return
    raise AssertionError("invalid calibration was accepted")


# @lat: [[hardware#Toy ANN2SNN Verification#Neuron synaptic weight calibration]]
def verify_neuron_weights():
    pool = ToyPoolConfig(pool_size=2, logical_neurons=2)
    weights = torch.ones(4, 6)
    _configure_grouped_synapse_weights(weights, pool, input_fan_in=3, synaptic_weight=63,
                                       neuron_synaptic_weights=torch.tensor([0, 3, 17, 63]))
    torch.testing.assert_close(weights, torch.tensor([
        [0, 0, 0, 0, 0, 0], [3, 3, 3, 0, 0, 0],
        [0, 0, 0, 17, 17, 17], [0, 0, 0, 63, 63, 63],
    ], dtype=torch.float32))
    for values in ([0, 1, 2, 64], [0, 1, 2, 3.5], [0, 1, 2, float("nan")], [1, 2]):
        rejects(lambda: _configure_grouped_synapse_weights(weights, pool, input_fan_in=3,
                synaptic_weight=63, neuron_synaptic_weights=torch.tensor(values)))

    nominal = torch.tensor([[float("nan"), float("nan")], [10e-6, 20e-6], [20e-6, 10e-6]])
    perfect = nominal + 2e-6
    count = torch.isfinite(perfect).int()
    good = delivery_statistics(perfect, count, nominal, deadline_s=60e-6)
    assert bool(passing_neurons(good).all())
    bad_first = perfect.clone()
    bad_first[1, 0] = float("nan")
    bad_count = count.clone()
    bad_count[1, 0] = 0
    bad_count[2, 1] = 2
    bad = delivery_statistics(bad_first, bad_count, nominal, deadline_s=60e-6)
    selected, valid = select_neuron_weights([40, 44, 48], [bad, good, good])
    assert selected.tolist() == [44, 44] and bool(valid.all())
    _, valid = select_neuron_weights([40], [bad])
    assert not bool(valid.any())
    quiet_count = count.clone()
    quiet_count[0] = 1
    assert not bool(passing_neurons(delivery_statistics(perfect, quiet_count, nominal, deadline_s=60e-6)).any())

    spiking = BrainScaleS2PoolConfig(input_fan_in=3)
    inputs, nominal = make_inputs(pool, spiking, seed=42, trials=2, mode="mixed")
    inputs2, nominal2 = make_inputs(pool, spiking, seed=42, trials=2, mode="mixed")
    assert torch.equal(inputs, inputs2)
    torch.testing.assert_close(nominal, nominal2, equal_nan=True)
    other, _ = make_inputs(pool, spiking, seed=43, trials=2, mode="mixed")
    assert not torch.equal(inputs, other)
    assert inputs[:, :2].sum() == 0 and inputs[:, -2:].sum() == 0
    assert inputs.sum() == 32 * 2 * 2 * 3

    with tempfile.TemporaryDirectory() as temporary:
        analog = Path(temporary) / "analog.pbin"
        analog.write_bytes(b"test calibration")
        path = Path(temporary) / "weights.json"
        spiking = replace(spiking, calibration_path=analog, neuron_weight_calibration_path=str(path))
        coordinates = resolve_grouped_physical_coordinates(2, 2, "local-pool", "dedicated")
        payload = {
            "schema_version": 1, "context": weight_context(spiking), "chip_identifier": ["test-chip"],
            "validation_deadline_s": 60e-6,
            "conditions": {condition_key(pool): {"viable": True, "physical_coordinates": coordinates.tolist(),
                                                 "neuron_synaptic_weights": [40, 42, 44, 46]}},
        }
        path.write_text(json.dumps(payload))
        spiking = replace(spiking, neuron_weight_calibration_sha256=calibration_sha256(path))
        result = load_neuron_weights(spiking, pool, coordinates, ["test-chip"])
        assert result.tolist() == [40, 42, 44, 46]
        rejects(lambda: load_neuron_weights(spiking, pool, coordinates, ["other-chip"]))
        rejects(lambda: load_neuron_weights(replace(spiking, input_fan_in=4), pool, coordinates, ["test-chip"]))
        rejects(lambda: load_neuron_weights(spiking, pool, coordinates.flip(0), ["test-chip"]))
        payload["conditions"][condition_key(pool)]["viable"] = False
        path.write_text(json.dumps(payload))
        rejects(lambda: load_neuron_weights(spiking, pool, coordinates, ["test-chip"]))
        spiking = replace(spiking, neuron_weight_calibration_sha256=calibration_sha256(path))
        rejects(lambda: load_neuron_weights(spiking, pool, coordinates, ["test-chip"]))


if __name__ == "__main__":
    verify_neuron_weights()
    print("Neuron weight calibration checks passed")
