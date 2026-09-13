#!/usr/bin/env python3
"""Verify diagnostic controls without importing hardware packages."""

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.brainscales2_gain_diagnostic import summarize_cadc
from scripts.evaluation.brainscales2_thresholds import validate_refinement_gain
from utils.hardware.brainscales2.backend import _cadc_inputs, resolve_physical_neuron_indices
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig


# @lat: [[hardware#Toy ANN2SNN Verification#Gain diagnostic checks]]
def verify_diagnostic():
    config = BrainScaleS2PoolConfig(trials=32, input_early_s=15e-6)
    for fan_in in (1, 4):
        inputs = _cadc_inputs(replace(config, input_fan_in=fan_in))
        assert inputs.shape == (config.runtime_steps, 64, fan_in)
        assert inputs[:, ::2].sum() == 0
        assert inputs[:, 1::2].sum() == 32 * fan_in
        assert bool((inputs[15, 1::2] == 1).all())
        assert inputs[:15].sum() == inputs[16:].sum() == 0
    coordinates = resolve_physical_neuron_indices(16, "cross-quadrant")
    assert len(set(coordinates)) == 16 and all(0 <= c < 512 for c in coordinates)
    assert torch.bincount(torch.tensor(coordinates) // 128).tolist() == [4] * 4
    validate_refinement_gain(500, 500)
    validate_refinement_gain(torch.full((2, 512), 700), 700)
    try:
        validate_refinement_gain(500, 700)
    except ValueError:
        pass
    else:
        raise AssertionError("gain change accepted as potential refinement")
    baseline = torch.full((32, config.runtime_steps, 16), 80., dtype=torch.float64)
    stimulated = baseline.clone()
    stimulated[:, 15] += 10
    first = torch.full((64, 16), torch.nan)
    count = torch.zeros((64, 16), dtype=torch.long)
    data = {"baseline_cadc": baseline, "stimulated_cadc": stimulated,
            "time_s": torch.arange(config.runtime_steps).double() * config.dt_s,
            "stimulus_time_s": config.input_early_s, "physical_coordinates": coordinates,
            "metadata": {"first_spike_s": first.tolist(), "spike_count": count.tolist()}}
    rows = summarize_cadc(data, config.observation_deadline_s)
    assert all(row["psp_peak_median"] == 10 for row in rows)
    assert all(row["single_spike_rate"] == 0 for row in rows)
    assert all(not row["trace_contains_spikes"] for row in rows)
    first[1::2] = 20e-6
    count[1::2] = 1
    count[1, 0] = 2
    first[0, 1], count[0, 1] = 10e-6, 1
    data["metadata"] = {"first_spike_s": first.tolist(), "spike_count": count.tolist()}
    rows = summarize_cadc(data, config.observation_deadline_s)
    assert rows[0]["single_spike_rate"] == 31 / 32
    assert rows[0]["multi_spike_rate"] == 1 / 32
    assert rows[1]["quiet_fired_rate"] == 1 / 32
    assert all(row["trace_contains_spikes"] for row in rows)


if __name__ == "__main__":
    verify_diagnostic()
    assert "hxtorch" not in sys.modules and "calix" not in sys.modules
    print("Gain diagnostic checks passed")
