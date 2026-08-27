#!/usr/bin/env python3
"""Pure-Python regression checks for the BrainScaleS-2 pooling adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hardware.brainscales2.analysis import (
    analyze_cadc_diagnostic,
    bootstrap_variance_floor,
    fit_variance_floor,
    summarize_pool_result,
)
from utils.hardware.brainscales2.artifacts import (
    write_cadc_diagnostic_artifacts,
    write_experiment_artifacts,
)
from utils.hardware.brainscales2.backend import (
    MockPoolBackend,
    _find_raw_spikes,
    _legacy_experiment_observables,
    _raw_events_to_tensors,
    resolve_physical_neuron_indices,
)
from utils.hardware.brainscales2.config import (
    BrainScaleS2PoolConfig,
    CADCDiagnosticResult,
)
from utils.hardware.brainscales2.encoding import encode_potential_for_brainscales2
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds


def verify_identity_encoding() -> None:
    config = BrainScaleS2PoolConfig(trials=8, pool_sizes=(1, 4))
    potential = Potential(
        torch.tensor([-400.0, 0.0, 400.0]),
        PotentialBounds(-400.0, 400.0),
    )
    encoded = encode_potential_for_brainscales2(
        potential,
        config,
        pool_size=4,
        routing="broadcast",
    )
    torch.testing.assert_close(
        encoded.injected_time_s,
        torch.tensor([25.0e-6, 15.0e-6, 5.0e-6], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert encoded.dense_spikes.shape == (config.runtime_steps, 3, 1)
    assert int(encoded.dense_spikes.sum()) == 3
    assert encoded.original_shape == (3,)
    assert not bool(encoded.clamp_mask.any())

    independent = encode_potential_for_brainscales2(
        potential,
        config,
        pool_size=4,
        routing="independent",
    )
    assert independent.dense_spikes.shape == (config.runtime_steps, 3, 4)
    assert int(independent.dense_spikes.sum()) == 12


def verify_log_encoding_and_validation() -> None:
    config = BrainScaleS2PoolConfig(
        encoding="log",
        project_tau_s=2.0,
        trials=4,
        pool_sizes=(1,),
    )
    potential = Potential(
        torch.tensor([1.0, 10.0, 100.0]),
        PotentialBounds(1.0, 100.0),
    )
    encoded = encode_potential_for_brainscales2(
        potential,
        config,
        pool_size=1,
        routing="broadcast",
    )
    assert torch.all(encoded.injected_time_s[:-1] >= encoded.injected_time_s[1:])
    assert abs(float(encoded.injected_time_s[0]) - config.input_late_s) < 1.0e-12
    assert abs(float(encoded.injected_time_s[-1]) - config.input_early_s) < 1.0e-12

    invalid = Potential(torch.ones(2), PotentialBounds(0.0, 1.0))
    try:
        encode_potential_for_brainscales2(
            invalid,
            config,
            pool_size=1,
            routing="broadcast",
        )
    except ValueError as error:
        assert "strictly positive" in str(error)
    else:
        raise AssertionError("non-positive logarithmic domain was accepted")

    try:
        BrainScaleS2PoolConfig(input_late_s=70.0e-6)
    except ValueError as error:
        assert "deadline" in str(error)
    else:
        raise AssertionError("input after deadline was accepted")


def verify_software_noise_guard() -> None:
    potential = Potential(torch.tensor([0.0]), PotentialBounds(-1.0, 1.0))
    config = BrainScaleS2PoolConfig(trials=2, pool_sizes=(1,))
    set_gaussian_time_noise(enabled=True, time_std=1.0e-6, seed=4)
    try:
        try:
            encode_potential_for_brainscales2(
                potential,
                config,
                pool_size=1,
                routing="broadcast",
            )
        except RuntimeError as error:
            assert "must be disabled" in str(error)
        else:
            raise AssertionError("software timing noise was accepted")
    finally:
        set_gaussian_time_noise(enabled=False)


def verify_placement_and_raw_events() -> None:
    assert resolve_physical_neuron_indices(4, "same-quadrant") == (0, 1, 2, 3)
    assert resolve_physical_neuron_indices(4, "cross-quadrant") == (0, 128, 256, 384)
    addresses = torch.tensor([[0, 0], [0, 0], [1, 1]], dtype=torch.int64)
    times = torch.tensor([8.0e-6, 7.0e-6, 9.0e-6], dtype=torch.float64)
    first, fired, count = _raw_events_to_tensors(
        (addresses, times),
        batch_count=2,
        pool_size=2,
        raw_time_scale_s=None,
        deadline_s=60.0e-6,
    )
    assert abs(float(first[0, 0]) - 7.0e-6) < 1.0e-12
    assert bool(fired[0, 0]) and bool(fired[1, 1])
    assert int(count[0, 0]) == 2
    assert torch.isnan(first[0, 1])

    class LegacySpikeHandle:
        def get_data(self):
            return [(17, 0, 1)]

    class LegacyObservables:
        spikes = LegacySpikeHandle()

    class LegacyExtractor:
        def get(self, descriptor):
            assert descriptor == "population-7"
            return LegacyObservables()

    class LegacyExperiment:
        _hw_data_extractor = LegacyExtractor()

    class LegacyLIF:
        descriptor = "population-7"

    legacy = _legacy_experiment_observables(LegacyExperiment(), LegacyLIF())
    raw, raw_api = _find_raw_spikes(legacy)
    assert raw == [(17, 0, 1)]
    assert raw_api.endswith("LegacySpikeHandle.get_data")
    first, fired, count = _raw_events_to_tensors(
        raw,
        batch_count=1,
        pool_size=2,
        raw_time_scale_s=1.0e-9,
        deadline_s=60.0e-6,
    )
    assert float(first[0, 1]) == 17.0e-9
    assert bool(fired[0, 1]) and int(count[0, 1]) == 1


def verify_mock_analysis_and_artifacts() -> None:
    config = BrainScaleS2PoolConfig(
        pool_sizes=(1, 2, 4, 8, 16),
        placements=("same-quadrant",),
        routings=("broadcast",),
        trials=512,
        seed=17,
    )
    potential = Potential(
        torch.linspace(-400.0, 400.0, 11),
        PotentialBounds(-400.0, 400.0),
    )
    backend = MockPoolBackend()
    results = [
        backend.run(
            potential,
            config,
            pool_size=pool_size,
            placement="same-quadrant",
            routing="broadcast",
        )
        for pool_size in config.pool_sizes
    ]
    repeated = backend.run(
        potential,
        config,
        pool_size=4,
        placement="same-quadrant",
        routing="broadcast",
    )
    torch.testing.assert_close(results[2].first_spike_s, repeated.first_spike_s, equal_nan=True)

    summaries = [
        summarize_pool_result(result, "corrected-mean") for result in results
    ]
    fit = fit_variance_floor(summaries)
    assert fit["a_s2"] > 0.0
    assert fit["c_s2"] > -1.0e-13
    ci = bootstrap_variance_floor(
        results,
        "corrected-mean",
        iterations=40,
        seed=11,
    )
    assert ci["a_ci_low_s2"] <= ci["a_ci_high_s2"]
    fit_row = {
        "placement": "same-quadrant",
        "routing": "broadcast",
        "estimator": "corrected-mean",
        **fit,
        **ci,
    }
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_experiment_artifacts(
            output,
            config=config,
            potential=potential,
            results=results,
            summaries=summaries,
            fits=[fit_row],
            extra_manifest={"verification": True},
        )
        for name in (
            "manifest.json",
            "events.csv",
            "events.pt",
            "summary.csv",
            "variance_fit.csv",
        ):
            assert (output / name).is_file(), name
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert len(manifest["conditions"]) == len(config.pool_sizes)


def verify_cadc_diagnostic_and_artifacts() -> None:
    config = BrainScaleS2PoolConfig(
        trials=8,
        pool_sizes=(1, 4),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    time_s = torch.arange(config.runtime_steps, dtype=torch.float64) * config.dt_s
    baseline = torch.full(
        (config.trials, config.runtime_steps, 4),
        80.0,
        dtype=torch.float64,
    )
    baseline += 0.25 * torch.sin(
        torch.arange(config.runtime_steps, dtype=torch.float64)
    ).reshape(1, -1, 1)
    stimulated = baseline.clone()
    post = time_s >= config.input_early_s
    response = 18.0 * torch.exp(
        -(time_s[post] - config.input_early_s) / 5.0e-6
    )
    stimulated[:, post] += response.reshape(1, -1, 1)
    zeros = torch.zeros_like(baseline)
    result = CADCDiagnosticResult(
        baseline_cadc=baseline,
        stimulated_cadc=stimulated,
        baseline_spikes=zeros,
        stimulated_spikes=zeros,
        time_s=time_s,
        stimulus_time_s=config.input_early_s,
        physical_coordinates=(0, 1, 2, 3),
        metadata={"backend": "synthetic"},
    )
    analysis = analyze_cadc_diagnostic(result, config)
    assert analysis["viable"] is True
    assert analysis["selected"] is None

    nonseparable = CADCDiagnosticResult(
        baseline_cadc=baseline,
        stimulated_cadc=baseline.clone(),
        baseline_spikes=zeros,
        stimulated_spikes=zeros,
        time_s=time_s,
        stimulus_time_s=config.input_early_s,
        physical_coordinates=(0, 1, 2, 3),
    )
    assert analyze_cadc_diagnostic(nonseparable, config)["viable"] is False

    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_cadc_diagnostic_artifacts(
            output,
            config=config,
            result=result,
            analysis=analysis,
            extra_manifest={"verification": True},
        )
        for name in (
            "manifest.json",
            "summary.csv",
            "cadc_traces.pt",
            "recommended_operating_point.json",
        ):
            assert (output / name).is_file(), name
        recommendation = json.loads(
            (output / "recommended_operating_point.json").read_text(
                encoding="utf-8"
            )
        )
        assert recommendation["viable"] is True
        assert recommendation["selected"] is None


def verify_notebook_is_valid_json() -> None:
    notebook = Path("scripts/notebooks/ebrains_brainscales2_pooling.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["display_name"] == "EBRAINS-experimental"
    notebook_python = str(payload["metadata"]["language_info"]["version"])
    for supported_python in (notebook_python, "3.11.10"):
        assert supported_python.split(".")[:2] == ["3", "11"]
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "setup_hardware_client()" in source
    assert "jupyter-notebooks-experimental" in source
    assert 'DEMOS_ROOT = Path("/tmp/brainscales2-demos")' in source
    assert "%pip install --quiet --disable-pip-version-check jaxtyping matplotlib" in source
    assert "sys.executable" in source
    for run_flag in (
        "RUN_CADC_DIAGNOSTIC",
        "RUN_HARDWARE_SMOKE",
        "RUN_OPERATING_POINT_SWEEP",
        "RUN_FULL_EXPERIMENT",
    ):
        assert f"{run_flag} =" in source
        assert f"if {run_flag}:" in source
    assert '"--allow-environment-calibration"' in source
    assert '"--pool-sizes", 1, 2, 4, 8, 16' in source
    assert '"--phase", "diagnose-cadc"' in source
    assert "recommended_operating_point.json" in source


def main() -> None:
    verify_identity_encoding()
    verify_log_encoding_and_validation()
    verify_software_noise_guard()
    verify_placement_and_raw_events()
    verify_mock_analysis_and_artifacts()
    verify_notebook_is_valid_json()
    verify_cadc_diagnostic_and_artifacts()
    print("BrainScaleS-2 TTFS pooling verification passed")


if __name__ == "__main__":
    main()

