#!/usr/bin/env python3
"""Pure-Python verification for independent BSS-2 primitive characterization."""

from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_artifacts import (
    load_primitive_observations,
    write_primitive_noise_artifacts,
)
from utils.hardware.brainscales2.backend import calibration_sha256
from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend
from utils.hardware.brainscales2.primitive_correlation_worker import (
    _decode_correlation,
    _point_orders,
    _restore_point_order,
)
import utils.hardware.brainscales2.primitive_backend as primitive_backend_module
import utils.hardware.brainscales2.primitive_noise as primitive_noise_module
import scripts.evaluation.brainscales2_primitive_noise as primitive_runner_module
from utils.hardware.brainscales2.primitive_pynn_worker import (
    _reuse_configured_hardware_endpoint,
)
from utils.hardware.brainscales2.primitive_noise import (
    MockPrimitiveNoiseBackend,
    PRIMITIVES,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    default_primitive_coordinates,
    validate_primitive_observation,
)
from utils.hardware.brainscales2.primitive_optimization import (
    build_encoder_operating_point_candidates,
    parse_current_stop_pair,
    parse_exponential_pair,
    parse_precharge_pair,
    score_encoder_operating_point,
)
from scripts.evaluation.brainscales2_primitive_noise import (
    _collect_encoder_search_observations,
    _pynn_worker_timeout_locus,
    _primitive_summary_prerequisites,
    RepeatedPynnWorkerTimeoutError,
    build_parser,
    collect_observations,
    make_config,
    parse_reset_code_table,
    run,
)


def rejects(action, exceptions=(ValueError, RuntimeError)) -> None:
    try:
        action()
    except exceptions:
        return
    raise AssertionError("invalid primitive observation was accepted")


# @lat: [[hardware#Independent Primitive Noise Verification#Configured hardware endpoint reuse]]
def verify_configured_hardware_endpoint_reuse() -> None:
    with patch.dict(
        os.environ,
        {
            "QUIGGELDY_IP": "192.0.2.1",
            "QUIGGELDY_PORT": "12345",
            "JUPYTERHUB_USER": "hardware-user",
        },
        clear=True,
    ):
        assert _reuse_configured_hardware_endpoint()
        assert os.environ["QUIGGELDY_ENABLED"] == "1"
        assert os.environ["QUIGGELDY_USER_NO_MUNGE"] == "hardware-user"
    with patch.dict(
        os.environ,
        {"QUIGGELDY_IP": "192.0.2.1"},
        clear=True,
    ):
        assert not _reuse_configured_hardware_endpoint()


# @lat: [[hardware#Independent Primitive Noise Verification#Worker timeout cleanup]]
def verify_worker_timeout_cleanup() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker = root / "cleanup.txt"
        child = root / "worker.py"
        child.write_text(
            f"""
from pathlib import Path
import sys
import time

marker = Path(sys.argv[1])
sys.path.insert(0, {str(ROOT)!r})

from utils.hardware.brainscales2.primitive_pynn_worker import (
    _install_cleanup_interrupt_handler,
)

_install_cleanup_interrupt_handler()
print("ready", flush=True)
try:
    while True:
        time.sleep(1.0)
finally:
    marker.write_text("cleanup-started", encoding="utf-8")
    time.sleep(0.5)
    marker.write_text("released", encoding="utf-8")
""".strip(),
            encoding="utf-8",
        )
        try:
            primitive_backend_module._run_isolated_worker(
                [sys.executable, str(child), str(marker)],
                timeout_s=5.0,
                interrupt_grace_s=2.0,
                terminate_grace_s=0.1,
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("worker timeout was accepted")
        assert marker.read_text(encoding="utf-8") == "released"

        process = subprocess.Popen(
            [sys.executable, str(child), str(marker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 2.0
        while (
            not marker.is_file()
            or marker.read_text(encoding="utf-8") != "cleanup-started"
        ):
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise AssertionError("worker did not enter cleanup")
            time.sleep(0.01)
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=2.0)
        assert marker.read_text(encoding="utf-8") == "released"

    class ParentInterruptedProcess:
        def __init__(self) -> None:
            self.calls = 0
            self.signal_received = None
            self.returncode = 130

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

        def poll(self):
            return None if self.calls == 1 else self.returncode

        def send_signal(self, received_signal):
            self.signal_received = received_signal

    interrupted_process = ParentInterruptedProcess()
    with patch.object(
        primitive_backend_module.subprocess,
        "Popen",
        return_value=interrupted_process,
    ) as popen:
        try:
            primitive_backend_module._run_isolated_worker(
                ["synthetic-worker"],
                timeout_s=1.0,
                interrupt_grace_s=1.0,
                terminate_grace_s=0.1,
            )
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("parent interrupt was accepted")
    assert interrupted_process.signal_received == signal.SIGINT
    assert interrupted_process.calls == 2
    assert popen.call_args.kwargs["start_new_session"] is True


# @lat: [[hardware#Independent Primitive Noise Verification#Transient worker retry]]
def verify_transient_worker_error_classification() -> None:
    assert primitive_backend_module._is_transient_pynn_worker_error(
        "RuntimeError: Could not submit request."
    )
    assert primitive_backend_module._is_transient_pynn_worker_error(
        "socket.gaierror: Name or service not known"
    )
    assert primitive_backend_module._is_transient_pynn_worker_error(
        "Remote call timeout exceeded"
    )
    assert not primitive_backend_module._is_transient_pynn_worker_error(
        "ValueError: invalid physical coordinate"
    )


# @lat: [[hardware#Independent Primitive Noise Verification#PyNN worker attempt budget]]
def verify_pynn_worker_attempt_budget() -> None:
    parser = build_parser()
    default_config = make_config(
        parser.parse_args(["--output-dir", "/tmp/primitive-output"])
    )
    assert default_config.pynn_worker_max_attempts == 3
    assert default_config.to_manifest_dict()["pynn_worker_max_attempts"] == 3
    explicit_config = make_config(
        parser.parse_args(
            [
                "--output-dir",
                "/tmp/primitive-output",
                "--pynn-worker-max-attempts",
                "1",
            ]
        )
    )
    assert explicit_config.pynn_worker_max_attempts == 1
    rejects(lambda: PrimitiveNoiseConfig(pynn_worker_max_attempts=0))

    backend = PrimitiveHardwareBackend()
    original_worker_runner = primitive_backend_module._run_isolated_worker
    original_sleep = primitive_backend_module.time.sleep
    attempts = 0

    def timeout_then_succeed(command, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout_s"])
        response_path = Path(command[-1])
        torch.save(
            {
                "first": torch.ones((2, 1), dtype=torch.float64),
                "count": torch.ones((2, 1), dtype=torch.int64),
                "precharge": None,
                "metadata": {"chip_identifier": ["synthetic-chip"]},
            },
            response_path,
        )
        return SimpleNamespace(stdout="synthetic worker")

    retry_config = PrimitiveNoiseConfig(
        repeats=4,
        calibration_repeats=2,
        device_count=1,
        physical_coordinates=(0,),
        pynn_worker_timeout_s=10.0,
        pynn_worker_max_attempts=2,
    )
    primitive_backend_module._run_isolated_worker = timeout_then_succeed
    primitive_backend_module.time.sleep = lambda _seconds: None
    try:
        _, _, _, metadata = backend._run_pynn_code_process(
            "phi-np",
            "static",
            15,
            retry_config,
            repeats=2,
            trial_start=0,
        )
    finally:
        primitive_backend_module._run_isolated_worker = original_worker_runner
        primitive_backend_module.time.sleep = original_sleep
    assert attempts == 2
    assert metadata["worker_attempts"] == 2
    assert metadata["worker_max_attempts"] == 2

    attempts = 0

    def always_timeout(command, **kwargs):
        nonlocal attempts
        attempts += 1
        raise subprocess.TimeoutExpired(command, kwargs["timeout_s"])

    single_attempt_config = replace(retry_config, pynn_worker_max_attempts=1)
    primitive_backend_module._run_isolated_worker = always_timeout
    primitive_backend_module.time.sleep = lambda _seconds: None
    try:
        try:
            backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                single_attempt_config,
                repeats=2,
                trial_start=0,
            )
        except RuntimeError as error:
            assert "after 1 attempts" in str(error)
        else:
            raise AssertionError("single-attempt timeout was accepted")
    finally:
        primitive_backend_module._run_isolated_worker = original_worker_runner
        primitive_backend_module.time.sleep = original_sleep
    assert attempts == 1


# @lat: [[hardware#Independent Primitive Noise Verification#PyNN worker cache identity]]
def verify_pynn_worker_cache_identity() -> None:
    backend = PrimitiveHardwareBackend()
    original_worker_runner = primitive_backend_module._run_isolated_worker
    worker_calls: list[list[str]] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = PrimitiveNoiseConfig(
            repeats=4,
            calibration_repeats=2,
            device_count=1,
            physical_coordinates=(0,),
            pynn_worker_cache_dir=root / "canonical",
        )

        def fake_subprocess_run(command, **_kwargs):
            worker_calls.append(command)
            response_path = Path(command[-1])
            torch.save(
                {
                    "first": torch.ones((2, 1), dtype=torch.float64),
                    "count": torch.ones((2, 1), dtype=torch.int64),
                    "precharge": None,
                    "metadata": {"chip_identifier": ["synthetic-chip"]},
                },
                response_path,
            )
            return SimpleNamespace(stdout="synthetic worker")

        primitive_backend_module._run_isolated_worker = fake_subprocess_run
        try:
            first = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                config,
                repeats=2,
                trial_start=0,
            )
            operational_change = replace(
                config,
                pynn_worker_timeout_s=3600.0,
                pynn_worker_max_attempts=1,
            )
            second = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                operational_change,
                repeats=2,
                trial_start=0,
            )
        finally:
            primitive_backend_module._run_isolated_worker = original_worker_runner
        assert len(worker_calls) == 1
        assert first[3]["worker_cache_hit"] is False
        assert first[3]["worker_cache_identity"] == "canonical-v1"
        assert second[3]["worker_cache_hit"] is True
        assert second[3]["worker_cache_identity"] == "canonical-v1"
        assert second[3]["worker_max_attempts"] == 3

        fingerprint = primitive_backend_module._pynn_cache_fingerprint
        canonical = fingerprint(
            "phi-np", "static", 15, config, repeats=2, trial_start=0
        )
        assert canonical == fingerprint(
            "phi-np",
            "static",
            15,
            operational_change,
            repeats=2,
            trial_start=0,
        )
        moved_cache = replace(config, pynn_worker_cache_dir=root / "moved")
        assert canonical == fingerprint(
            "phi-np", "static", 15, moved_cache, repeats=2, trial_start=0
        )
        changed_chunks = replace(config, pynn_chunk_repeats=4)
        assert canonical != fingerprint(
            "phi-np", "static", 15, changed_chunks, repeats=2, trial_start=0
        )
        for changed in (
            replace(config, excitatory_input_i_bias_tau_code=79),
            replace(config, excitatory_input_i_bias_gm_code=524),
            replace(config, synaptic_input_drop_bias_code=300),
            replace(config, dynamic_reset_release_s=2.5e-6),
        ):
            assert canonical != fingerprint(
                "phi-np", "static", 15, changed, repeats=2, trial_start=0
            )

        for legacy in ("full-config", "before-attempt-budget"):
            legacy_config = replace(
                config, pynn_worker_cache_dir=root / legacy
            )
            legacy_config.pynn_worker_cache_dir.mkdir(parents=True)
            legacy_fingerprint = fingerprint(
                "phi-np",
                "static",
                15,
                legacy_config,
                repeats=2,
                trial_start=0,
                legacy=legacy,
            )
            cache_path = legacy_config.pynn_worker_cache_dir / (
                "phi-np_static_code15_trials0-2_"
                f"{legacy_fingerprint[:16]}.pt"
            )
            legacy_metadata = {
                "chip_identifier": ["synthetic-chip"],
                "worker_attempts": 1,
            }
            if legacy == "full-config":
                legacy_metadata["worker_max_attempts"] = 3
            torch.save(
                {
                    "fingerprint": legacy_fingerprint,
                    "first": torch.ones((2, 1), dtype=torch.float64),
                    "count": torch.ones((2, 1), dtype=torch.int64),
                    "precharge": None,
                    "metadata": legacy_metadata,
                },
                cache_path,
            )
            _, _, _, metadata = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                legacy_config,
                repeats=2,
                trial_start=0,
            )
            assert metadata["worker_cache_hit"] is True
            assert metadata["worker_cache_identity"] == legacy
            assert metadata["worker_max_attempts"] == 3


# @lat: [[hardware#Independent Primitive Noise Verification#Repeated acquisition timeout abort]]
def verify_repeated_acquisition_timeout_abort() -> None:
    timeout = RuntimeError(
        "PyNN worker timed out for phi-nl/static/code=0 after 3 attempts"
    )
    locus = _pynn_worker_timeout_locus(timeout)
    assert locus == {
        "primitive": "phi-nl",
        "stage": "static",
        "code": 0,
        "locus": "phi-nl/static/code=0",
    }
    assert _pynn_worker_timeout_locus(
        RuntimeError(
            "PyNN worker failed for phi-nl/static/code=0 after 3 attempts: "
            "Remote call timeout exceeded"
        )
    ) == locus
    assert _pynn_worker_timeout_locus(
        RuntimeError(
            "PyNN worker failed for phi-nl/static/code=0 after 1 attempts: "
            "ValueError: invalid physical coordinate"
        )
    ) is None

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "repeated-timeout"
        arguments = build_parser().parse_args(
            [
                "--phase",
                "optimize",
                "--primitive",
                "phi-np",
                "--backend",
                "mock",
                "--quick",
                "--device-count",
                "4",
                "--search-current-stop-pairs",
                "256:25",
                "384:25",
                "512:25",
                "--search-threshold-codes",
                "600",
                "--search-precharge-pairs",
                "1:63",
                "--search-timeout-abort-threshold",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        with patch.object(
            primitive_runner_module,
            "_collect_encoder_search_observations",
            side_effect=timeout,
        ) as collect:
            try:
                run(arguments)
            except RepeatedPynnWorkerTimeoutError as error:
                assert "phi-nl/static/code=0" in str(error)
                assert "outer controller can retry" in str(error)
            else:
                raise AssertionError("repeated PyNN timeout did not abort search")
        assert collect.call_count == 2
        payload = json.loads(
            (output / "infrastructure_error.json").read_text(encoding="utf-8")
        )
        assert payload["timeout_locus"] == "phi-nl/static/code=0"
        assert payload["consecutive_timeout_count"] == 2
        assert payload["configured_threshold"] == 2
        assert len(payload["trigger_candidate_ids"]) == 2
        candidate_results = sorted(
            (output / "candidates").glob("*/candidate_result.json")
        )
        assert len(candidate_results) == 2
        written = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in candidate_results
        ]
        trigger = next(
            item for item in written if item["consecutive_timeout_count"] == 2
        )
        assert trigger["timeout_locus"] == "phi-nl/static/code=0"

    assert (
        build_parser().parse_args(
            ["--output-dir", "."]
        ).search_timeout_abort_threshold
        == 2
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Synthetic transfer recovery]]
def verify_synthetic_transfer_recovery() -> None:
    config = PrimitiveNoiseConfig(repeats=32, calibration_repeats=16)
    backend = MockPrimitiveNoiseBackend()
    cases = (
        ("phi-np", "static", "slope_s_per_code", -20.0e-6 / 31.0, 0.12),
        ("phi-nl", "transfer", "tau_effective_s", 4.0e-6, 0.12),
        ("psi-int", "transfer", "gain", 0.98, 0.03),
        ("psi-ne", "transfer", "tau_effective_s", 10.0e-6, 0.18),
        ("psi-ed", "transfer", "tau_effective_s", 10.0e-6, 0.18),
    )
    for primitive, stage, key, expected, relative_tolerance in cases:
        observation = backend.collect(primitive, config, stage=stage, quick=False)
        result = validate_primitive_observation(observation, config)
        recovered = torch.tensor(
            [parameters[key] for parameters in result.calibration_parameters]
        ).median()
        assert abs(float(recovered) - expected) <= abs(expected) * relative_tolerance
        if primitive == "phi-nl":
            lower_bound = torch.tensor(
                [
                    parameters["lower_bound_code"]
                    for parameters in result.calibration_parameters
                ]
            ).median()
            assert abs(float(lower_bound) + 8.0) <= 1.0
        assert result.noise["temporal_sigma"]["representative_median"] > 0
        assert len(result.moment_rows) == (
            2 * observation.point_count * observation.device_count
        )
        assert all("mean" in row and "variance" in row for row in result.moment_rows)
        assert ("monotonicity" in result.gates) == (primitive == "phi-np")
        assert "rank_monotonicity_minimum" in result.noise
        assert "adjacent_order_fraction_minimum" in result.noise
        if primitive == "phi-np":
            assert result.noise["monotonicity_assessment"] == {
                "acceptance_metric": "rank_monotonicity",
                "diagnostic_metric": "adjacent_order_fraction",
                "rationale": (
                    "Temporal jitter can reverse neighboring measured means without "
                    "breaking the global transfer direction."
                ),
            }


# @lat: [[hardware#Independent Primitive Noise Verification#Held-out validation isolation]]
def verify_held_out_validation_isolation() -> None:
    config = PrimitiveNoiseConfig(repeats=16, calibration_repeats=8)
    observation = MockPrimitiveNoiseBackend().collect(
        "psi-int", config, stage="transfer", quick=True
    )
    clean_values = (
        0.4
        + 0.98
        * observation.ideal_variable.reshape(1, -1, 1).expand_as(
            observation.observed
        )
    )
    observation = replace(
        observation,
        observed=clean_values,
        delivered=torch.ones_like(observation.delivered),
        saturated=torch.zeros_like(observation.saturated),
    )
    reference = validate_primitive_observation(observation, config)
    balanced_values = observation.observed.clone()
    balanced_offsets = torch.tensor(
        [20.0, -20.0] * (config.validation_repeats // 2),
        dtype=balanced_values.dtype,
    ).reshape(-1, 1, 1)
    balanced_values[config.calibration_repeats :] += balanced_offsets
    balanced = validate_primitive_observation(
        replace(observation, observed=balanced_values), config
    )
    torch.testing.assert_close(
        torch.tensor(
            balanced.noise["validation_normalized_rmse_maximum"]
        ),
        torch.tensor(
            reference.noise["validation_normalized_rmse_maximum"]
        ),
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    shifted_values = observation.observed.clone()
    shifted_values[config.calibration_repeats :] += 20.0
    shifted = replace(observation, observed=shifted_values)
    result = validate_primitive_observation(shifted, config)
    for changed, original in zip(
        result.calibration_parameters, reference.calibration_parameters
    ):
        assert changed.keys() == original.keys()
        torch.testing.assert_close(
            torch.tensor(list(changed.values())),
            torch.tensor(list(original.values())),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert result.validation_refit_parameters != reference.validation_refit_parameters
    assert not result.gates["normalized_rmse"]


# @lat: [[hardware#Independent Primitive Noise Verification#Temporal and fixed-pattern separation]]
def verify_temporal_and_fixed_pattern_separation() -> None:
    coordinates = default_primitive_coordinates(4)
    config = PrimitiveNoiseConfig(
        repeats=8,
        calibration_repeats=4,
        device_count=4,
        physical_coordinates=coordinates,
    )
    codes = torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64)
    offsets = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64) * 1.0e-6
    mean = 25.0e-6 - codes * (20.0e-6 / 31.0)
    observed = (mean[None, :, None] + offsets[None, None, :]).repeat(8, 1, 1)
    delivered = torch.ones_like(observed, dtype=torch.bool)
    observation = PrimitiveObservation(
        primitive="phi-np",
        stage="static",
        output_kind="spike-time-s",
        input_code=codes,
        ideal_variable=codes,
        observed=observed,
        delivered=delivered,
        spike_count=torch.ones_like(observed, dtype=torch.int64),
        saturated=torch.zeros_like(delivered),
        physical_coordinates=coordinates,
    )
    result = validate_primitive_observation(observation, config)
    assert result.noise["temporal_sigma"]["representative_median"] < 1.0e-18
    fixed = result.noise["fixed_pattern_parameters"]["offset_s"]
    assert fixed["standard_deviation"] > 0


# @lat: [[hardware#Independent Primitive Noise Verification#Validated NP placement]]
def verify_validated_np_placement() -> None:
    coordinates = default_primitive_coordinates()
    assert coordinates == (
        3,
        11,
        13,
        1,
        134,
        141,
        128,
        130,
        273,
        261,
        262,
        274,
        390,
        386,
        388,
        392,
    )
    counts = [
        sum(base <= value < base + 128 for value in coordinates)
        for base in (0, 128, 256, 384)
    ]
    assert counts == [4, 4, 4, 4]
    assert len(coordinates) == len(set(coordinates))


# @lat: [[hardware#Independent Primitive Noise Verification#Miss and saturation semantics]]
def verify_miss_and_saturation_semantics() -> None:
    config = PrimitiveNoiseConfig(repeats=16, calibration_repeats=8)
    backend = MockPrimitiveNoiseBackend()
    base = backend.collect("phi-np", config, stage="static", quick=True)
    observed = base.observed.clone()
    delivered = base.delivered.clone()
    count = base.spike_count.clone()
    delivered[:, 0, :] = False
    count[:, 0, :] = 0
    observed[:, 0, :] = float("nan")
    missed = replace(base, observed=observed, delivered=delivered, spike_count=count)
    result = validate_primitive_observation(missed, config)
    assert not result.gates["point_observed"]
    assert bool(torch.isnan(missed.observed[:, 0]).all())

    invalid = observed.clone()
    invalid[:, 0, :] = config.deadline_s
    rejects(lambda: replace(missed, observed=invalid))

    sparse_observed = base.observed.clone()
    sparse_delivered = base.delivered.clone()
    sparse_count = base.spike_count.clone()
    sparse_observed[1:config.calibration_repeats] = float("nan")
    sparse_observed[config.calibration_repeats + 1 :] = float("nan")
    sparse_delivered[1:config.calibration_repeats] = False
    sparse_delivered[config.calibration_repeats + 1 :] = False
    sparse_count[1:config.calibration_repeats] = 0
    sparse_count[config.calibration_repeats + 1 :] = 0
    sparse = replace(
        base,
        observed=sparse_observed,
        delivered=sparse_delivered,
        spike_count=sparse_count,
    )
    sparse_result = validate_primitive_observation(sparse, config)
    assert sparse_result.gates["point_observed"]
    assert sparse_result.gates["multiple_spike_rate"]
    assert sparse_result.noise["miss_rate"] > 0.5

    multiple_count = base.spike_count.clone()
    multiple_count[0, 0, 0] = 2
    multiple = replace(base, spike_count=multiple_count)
    base_result = validate_primitive_observation(base, config)
    result = validate_primitive_observation(multiple, config)
    assert not result.gates["multiple_spike_rate"]
    assert result.noise["multiple_spike_rate"] > 0.0
    assert result.noise["maximum_spike_count"] == 2
    assert result.noise["miss_rate"] == base_result.noise["miss_rate"]
    assert result.noise["exactly_one_spike_rate"] < 1.0
    tolerant = validate_primitive_observation(
        multiple,
        replace(config, multiple_spike_rate_limit=0.002),
    )
    assert tolerant.gates["multiple_spike_rate"]

    one_deadline_miss_observed = base.observed.clone()
    one_deadline_miss_delivered = base.delivered.clone()
    one_deadline_miss_count = base.spike_count.clone()
    one_deadline_miss_observed[0, 0, 0] = float("nan")
    one_deadline_miss_delivered[0, 0, 0] = False
    one_deadline_miss_count[0, 0, 0] = 0
    one_deadline_miss = replace(
        base,
        observed=one_deadline_miss_observed,
        delivered=one_deadline_miss_delivered,
        spike_count=one_deadline_miss_count,
    )
    within_limit = validate_primitive_observation(one_deadline_miss, config)
    assert within_limit.gates["deadline_miss_rate"]
    excessive_observed = base.observed.clone()
    excessive_delivered = base.delivered.clone()
    excessive_count = base.spike_count.clone()
    excessive_observed[:8, 0, 0] = float("nan")
    excessive_delivered[:8, 0, 0] = False
    excessive_count[:8, 0, 0] = 0
    excessive = validate_primitive_observation(
        replace(
            base,
            observed=excessive_observed,
            delivered=excessive_delivered,
            spike_count=excessive_count,
        ),
        config,
    )
    assert not excessive.gates["deadline_miss_rate"]

    pwm = backend.collect("psi-int", config, stage="transfer", quick=True)
    saturated = pwm.saturated.clone()
    saturated[:, 0, :] = True
    result = validate_primitive_observation(replace(pwm, saturated=saturated), config)
    assert not result.gates["saturation"]

    ne = backend.collect("psi-ne", config, stage="transfer", quick=True)
    spike_count = ne.spike_count.clone()
    spike_count[0, 0, 0] = 1
    result = validate_primitive_observation(replace(ne, spike_count=spike_count), config)
    assert not result.gates["premature_spike"]

    endpoint_base = backend.collect("phi-np", config, stage="static", quick=False)
    endpoint_shifted = endpoint_base.observed.clone()
    endpoint_shifted[:, -1, :] += 10.0e-6
    base_result = validate_primitive_observation(endpoint_base, config)
    endpoint_result = validate_primitive_observation(
        replace(endpoint_base, observed=endpoint_shifted), config
    )
    torch.testing.assert_close(
        torch.tensor(
            endpoint_result.noise["validation_normalized_rmse_maximum"]
        ),
        torch.tensor(base_result.noise["validation_normalized_rmse_maximum"]),
        rtol=1.0e-6,
        atol=1.0e-12,
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Rank monotonicity gate]]
def verify_rank_monotonicity_gate() -> None:
    values = torch.arange(31, dtype=torch.float64)
    for left in (2, 7, 12, 17, 22):
        values[left], values[left + 1] = (
            values[left + 1].clone(),
            values[left].clone(),
        )
    assert primitive_noise_module._monotonicity(values, 1.0) < 0.95
    assert primitive_noise_module._rank_monotonicity(values, 1.0) >= 0.95

# @lat: [[hardware#Independent Primitive Noise Verification#Segmented PyNN recording decode]]
def verify_segmented_pynn_recording_decode() -> None:
    def train(
        values: list[float], source_index: int | None = None
    ) -> SimpleNamespace:
        annotations = (
            {} if source_index is None else {"source_index": source_index}
        )
        return SimpleNamespace(magnitude=values, annotations=annotations)

    segmented = [
        train([], 0),
        train([], 1),
        train([0.0052, 0.0061], 0),
        train([0.0054], 1),
        train([], 0),
        train([], 1),
        train([], 0),
        train([], 1),
        train([0.0653], 0),
        train([0.0655, 0.0660], 1),
        train([], 0),
        train([], 1),
    ]
    first, count = PrimitiveHardwareBackend._decode_pynn_spikes(
        segmented,
        repeats=2,
        devices=2,
        window_ms=0.060,
    )
    torch.testing.assert_close(
        first,
        torch.tensor(
            [[5.2e-6, 5.4e-6], [5.3e-6, 5.5e-6]], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        count,
        torch.tensor([[2, 1], [1, 2]], dtype=torch.int64),
    )

    unsegmented, unsegmented_count = PrimitiveHardwareBackend._decode_pynn_spikes(
        [train([0.005]), train([0.006])],
        repeats=1,
        devices=2,
        window_ms=0.060,
    )
    torch.testing.assert_close(
        unsegmented,
        torch.tensor([[5.0e-6, 6.0e-6]], dtype=torch.float64),
    )
    assert bool((unsegmented_count == 1).all())

    rejects(
        lambda: PrimitiveHardwareBackend._decode_pynn_spikes(
            [train([], 0), train([], 0)],
            repeats=1,
            devices=2,
            window_ms=0.060,
        )
    )
    rejects(
        lambda: PrimitiveHardwareBackend._decode_pynn_spikes(
            [train([]), train([]), train([]), train([])],
            repeats=1,
            devices=2,
            window_ms=0.060,
        )
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Dynamic recording selection]]
def verify_dynamic_recording_selection() -> None:
    class Population:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def record(self, variable: str, *, device: str | None = None) -> None:
            self.calls.append((variable, device))

    static = Population()
    PrimitiveHardwareBackend._record_pynn_population(static, membrane=False)
    assert static.calls == [("spikes", None)]

    dynamic = Population()
    PrimitiveHardwareBackend._record_pynn_population(dynamic, membrane=True)
    assert dynamic.calls == [("spikes", None), ("v", "cadc")]


# @lat: [[hardware#Independent Primitive Noise Verification#Segmented dense recording decode]]
def verify_segmented_dense_recording_decode() -> None:
    def signal(
        source_id: int, times: list[float], values: list[float]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            magnitude=torch.tensor(values, dtype=torch.float64).reshape(-1, 1),
            times=SimpleNamespace(magnitude=times),
            annotations={"source_ids": [source_id]},
        )

    segment = SimpleNamespace(
        irregularlysampledsignals=[
            signal(0, [0.001, 0.003], [10.0, 30.0]),
            signal(1, [0.002, 0.004], [20.0, 40.0]),
            signal(0, [0.010], [100.0]),
            signal(1, [0.011], [110.0]),
        ],
        analogsignals=[],
    )
    selected = PrimitiveHardwareBackend._decode_pynn_precharge(
        segment,
        devices=2,
        sample_ms=0.0045,
    )
    torch.testing.assert_close(
        selected, torch.tensor([30.0, 40.0], dtype=torch.float64)
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Sparse precharge evidence]]
def verify_sparse_precharge_evidence() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    observation = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="dynamic", quick=True
    )
    sparse = torch.full_like(observation.precharge_cadc, torch.nan)
    sparse[0] = observation.precharge_cadc[0]
    validation = validate_primitive_observation(
        replace(observation, precharge_cadc=sparse), config
    )
    assert validation.gates["precharge_recorded"]
    assert "precharge_rank_monotonicity_minimum" in validation.noise

    sparse[:, :, 0] = torch.nan
    validation = validate_primitive_observation(
        replace(observation, precharge_cadc=sparse), config
    )
    assert not validation.gates["precharge_recorded"]


# @lat: [[hardware#Independent Primitive Noise Verification#Installed component resolution]]
def verify_installed_component_resolution() -> None:
    spike_source = object()
    static_synapse = object()
    pynn = SimpleNamespace(
        cells=SimpleNamespace(SpikeSourceArray=spike_source),
        synapses=SimpleNamespace(StaticSynapse=static_synapse),
    )
    assert (
        PrimitiveHardwareBackend._pynn_component(
            pynn, "SpikeSourceArray", "cells"
        )
        is spike_source
    )
    assert (
        PrimitiveHardwareBackend._pynn_component(
            pynn, "StaticSynapse", "synapses"
        )
        is static_synapse
    )
    rejects(
        lambda: PrimitiveHardwareBackend._pynn_component(
            SimpleNamespace(), "SpikeSourceArray", "cells"
        )
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Correlation recording decode]]
def verify_correlation_recording_decode() -> None:
    recording = [
        [
            SimpleNamespace(data=torch.tensor([150, 151])),
            SimpleNamespace(data=torch.tensor([145, 147])),
            SimpleNamespace(data=torch.tensor([140, 143])),
        ]
    ]
    decoded = _decode_correlation(recording, periods=3, devices=2)
    torch.testing.assert_close(
        decoded,
        torch.tensor(
            [[150.0, 151.0], [145.0, 147.0], [140.0, 143.0]],
            dtype=torch.float64,
        ),
    )
    rejects(
        lambda: _decode_correlation(recording, periods=2, devices=2)
    )
    rejects(
        lambda: _decode_correlation(recording, periods=3, devices=3)
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Correlation worker boundary]]
def verify_correlation_worker_boundary() -> None:
    orders = _point_orders(repeats=3, point_count=5, seed=7, trial_start=11)
    assert orders == _point_orders(
        repeats=3, point_count=5, seed=7, trial_start=11
    )
    assert orders != _point_orders(
        repeats=3, point_count=5, seed=7, trial_start=12
    )
    canonical = torch.arange(30, dtype=torch.float64).reshape(3, 5, 2)
    scheduled = torch.empty_like(canonical)
    for trial, order in enumerate(orders):
        for acquisition_point, point in enumerate(order):
            scheduled[trial, acquisition_point] = canonical[trial, point]
    torch.testing.assert_close(
        _restore_point_order(scheduled, orders), canonical
    )
    rejects(
        lambda: _restore_point_order(scheduled, [[0, 1, 2, 3, 3]] * 3)
    )

    calls: list[tuple[int, int]] = []

    def fake_process(config, *, differences, repeats, trial_start):
        calls.append((trial_start, repeats))
        shape = (repeats, differences.numel(), config.device_count)
        response = 5.0 + 20.0 * torch.exp(differences / 10.0e-6)
        quiet = torch.full(shape, 180.0, dtype=torch.float64)
        stimulated = quiet - response.reshape(1, -1, 1)
        first = torch.full(shape, 35.0e-6, dtype=torch.float64)
        count = torch.ones(shape, dtype=torch.int64)
        if trial_start == 0:
            first[0, 0, 0] = config.deadline_s + 1.0e-6
            count[0, 1, 0] = 2
        return {
            "quiet": quiet,
            "stimulated": stimulated,
            "first": first,
            "count": count,
            "metadata": {"chip_identifier": ["synthetic-chip"]},
        }

    original_probe = primitive_backend_module.probe_primitive_capabilities
    with tempfile.TemporaryDirectory() as temporary:
        calibration = Path(temporary) / "correlation.pkl"
        calibration.write_bytes(b"correlation calibration")
        config = PrimitiveNoiseConfig(
            repeats=8,
            calibration_repeats=4,
            device_count=2,
            physical_coordinates=(0, 1),
            correlation_calibration_path=calibration,
            psi_ed_chunk_repeats=3,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_psi_ed_process = fake_process  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "psi-ed": True,
            "pynn_version": "synthetic",
        }
        try:
            observation = backend.collect("psi-ed", config, quick=True)
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 3), (3, 3), (6, 2)]
    assert tuple(observation.observed.shape) == (8, 3, 2)
    assert int((~observation.delivered).sum()) == 1
    assert int((observation.spike_count > 1).sum()) == 1
    assert int(torch.isfinite(observation.first_spike_time_s).sum()) == 48
    finite_first = observation.first_spike_time_s[
        torch.isfinite(observation.first_spike_time_s)
    ]
    assert float(finite_first.max()) > config.deadline_s
    assert observation.metadata["multiple_spike_handling"] == "first-spike-time"
    assert observation.metadata["deadline_applies"] is True


# @lat: [[hardware#Independent Primitive Noise Verification#Nonlinear drive configuration]]
def verify_nonlinear_drive_configuration() -> None:
    config = PrimitiveNoiseConfig()
    assert config.exponential_input_weight == 63
    assert config.exponential_input_fan_in == 8
    assert config.to_manifest_dict()["exponential_input_fan_in"] == 8
    assert config.pynn_chunk_repeats == 8
    assert config.to_manifest_dict()["pynn_chunk_repeats"] == 8
    assert config.pynn_process_repeats == 32
    assert config.to_manifest_dict()["pynn_process_repeats"] == 32
    assert config.hagen_num_sends == 1024
    assert config.hagen_candidate_count == 128
    assert config.hagen_chunk_repeats == 16
    assert config.psi_ed_chunk_repeats == 32
    assert config.deadline_miss_rate_limit == 0.01
    manifest = config.to_manifest_dict()
    assert manifest["hagen_output_coordinates"] is None
    assert len(manifest["hagen_candidate_output_indices"]) == 128
    rejects(lambda: PrimitiveNoiseConfig(exponential_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_process_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(precharge_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_worker_timeout_s=0))
    rejects(lambda: PrimitiveNoiseConfig(reset_code_table=(300,) * 31))
    rejects(lambda: PrimitiveNoiseConfig(reset_code_table=(1023,) * 32))
    lookup = tuple(range(32))
    lookup_config = PrimitiveNoiseConfig(reset_code_table=lookup)
    assert PrimitiveHardwareBackend._reset_code(17, lookup_config) == 17
    assert lookup_config.to_manifest_dict()["reset_code_table"] == lookup
    vector_lookup = tuple(tuple(code for _ in range(16)) for code in range(32))
    vector_config = PrimitiveNoiseConfig(reset_code_table=vector_lookup)
    assert PrimitiveHardwareBackend._reset_code(17, vector_config) == (17,) * 16
    assert vector_config.to_manifest_dict()["reset_code_table"] == vector_lookup
    rejects(
        lambda: PrimitiveNoiseConfig(
            reset_code_table=tuple((code,) for code in range(32))
        )
    )
    assert parse_reset_code_table(list(range(32)), device_count=16) == lookup
    flat_vector = [code for code in range(32) for _ in range(16)]
    assert parse_reset_code_table(flat_vector, device_count=16) == vector_lookup
    rejects(lambda: parse_reset_code_table([0, 1], device_count=16))
    rejects(lambda: PrimitiveNoiseConfig(hagen_num_sends=0))
    rejects(lambda: PrimitiveNoiseConfig(hagen_candidate_count=15))
    rejects(lambda: PrimitiveNoiseConfig(hagen_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(psi_ed_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(deadline_miss_rate_limit=1.01))

    calls: list[tuple[int, int]] = []

    def fake_run(primitive, stage, code, config, *, repeats=None):
        assert primitive == "phi-np"
        assert stage == "static"
        assert repeats is not None
        calls.append((code, repeats))
        first = torch.full(
            (repeats, config.device_count),
            10.0e-6 - code * 0.1e-6,
            dtype=torch.float64,
        )
        count = torch.ones_like(first, dtype=torch.int64)
        return first, count, None, {"chip_identifier": ["synthetic-chip"]}

    original_probe = primitive_backend_module.probe_primitive_capabilities
    with tempfile.TemporaryDirectory() as directory:
        calibration = Path(directory) / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        chunk_config = PrimitiveNoiseConfig(
            repeats=5,
            calibration_repeats=2,
            device_count=2,
            physical_coordinates=(0, 1),
            spiking_calibration_path=calibration,
            pynn_chunk_repeats=2,
            pynn_process_repeats=None,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_pynn_code = fake_run  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "phi-np": True,
            "pynn_version": "synthetic",
        }
        try:
            observation = backend.collect(
                "phi-np", chunk_config, stage="static", quick=True
            )
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 2), (0, 2), (0, 1), (15, 2), (15, 2), (15, 1), (30, 2), (30, 2), (30, 1)]
    assert tuple(observation.observed.shape) == (5, 3, 2)
    assert observation.metadata["chunk_repeats"] == 2
    assert observation.metadata["chip_identifier"] == ["synthetic-chip"]
    calibration_indices = observation.metadata["calibration_acquisition_indices"]
    validation_indices = observation.metadata["validation_acquisition_indices"]
    assert len(calibration_indices) == 2
    assert len(validation_indices) == 3
    assert sorted(calibration_indices + validation_indices) == list(range(5))
    repeated_order = PrimitiveHardwareBackend._split_trial_order(5, 2, seed=0)
    assert calibration_indices == list(repeated_order[1])
    assert validation_indices == list(repeated_order[2])
    process_chunks = observation.metadata["per_code"][0]["chunks"]
    assert [(chunk["trial_start"], chunk["trial_stop"]) for chunk in process_chunks] == [
        (0, 5)
    ]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in process_chunks[0]["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]

    process_calls: list[tuple[int, int]] = []

    def fake_process(primitive, stage, code, config, *, repeats, trial_start):
        process_calls.append((code, repeats))
        first = torch.full(
            (repeats, config.device_count),
            10.0e-6 - code * 0.1e-6,
            dtype=torch.float64,
        )
        count = torch.ones_like(first, dtype=torch.int64)
        return first, count, None, {"chip_identifier": ["synthetic-chip"]}

    with tempfile.TemporaryDirectory() as directory:
        calibration = Path(directory) / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        process_config = replace(
            chunk_config,
            spiking_calibration_path=calibration,
            pynn_process_repeats=2,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_pynn_code_process = fake_process  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "phi-np": True,
            "pynn_version": "synthetic",
        }
        try:
            isolated = backend.collect(
                "phi-np", process_config, stage="static", quick=True
            )
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert process_calls == [
        (0, 2), (0, 2), (0, 1),
        (15, 2), (15, 2), (15, 1),
        (30, 2), (30, 2), (30, 1),
    ]
    assert tuple(isolated.observed.shape) == (5, 3, 2)
    assert isolated.metadata["process_repeats"] == 2

    worker_calls: list[list[str]] = []
    original_worker_runner = primitive_backend_module._run_isolated_worker
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calibration = root / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        cache_config = PrimitiveNoiseConfig(
            repeats=8,
            calibration_repeats=4,
            device_count=2,
            physical_coordinates=(0, 1),
            spiking_calibration_path=calibration,
            pynn_worker_cache_dir=root / "cache",
        )

        def fake_subprocess_run(command, **_kwargs):
            worker_calls.append(command)
            response_path = Path(command[-1])
            torch.save(
                {
                    "first": torch.ones((8, 2), dtype=torch.float64),
                    "count": torch.ones((8, 2), dtype=torch.int64),
                    "precharge": None,
                    "metadata": {"chip_identifier": ["synthetic-chip"]},
                },
                response_path,
            )
            return SimpleNamespace(stdout="synthetic worker")

        primitive_backend_module._run_isolated_worker = fake_subprocess_run
        try:
            backend = PrimitiveHardwareBackend()
            first = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                cache_config,
                repeats=8,
                trial_start=0,
            )
            second = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                cache_config,
                repeats=8,
                trial_start=0,
            )
        finally:
            primitive_backend_module._run_isolated_worker = original_worker_runner
        assert len(worker_calls) == 1
        assert first[3]["worker_cache_hit"] is False
        assert first[3]["worker_attempts"] == 1
        assert first[3]["worker_max_attempts"] == 3
        assert second[3]["worker_cache_hit"] is True
        assert second[3]["worker_attempts"] == 1
        assert second[3]["worker_max_attempts"] == 3
        torch.testing.assert_close(first[0], second[0])


# @lat: [[hardware#Independent Primitive Noise Verification#Static reset separation]]
def verify_static_reset_separation() -> None:
    class Population:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def set(self, **parameters: object) -> None:
            self.calls.append(parameters)

    population = Population()
    PrimitiveHardwareBackend._set_population_reset(population, 300)
    PrimitiveHardwareBackend._set_population_reset(population, (301, 302))
    assert population.calls == [
        {"reset_v_reset": 300},
        {"reset_v_reset": [301, 302]},
    ]
    release_config = PrimitiveNoiseConfig(
        static_reset_release_s=4.5e-6,
        dynamic_reset_release_s=2.5e-6,
    )
    assert PrimitiveHardwareBackend._reset_release_s(
        "phi-np", "static", release_config
    ) == 4.5e-6
    assert PrimitiveHardwareBackend._reset_release_s(
        "phi-np", "dynamic", release_config
    ) == 2.5e-6
    assert PrimitiveHardwareBackend._reset_release_s(
        "phi-nl", "transfer", release_config
    ) == 2.5e-6

    class FakeEnum(int):
        pass

    class FakeBlock:
        size = 4

        def __init__(self, value: int) -> None:
            self.value = int(value)

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeBlock) and self.value == other.value

    class FakeValue(int):
        pass

    comparator_values = {
        FakeBlock(index): FakeValue(200) for index in range(FakeBlock.size)
    }
    synaptic_drop_values = {
        FakeBlock(index): FakeValue(300) for index in range(FakeBlock.size)
    }
    comparator_chip = SimpleNamespace(
        neuron_block=SimpleNamespace(
            i_bias_threshold_comparator=comparator_values,
            i_bias_synin_drop=synaptic_drop_values,
        )
    )
    fake_vx = SimpleNamespace(
        halco=SimpleNamespace(
            CapMemBlockOnDLS=FakeBlock,
            common=SimpleNamespace(Enum=FakeEnum),
        )
    )
    original_import_module = primitive_backend_module.import_module
    primitive_backend_module.import_module = (
        lambda name: fake_vx
        if name == "dlens_vx_v3"
        else original_import_module(name)
    )
    try:
        PrimitiveHardwareBackend._configure_threshold_comparator_bias(
            comparator_chip, None
        )
        assert {int(value) for value in comparator_values.values()} == {200}
        PrimitiveHardwareBackend._configure_threshold_comparator_bias(
            comparator_chip, 800
        )
        PrimitiveHardwareBackend._configure_synaptic_input_drop_bias(
            comparator_chip, None
        )
        assert {int(value) for value in synaptic_drop_values.values()} == {300}
        PrimitiveHardwareBackend._configure_synaptic_input_drop_bias(
            comparator_chip, 450
        )
    finally:
        primitive_backend_module.import_module = original_import_module
    assert {int(value) for value in comparator_values.values()} == {800}
    assert {int(value) for value in synaptic_drop_values.values()} == {450}

    cell_config = PrimitiveNoiseConfig(
        excitatory_input_i_bias_tau_code=79,
        excitatory_input_i_bias_gm_code=524,
        membrane_capacitance_code=32,
    )
    cell_parameters = PrimitiveHardwareBackend._pynn_cell_parameters(
        "dynamic",
        cell_config,
        300,
        {"refractory_period_refractory_time": 511},
    )
    assert cell_parameters["excitatory_input_i_bias_tau"] == 79
    assert cell_parameters["excitatory_input_i_bias_gm"] == 524
    assert cell_parameters["membrane_capacitance_capacitance"] == 32
    assert cell_parameters["reset_v_reset"] == 300
    assert cell_parameters["refractory_period_refractory_time"] == 511

    settings = SimpleNamespace(
        refractory_counters=list(range(512)),
        reset_holdoff=[value + 1 for value in range(512)],
        input_clock=[value % 2 for value in range(512)],
    )
    selected = PrimitiveHardwareBackend._selected_refractory_parameters(
        settings, [3, 134, 273, 390]
    )
    assert selected == {
        "refractory_period_refractory_time": [3, 134, 273, 390],
        "refractory_period_reset_holdoff": [4, 135, 274, 391],
        "refractory_period_input_clock": [1, 0, 1, 0],
    }
    malformed_settings = SimpleNamespace(
        refractory_counters=[1],
        reset_holdoff=settings.reset_holdoff,
        input_clock=settings.input_clock,
    )
    rejects(
        lambda: PrimitiveHardwareBackend._selected_refractory_parameters(
            malformed_settings, [0]
        ),
        (RuntimeError,),
    )

    applied: dict[str, object] = {}

    class RefractorySettings:
        refractory_counters = [31] * 512
        reset_holdoff = [0] * 512
        input_clock = [0] * 512
        fast_clock = 9
        slow_clock = 12

        def apply_to_chip(self, chip: object) -> None:
            applied["chip"] = chip

    def calculate_settings(targets: torch.Tensor) -> RefractorySettings:
        applied["targets"] = targets.clone()
        return RefractorySettings()

    original_import_module = primitive_backend_module.import_module

    def fake_import_module(name: str):
        if name == "numpy":
            return SimpleNamespace(
                full=lambda count, value: torch.full((count,), value)
            )
        if name == "quantities":
            return SimpleNamespace(s=1.0)
        if name == "calix.spiking.refractory_period":
            return SimpleNamespace(calculate_settings=calculate_settings)
        return original_import_module(name)

    primitive_backend_module.import_module = fake_import_module
    try:
        refractory_config = PrimitiveNoiseConfig(
            repeats=2,
            calibration_repeats=1,
            device_count=4,
            physical_coordinates=(3, 134, 273, 390),
            deadline_s=900.0e-6,
        )
        chip = object()
        _, refractory_metadata = (
            PrimitiveHardwareBackend._configure_first_spike_refractory(
                chip, refractory_config
            )
        )
    finally:
        primitive_backend_module.import_module = original_import_module
    assert applied["chip"] is chip
    torch.testing.assert_close(
        applied["targets"], torch.full((512,), 1.8e-3)
    )
    assert refractory_metadata["target_s"] == 1.8e-3

    source = inspect.getsource(
        PrimitiveHardwareBackend._run_pynn_code
    )
    trial_loop = source.index("for trial in range(total_trials)")
    initial_run = source.index("pynn.run(reference_ms, append)", trial_loop)
    ramp_enable = source.index(
        "population.set(constant_current_enable=True)", initial_run
    )
    state_retention = source[trial_loop:ramp_enable]
    assert "_set_population_reset" not in state_retention
    assert "config.reset_code_minimum" not in state_retention
    assert '"static_reset_policy"' in source
    assert '"input code retained for the full trial"' in source
    assert "2.0 * config.deadline_s" in source
    assert "_configure_first_spike_refractory" in source


# @lat: [[hardware#Independent Primitive Noise Verification#Hagen output qualification]]
def verify_hagen_output_qualification() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/primitive-output",
            "--hagen-chunk-repeats",
            "7",
        ]
    )
    config = make_config(args)
    assert config.hagen_chunk_repeats == 7
    assert config.to_manifest_dict()["hagen_chunk_repeats"] == 7

    class FakeHagenBackend:
        calls: list[int] = []

        def __init__(self, _config) -> None:
            pass

        def hardware_session(self):
            return nullcontext()

        def direct_linear(self, value, weight, *, avg):
            assert avg == 1
            self.calls.append(int(value.shape[0]))
            output = value @ weight.T
            return SimpleNamespace(
                value=output,
                metadata={"chip_identifier": ["synthetic-chip"]},
            )

    chunk_config = PrimitiveNoiseConfig(
        repeats=5,
        calibration_repeats=2,
        device_count=2,
        physical_coordinates=(0, 1),
        allow_environment_calibration=True,
        hagen_candidate_count=2,
        hagen_chunk_repeats=2,
    )
    original_backend = primitive_backend_module.HagenPWMBackend
    primitive_backend_module.HagenPWMBackend = FakeHagenBackend
    try:
        chunked = PrimitiveHardwareBackend().collect(
            "psi-int", chunk_config, quick=True
        )
    finally:
        primitive_backend_module.HagenPWMBackend = original_backend
    assert FakeHagenBackend.calls == [6, 6, 3] * 4
    assert tuple(chunked.observed.shape) == (5, 12, 2)
    assert chunked.metadata["chunk_repeats"] == 2
    assert chunked.metadata["chip_identifier"] == ["synthetic-chip"]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in chunked.metadata["per_drive"][0]["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]

    ideal = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    observed = torch.zeros((6, 3, 3), dtype=torch.float64)
    observed[:, :, 0] = ideal
    observed[:, :, 1] = ideal + torch.tensor([0.4, -0.3, 0.2])
    observed[:, :, 2] = ideal + torch.tensor([0.05, -0.05, 0.05])
    observed[3:, :, 0] += 50.0
    saturated = torch.zeros_like(observed, dtype=torch.bool)
    selected, scores = PrimitiveHardwareBackend._select_hagen_outputs(
        observed,
        saturated,
        ideal,
        calibration_repeats=3,
        selected_count=2,
    )
    assert selected == (0, 2)
    assert len(scores) == 3
    assert [int(row["output_index"]) for row in scores] == [0, 1, 2]
    observed[3:] -= 100.0
    repeated, _ = PrimitiveHardwareBackend._select_hagen_outputs(
        observed,
        saturated,
        ideal,
        calibration_repeats=3,
        selected_count=2,
    )
    assert repeated == selected


# @lat: [[hardware#Independent Primitive Noise Verification#Exponential response observation time]]
def verify_exponential_response_observation_time() -> None:
    parser = build_parser()
    default_args = parser.parse_args(["--output-dir", "/tmp/primitive-output"])
    default_config = make_config(default_args)
    assert default_config.observation_time_s == 40.0e-6
    assert default_config.psi_ne_input_fan_in == 2
    assert default_config.psi_ne_chunk_repeats == 4
    input_times = torch.tensor([5.0e-6, 25.0e-6], dtype=torch.float64)
    events = PrimitiveHardwareBackend._psi_ne_input_events(
        input_times,
        repeats=2,
        runtime_steps=61,
        dt_s=1.0e-6,
        fan_in=2,
    )
    assert tuple(events.shape) == (61, 8, 2)
    assert int(events[:, 0::2].sum()) == 0
    assert int(events[:, 1::2].sum()) == 8
    assert bool((events[5, 1, :] == 1).all())
    assert bool((events[25, 3, :] == 1).all())
    explicit_args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/primitive-output",
            "--observation-time",
            "45e-6",
            "--psi-ne-input-fan-in",
            "3",
            "--pynn-chunk-repeats",
            "5",
            "--psi-ne-chunk-repeats",
            "7",
        ]
    )
    explicit_config = make_config(explicit_args)
    assert explicit_config.observation_time_s == 45.0e-6
    assert explicit_config.psi_ne_input_fan_in == 3
    assert explicit_config.pynn_chunk_repeats == 5
    assert explicit_config.psi_ne_chunk_repeats == 7
    rejects(lambda: PrimitiveNoiseConfig(psi_ne_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(psi_ne_chunk_repeats=0))

    calls: list[tuple[int, int]] = []

    def fake_process(
        config,
        *,
        input_times,
        runtime_steps,
        observation_step,
        repeats,
        trial_start,
    ):
        calls.append((trial_start, repeats))
        shape = (repeats, input_times.numel(), config.device_count)
        mean = 5.0 + 30.0 * torch.exp(
            -(config.observation_time_s - input_times) / 10.0e-6
        )
        observed = mean.reshape(1, -1, 1).expand(shape).clone()
        return {
            "baseline": torch.full(shape, -40.0, dtype=torch.float64),
            "observed": observed,
            "spike_count": torch.zeros(shape, dtype=torch.int64),
            "saturated": torch.zeros(shape, dtype=torch.bool),
            "calibration_loader": "synthetic-loader",
            "batch_count": 2 * repeats * input_times.numel(),
            "metadata": {"chip_identifier": ["synthetic-chip"]},
        }

    process_config = PrimitiveNoiseConfig(
        repeats=5,
        calibration_repeats=2,
        device_count=2,
        physical_coordinates=(0, 1),
        allow_environment_calibration=True,
        psi_ne_chunk_repeats=2,
    )
    original_probe = primitive_backend_module.probe_primitive_capabilities
    backend = PrimitiveHardwareBackend()
    backend._run_psi_ne_process = fake_process  # type: ignore[method-assign]
    primitive_backend_module.probe_primitive_capabilities = lambda: {
        "psi-ne": True,
        "hxtorch_version": "synthetic",
    }
    try:
        observation = backend.collect("psi-ne", process_config, quick=True)
    finally:
        primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 2), (2, 2), (4, 1)]
    assert tuple(observation.observed.shape) == (5, 3, 2)
    assert observation.metadata["chip_identifier"] == ["synthetic-chip"]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in observation.metadata["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]


# @lat: [[hardware#Independent Primitive Noise Verification#Insufficient fit preservation]]
def verify_insufficient_fit_preservation() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    base = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="static", quick=True
    )
    observed = base.observed.clone()
    delivered = base.delivered.clone()
    spike_count = base.spike_count.clone()
    observed[:, :, 0] = float("nan")
    delivered[:, :, 0] = False
    spike_count[:, :, 0] = 0
    missing_device = replace(
        base,
        observed=observed,
        delivered=delivered,
        spike_count=spike_count,
    )
    validation = validate_primitive_observation(missing_device, config)
    assert not validation.validated
    assert validation.gates["fit_available"] is False
    assert validation.device_statistics[0]["fit_available"] is False
    assert validation.device_statistics[1]["fit_available"] is True

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[missing_device],
            validations=[validation],
        )
        assert (output / "raw" / "index.json").is_file()
        payload = json.loads(
            (output / "primitive_noise_calibration.json").read_text()
        )
        assert not payload["validated"]


# @lat: [[hardware#Independent Primitive Noise Verification#NP stage gate]]
def verify_np_stage_gate() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    static = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="static", quick=True
    )
    validation = validate_primitive_observation(static, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        calibration = write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[static],
            validations=[validation],
        )
        assert not calibration["primitives"]["phi-np"]["validated"]
        assert calibration["primitives"]["phi-np"]["diagnostic_only"]
        assert not calibration["validated"]

    dynamic = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="dynamic", quick=True
    )
    failed_dynamic = replace(
        dynamic,
        spike_count=torch.full_like(dynamic.spike_count, 2),
    )
    failed_validation = validate_primitive_observation(failed_dynamic, config)
    with tempfile.TemporaryDirectory() as temporary:
        calibration = write_primitive_noise_artifacts(
            Path(temporary),
            config=config,
            observations=[static, failed_dynamic],
            validations=[validation, failed_validation],
        )
        np_result = calibration["primitives"]["phi-np"]
        assert not np_result["validated"] and np_result["diagnostic_only"]
        assert np_result["transfer_parameters"] is None
        assert np_result["noise"] is None

    original_collect = MockPrimitiveNoiseBackend.collect
    calls: list[tuple[str, str]] = []

    def failing_static(self, primitive, run_config, *, stage="transfer", quick=False):
        calls.append((primitive, stage))
        result = original_collect(
            self, primitive, run_config, stage=stage, quick=quick
        )
        if primitive == "phi-np" and stage == "static":
            return replace(
                result,
                spike_count=torch.full_like(result.spike_count, 2),
            )
        return result

    MockPrimitiveNoiseBackend.collect = failing_static
    try:
        observations, _ = collect_observations(
            SimpleNamespace(backend="mock", primitive="all", quick=True),
            config,
        )
    finally:
        MockPrimitiveNoiseBackend.collect = original_collect
    assert ("phi-np", "dynamic") not in calls
    assert ("phi-nl", "transfer") not in calls
    assert {item.primitive for item in observations} == {
        "phi-np",
        "psi-int",
        "psi-ne",
        "psi-ed",
    }


# @lat: [[hardware#Independent Primitive Noise Verification#Encoder operating point score]]
def verify_encoder_operating_point_score() -> None:
    config = PrimitiveNoiseConfig(
        repeats=32,
        calibration_repeats=16,
        device_count=4,
        physical_coordinates=default_primitive_coordinates(4),
    )
    backend = MockPrimitiveNoiseBackend()

    original_collect = MockPrimitiveNoiseBackend.collect

    def one_bad_static(self, primitive, run_config, *, stage="transfer", quick=False):
        result = original_collect(
            self, primitive, run_config, stage=stage, quick=quick
        )
        if primitive == "phi-np" and stage == "static":
            spike_count = result.spike_count.clone()
            spike_count[:, :, 0] = 2
            return replace(result, spike_count=spike_count)
        return result

    MockPrimitiveNoiseBackend.collect = one_bad_static
    try:
        screened_observations, _, failure = _collect_encoder_search_observations(
            SimpleNamespace(
                backend="mock",
                primitive="phi-np",
                quick=False,
                search_quick_codes=True,
            ),
            config,
        )
    finally:
        MockPrimitiveNoiseBackend.collect = original_collect
    assert failure is None
    assert any(
        observation.stage == "dynamic" for observation in screened_observations
    )

    observations = [
        backend.collect("phi-np", config, stage="static", quick=False),
        backend.collect("phi-np", config, stage="dynamic", quick=False),
    ]
    validations = [
        validate_primitive_observation(observation, config)
        for observation in observations
    ]
    reference = score_encoder_operating_point(
        observations, validations, config, primitive="phi-np"
    )
    assert reference["selection_eligible"]
    assert reference["held_out_validated"]
    assert reference["selection_objective_rt"] > 0
    assert reference["validation_objective_rt"] > 0
    selected = reference["calibration_selected_device"]
    calibration_devices = reference["primitive_scores"]["phi-np"][
        "calibration"
    ]["devices"]
    assert selected["calibration_rt"] == min(
        row["r_t"] for row in calibration_devices if row["complete"]
    )

    changed_values = observations[1].observed.clone()
    offsets = torch.tensor(
        [1.0, -1.0] * (config.validation_repeats // 2),
        dtype=changed_values.dtype,
    ).reshape(-1, 1, 1)
    changed_values[config.calibration_repeats :] += offsets * 1.0e-6
    changed_dynamic = replace(observations[1], observed=changed_values)
    changed_observations = [observations[0], changed_dynamic]
    changed_validations = [
        validations[0], validate_primitive_observation(changed_dynamic, config)
    ]
    changed = score_encoder_operating_point(
        changed_observations,
        changed_validations,
        config,
        primitive="phi-np",
    )
    assert math.isclose(
        changed["selection_objective_rt"],
        reference["selection_objective_rt"],
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )
    assert (
        changed["calibration_selected_device"]["physical_coordinate"]
        == selected["physical_coordinate"]
    )
    assert changed["validation_objective_rt"] > reference["validation_objective_rt"]

    unrelated_device = (selected["device"] + 1) % config.device_count
    unrelated_values = observations[1].observed.clone()
    unrelated_values[
        config.calibration_repeats :, :, unrelated_device
    ] += offsets.reshape(-1, 1) * 5.0e-6
    unrelated_dynamic = replace(observations[1], observed=unrelated_values)
    unrelated = score_encoder_operating_point(
        [observations[0], unrelated_dynamic],
        [validations[0], validate_primitive_observation(unrelated_dynamic, config)],
        config,
        primitive="phi-np",
    )
    assert unrelated["held_out_validated"]
    assert math.isclose(
        unrelated["validation_objective_rt"],
        reference["validation_objective_rt"],
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )

    screened_values = observations[0].observed.clone()
    screened_values[: config.calibration_repeats, 15, :] += 10.0e-6
    screened_static = replace(observations[0], observed=screened_values)
    screened_observations = [screened_static, observations[1]]
    screened_validations = [
        validate_primitive_observation(screened_static, config), validations[1]
    ]
    strict = score_encoder_operating_point(
        screened_observations,
        screened_validations,
        config,
        primitive="phi-np",
    )
    screening = score_encoder_operating_point(
        screened_observations,
        screened_validations,
        config,
        primitive="phi-np",
        screening=True,
    )
    assert not strict["selection_eligible"]
    assert screening["selection_eligible"]
    assert not screening["np_static"]["calibration_transfer"]["strict_eligible"]
    assert not any(screening["targets"].values())

    spike_only_config = replace(config, record_precharge_cadc=False)
    spike_only = backend.collect(
        "phi-np", spike_only_config, stage="dynamic", quick=True
    )
    assert spike_only.precharge_cadc is None

    prerequisite_score = {
        "primitive_scores": {
            "phi-np": {
                "selection_eligible": True,
                "held_out_validated": False,
            },
            "phi-nl": {
                "selection_eligible": True,
                "held_out_validated": True,
            },
        },
    }
    calibration_valid, held_out_valid = _primitive_summary_prerequisites(
        prerequisite_score, "phi-nl"
    )
    assert calibration_valid
    assert held_out_valid
    prerequisite_score["primitive_scores"]["phi-nl"][
        "selection_eligible"
    ] = False
    calibration_valid, _ = _primitive_summary_prerequisites(
        prerequisite_score, "phi-nl"
    )
    assert not calibration_valid


# @lat: [[hardware#Independent Primitive Noise Verification#Resumable operating point search]]
def verify_resumable_operating_point_search() -> None:
    config = PrimitiveNoiseConfig(
        repeats=8,
        calibration_repeats=4,
        device_count=4,
        physical_coordinates=default_primitive_coordinates(4),
    )
    assert parse_precharge_pair("2:31") == (2, 31)
    rejects(lambda: parse_precharge_pair("2x31"), (ValueError,))
    assert parse_exponential_pair("4:48") == (4, 48)
    rejects(lambda: parse_exponential_pair("4:0"), (ValueError,))
    parsed_current, parsed_stop = parse_current_stop_pair("512:40")
    assert parsed_current == 512
    torch.testing.assert_close(
        torch.tensor(parsed_stop), torch.tensor(40.0e-6), rtol=0.0, atol=1.0e-12
    )
    rejects(lambda: parse_current_stop_pair("512x40"), (ValueError,))
    candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(512, 1022),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
    )
    assert len(candidates) == 2
    assert candidates[0].apply(config).observation_time_s == 42.5e-6
    assert candidates[0].apply(config).exponential_input_fan_in == 8
    assert candidates[0].apply(config).exponential_input_weight == 63
    assert candidates[0].apply(config).membrane_capacitance_code is None
    assert candidates[0].apply(config).threshold_comparator_bias_code is None
    assert candidates[0].apply(config).excitatory_input_i_bias_tau_code is None
    assert candidates[0].apply(config).excitatory_input_i_bias_gm_code is None
    assert candidates[0].apply(config).synaptic_input_drop_bias_code is None
    assert candidates[0].apply(config).reset_current_code == 1022
    assert candidates[0].apply(config).reset_current_enable_multiplication is True
    assert candidates[0].apply(config).static_reset_release_s == 4.0e-6
    assert candidates[0].apply(config).dynamic_reset_release_s == 2.0e-6
    capacitance_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        membrane_capacitance_codes=(16, 63),
    )
    assert len(capacitance_candidates) == 2
    assert {
        item.apply(config).membrane_capacitance_code
        for item in capacitance_candidates
    } == {16, 63}
    comparator_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        threshold_comparator_bias_codes=(200, 800),
    )
    assert len(comparator_candidates) == 2
    assert {
        item.apply(config).threshold_comparator_bias_code
        for item in comparator_candidates
    } == {200, 800}
    synaptic_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        excitatory_input_i_bias_tau_codes=(64, 96),
        excitatory_input_i_bias_gm_codes=(480, 524),
        synaptic_input_drop_bias_codes=(300,),
    )
    assert len(synaptic_candidates) == 4
    assert {
        (
            item.apply(config).excitatory_input_i_bias_tau_code,
            item.apply(config).excitatory_input_i_bias_gm_code,
            item.apply(config).synaptic_input_drop_bias_code,
        )
        for item in synaptic_candidates
    } == {
        (tau_code, gain_code, 300)
        for tau_code in (64, 96)
        for gain_code in (480, 524)
    }
    reset_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        reset_current_codes=(512, 1022),
        static_reset_release_times_s=(3.5e-6, 4.5e-6),
        dynamic_reset_release_times_s=(1.5e-6, 2.5e-6),
    )
    assert len(reset_candidates) == 8
    assert {
        (
            item.apply(config).reset_current_code,
            item.apply(config).static_reset_release_s,
            item.apply(config).dynamic_reset_release_s,
        )
        for item in reset_candidates
    } == {
        (current, static_release, dynamic_release)
        for current in (512, 1022)
        for static_release in (3.5e-6, 4.5e-6)
        for dynamic_release in (1.5e-6, 2.5e-6)
    }
    reset_mode_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        reset_current_multiplication_modes=(True, False),
    )
    assert len(reset_mode_candidates) == 2
    assert {
        item.apply(config).reset_current_enable_multiplication
        for item in reset_mode_candidates
    } == {True, False}
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1022,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
            reset_current_multiplication_modes=(1,),
        ),
        (TypeError,),
    )
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1022,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
            membrane_capacitance_codes=(64,),
        ),
        (ValueError,),
    )
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1022,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
            excitatory_input_i_bias_tau_codes=(1023,),
        ),
        (ValueError,),
    )
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1022,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
            threshold_comparator_bias_codes=(1023,),
        ),
        (ValueError,),
    )
    nonlinear_candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(1022,),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
        exponential_pairs=((2, 48), (4, 32)),
    )
    assert len(nonlinear_candidates) == 2
    assert {
        (
            item.apply(config).exponential_input_fan_in,
            item.apply(config).exponential_input_weight,
        )
        for item in nonlinear_candidates
    } == {(2, 48), (4, 32)}
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1023,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
        ),
        (ValueError,),
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "search"
        arguments = build_parser().parse_args(
            [
                "--phase",
                "optimize",
                "--primitive",
                "phi-np",
                "--backend",
                "mock",
                "--quick",
                "--search-quick-codes",
                "--search-spike-times-only",
                "--device-count",
                "4",
                "--search-current-stop-pairs",
                "512:25",
                "1022:25",
                "--search-threshold-codes",
                "600",
                "--search-precharge-pairs",
                "1:63",
                "--output-dir",
                str(output),
            ]
        )
        run(arguments)
        selection_path = output / "selected_operating_point.json"
        first = selection_path.read_text(encoding="utf-8")
        assert (output / "search_manifest.json").is_file()
        assert (output / "operating_point_results.csv").is_file()
        search_manifest = json.loads(
            (output / "search_manifest.json").read_text(encoding="utf-8")
        )
        assert not search_manifest["base_config"]["record_precharge_cadc"]
        assert search_manifest["selection_contract"]["spike_times_only"]
        assert search_manifest["selection_contract"][
            "formal_confirmation_required"
        ]
        selected = json.loads(first)
        assert all(
            not any(item["targets"].values())
            for item in selected["best_by_primitive"].values()
        )
        assert len(list((output / "candidates").glob("*/candidate_result.json"))) == 2
        run(arguments)
        assert selection_path.read_text(encoding="utf-8") == first

        nonlinear_output = Path(temporary) / "nonlinear-search"
        nonlinear_arguments = build_parser().parse_args(
            [
                "--phase",
                "optimize",
                "--primitive",
                "phi-nl",
                "--backend",
                "mock",
                "--quick",
                "--device-count",
                "4",
                "--search-current-stop-pairs",
                "1022:25",
                "--search-threshold-codes",
                "600",
                "--search-precharge-pairs",
                "1:63",
                "--search-exponential-pairs",
                "2:48",
                "4:32",
                "--search-max-candidates",
                "2",
                "--output-dir",
                str(nonlinear_output),
            ]
        )
        run(nonlinear_arguments)
        nonlinear_manifest = json.loads(
            (nonlinear_output / "search_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert {
            (
                item["exponential_input_fan_in"],
                item["exponential_input_weight"],
            )
            for item in nonlinear_manifest["candidates"]
        } == {(2, 48), (4, 32)}
        assert len(
            list(
                (nonlinear_output / "candidates").glob(
                    "*/candidate_result.json"
                )
            )
        ) == 2

        invalid_arguments = build_parser().parse_args(
            [
                "--phase",
                "optimize",
                "--primitive",
                "phi-np",
                "--backend",
                "mock",
                "--search-spike-times-only",
                "--output-dir",
                str(Path(temporary) / "invalid-spike-only-search"),
            ]
        )
        rejects(lambda: run(invalid_arguments), (ValueError,))

    notebook = json.loads(
        (
            ROOT
            / "scripts"
            / "notebooks"
            / "ebrains_brainscales2_primitive_noise.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "RUN_OPERATING_POINT_SEARCH = False" in source
    assert "'--phase', 'optimize'" in source
    assert "'--search-spike-times-only'" in source
    assert "for start in range(0, 512, 64):" in source
    assert "'--device-count', 64" in source
    assert "'--physical-coordinates', *range(start, stop)" in source
    assert "for name, item in circuit_screen['best_by_primitive'].items()" in source
    assert "screened[name].append(item)" in source
    assert "item['calibration_rt']" in source
    assert "'--deadline', 60e-6" in source
    assert "'--pynn-process-repeats', 64" in source
    assert "'--pynn-chunk-repeats', 8" in source
    assert "'--physical-coordinates', best['physical_coordinate']" in source
    assert "selected_operating_point.json" in source
    assert notebook["metadata"]["language_info"]["version"] == "3.11"


# @lat: [[hardware#Independent Primitive Noise Verification#Artifact integrity]]
def verify_artifact_integrity() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    assert config.reset_code_minimum == 300
    assert config.reset_code_maximum == 900
    assert config.constant_current_code == 1022
    assert config.reset_current_code == 1022
    assert config.reset_current_enable_multiplication is True
    assert config.static_reset_release_s == 4.0e-6
    assert config.dynamic_reset_release_s == 2.0e-6
    assert config.membrane_capacitance_code is None
    assert config.threshold_comparator_bias_code is None
    assert config.excitatory_input_i_bias_tau_code is None
    assert config.excitatory_input_i_bias_gm_code is None
    assert config.synaptic_input_drop_bias_code is None
    rejects(
        lambda: PrimitiveNoiseConfig(membrane_capacitance_code=64),
        (ValueError,),
    )
    rejects(
        lambda: PrimitiveNoiseConfig(threshold_comparator_bias_code=1023),
        (ValueError,),
    )
    rejects(
        lambda: PrimitiveNoiseConfig(excitatory_input_i_bias_tau_code=1023),
        (ValueError,),
    )
    rejects(
        lambda: PrimitiveNoiseConfig(excitatory_input_i_bias_gm_code=-1),
        (ValueError,),
    )
    rejects(
        lambda: PrimitiveNoiseConfig(synaptic_input_drop_bias_code=1023),
        (ValueError,),
    )
    rejects(lambda: PrimitiveNoiseConfig(reset_current_code=1023), (ValueError,))
    rejects(
        lambda: PrimitiveNoiseConfig(
            reset_current_enable_multiplication=1  # type: ignore[arg-type]
        ),
        (TypeError,),
    )
    rejects(lambda: PrimitiveNoiseConfig(static_reset_release_s=5.0e-6), (ValueError,))
    rejects(lambda: PrimitiveNoiseConfig(static_reset_release_s=1.0e-6), (ValueError,))
    rejects(lambda: PrimitiveNoiseConfig(dynamic_reset_release_s=1.0e-6), (ValueError,))
    rejects(lambda: PrimitiveNoiseConfig(dynamic_reset_release_s=3.0e-6), (ValueError,))
    parser_defaults = build_parser().parse_args(["--output-dir", "unused"])
    assert parser_defaults.search_ramp_stop_us == (25.0,)
    assert parser_defaults.reset_current_enable_multiplication is True
    parsed = build_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--excitatory-input-i-bias-tau-code",
            "79",
            "--excitatory-input-i-bias-gm-code",
            "524",
            "--synaptic-input-drop-bias-code",
            "300",
        ]
    )
    parsed_config = make_config(parsed)
    assert parsed_config.excitatory_input_i_bias_tau_code == 79
    assert parsed_config.excitatory_input_i_bias_gm_code == 524
    assert parsed_config.synaptic_input_drop_bias_code == 300
    observation = MockPrimitiveNoiseBackend().collect(
        "psi-int", config, stage="transfer", quick=True
    )
    validation = validate_primitive_observation(observation, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[observation],
            validations=[validation],
        )
        loaded = load_primitive_observations(output, config)
        torch.testing.assert_close(
            loaded[0].observed, observation.observed, equal_nan=True
        )
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["replicas_are_pooled"] is False
        assert manifest["transformer_forward_included"] is False
        assert manifest["multiple_spike_handling"] == "first-spike-time"
        assert (output / "psi-int_transfer" / "moments.csv").is_file()
        chunk = next((output / "raw").glob("*.pt"))
        chunk.write_bytes(chunk.read_bytes() + b"changed")
        rejects(lambda: load_primitive_observations(output, config), (ValueError,))

    ed_observation = MockPrimitiveNoiseBackend().collect(
        "psi-ed", config, stage="transfer", quick=True
    )
    ed_validation = validate_primitive_observation(ed_observation, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[ed_observation],
            validations=[ed_validation],
        )
        loaded = load_primitive_observations(output, config)[0]
        torch.testing.assert_close(
            loaded.first_spike_time_s,
            ed_observation.first_spike_time_s,
            equal_nan=True,
        )
        moments = (output / "psi-ed_transfer" / "moments.csv").read_text()
        assert "mean" in moments and "variance" in moments

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calibration_path = root / "hagen.pbin"
        calibration_path.write_bytes(b"pinned calibration")
        hardware_config = replace(config, hagen_calibration_path=calibration_path)
        bad_checksum = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-a"],
                "calibration_sha256": "wrong",
            },
        )
        rejects(
            lambda: write_primitive_noise_artifacts(
                root / "bad-checksum",
                config=hardware_config,
                observations=[bad_checksum],
                validations=[validation],
            ),
            (ValueError,),
        )
        checksum = calibration_sha256(calibration_path)
        first = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-a"],
                "calibration_sha256": checksum,
            },
        )
        second = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-b"],
                "calibration_sha256": checksum,
            },
        )
        rejects(
            lambda: write_primitive_noise_artifacts(
                root / "bad-chip",
                config=hardware_config,
                observations=[first, second],
                validations=[validation, validation],
            ),
            (ValueError,),
        )

    rejects(
        lambda: PrimitiveNoiseConfig(
            device_count=2,
            physical_coordinates=(0, 0),
        )
    )
    rejects(
        lambda: PrimitiveNoiseConfig(
            device_count=2,
            physical_coordinates=(0, 512),
        )
    )


def main() -> None:
    verify_configured_hardware_endpoint_reuse()
    verify_worker_timeout_cleanup()
    verify_transient_worker_error_classification()
    verify_pynn_worker_attempt_budget()
    verify_pynn_worker_cache_identity()
    verify_repeated_acquisition_timeout_abort()
    verify_synthetic_transfer_recovery()
    verify_held_out_validation_isolation()
    verify_temporal_and_fixed_pattern_separation()
    verify_validated_np_placement()
    verify_miss_and_saturation_semantics()
    verify_rank_monotonicity_gate()
    verify_segmented_pynn_recording_decode()
    verify_dynamic_recording_selection()
    verify_segmented_dense_recording_decode()
    verify_sparse_precharge_evidence()
    verify_installed_component_resolution()
    verify_correlation_recording_decode()
    verify_correlation_worker_boundary()
    verify_nonlinear_drive_configuration()
    verify_static_reset_separation()
    verify_hagen_output_qualification()
    verify_exponential_response_observation_time()
    verify_insufficient_fit_preservation()
    verify_np_stage_gate()
    verify_encoder_operating_point_score()
    verify_resumable_operating_point_search()
    verify_artifact_integrity()
    print("BrainScaleS-2 primitive-noise checks passed")


if __name__ == "__main__":
    main()
