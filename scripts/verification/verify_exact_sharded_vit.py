#!/usr/bin/env python3
"""Verify exact CUDA stream sharding and shard artifact reconstruction."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_exact_sharded_vit import (
    build_rng_preflight_contract,
    canonical_json_sha256,
    checked_batch_offset,
    decode_predictions,
    exact_batch_shard_bounds,
    load_rng_contract,
    merge_shard_results,
    noise_profile_fractions,
    forward_termination_signals,
    RunnerInterrupted,
    _signal_process_groups,
    _spawn_managed,
    _terminate,
    _wait_and_unregister,
    validate_local_checkpoint,
    verify_rng_batch_trace,
    write_shard_result,
)
from scripts.runtime import identity
from utils.transforms.noise import (
    begin_gaussian_rng_trace,
    end_gaussian_rng_trace,
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.potential_to_spike import neg_linear_transform, neg_log_transform
from utils.transforms.types import PotentialBounds


BATCH_SIZE = 32
SEED = 1701
RUN_IDENTITY = {
    "model": "synthetic-vit",
    "source_commit": "verification",
    "checkpoint_sha256": "1" * 64,
    "loaded_model_state_sha256": "2" * 64,
    "activation": "gelu",
    "batch_size": BATCH_SIZE,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
}


def normalized_stats() -> dict[str, dict[str, int | float]]:
    snapshot = get_gaussian_noise_stats()
    for counts in snapshot.values():
        if counts["deadline_ulp_min"] == float("inf"):
            counts["deadline_ulp_min"] = 0.0
    return snapshot


def configure(profile: str) -> tuple[float, float]:
    linear, logarithmic = noise_profile_fractions(
        profile,
        linear_std_frac=0.125,
        log_std_frac=0.25,
    )
    set_gaussian_time_noise(
        enabled=True,
        time_std=0.0,
        linear_time_std=linear,
        log_time_std=logarithmic,
        seed=SEED,
        device="cuda",
    )
    return linear, logarithmic


def synthetic_batch(
    global_sample_start: int,
    sample_count: int,
) -> tuple[list[int], list[int], tuple[int, int, tuple[dict[str, Any], ...]]]:
    ordinal = torch.arange(
        global_sample_start,
        global_sample_start + sample_count,
        device="cuda",
        dtype=torch.float64,
    )
    linear_input = (ordinal.remainder(97.0) / 97.0).clamp(0.0, 1.0)
    log_input = 1.0 + ordinal.remainder(89.0) / 89.0
    trace_start = begin_gaussian_rng_trace()
    linear = neg_linear_transform(
        linear_input,
        PotentialBounds(0.0, 1.0),
        return_spike_sample=True,
        noise_site="synthetic.phi_np",
    )
    logarithmic = neg_log_transform(
        log_input,
        PotentialBounds(1.0, 2.0),
        return_spike_sample=True,
        noise_site="synthetic.phi_nl",
    )
    trace_stop, trace = end_gaussian_rng_trace()
    predictions = ((linear.time + logarithmic.time) > 0.75).to(torch.int64).cpu().tolist()
    labels = (ordinal.to(torch.int64).remainder(2)).cpu().tolist()
    return predictions, labels, (trace_start, trace_stop, trace)


def serial_result(population: int, profile: str) -> dict[str, Any]:
    configure(profile)
    predictions: list[int] = []
    labels: list[int] = []
    traces = []
    for batch_start in range(0, population, BATCH_SIZE):
        count = min(BATCH_SIZE, population - batch_start)
        batch_predictions, batch_labels, trace = synthetic_batch(batch_start, count)
        predictions.extend(batch_predictions)
        labels.extend(batch_labels)
        traces.append(trace)
    return {
        "predictions": predictions,
        "labels": labels,
        "correct": sum(a == b for a, b in zip(predictions, labels, strict=True)),
        "stats": normalized_stats(),
        "contract": build_rng_preflight_contract(
            run_identity=RUN_IDENTITY,
            batch_size=BATCH_SIZE,
            batches=traces[:2],
        ),
    }


def sharded_result(
    root: Path,
    population: int,
    profile: str,
    serial: dict[str, Any],
) -> dict[str, Any]:
    paths = []
    for shard_index in range(2):
        sample_start, sample_stop, batch_start, batch_stop = exact_batch_shard_bounds(
            population, BATCH_SIZE, 2, shard_index
        )
        configure(profile)
        generator = get_gaussian_time_noise().generator
        assert isinstance(generator, torch.Generator)
        start_offset = checked_batch_offset(
            batch_start, serial["contract"]["batch_stride"]
        )
        generator.set_offset(start_offset)
        predictions: list[int] = []
        labels: list[int] = []
        for global_batch in range(batch_start, batch_stop):
            global_start = global_batch * BATCH_SIZE
            count = min(BATCH_SIZE, population - global_start)
            batch_predictions, batch_labels, trace = synthetic_batch(global_start, count)
            verify_rng_batch_trace(
                serial["contract"],
                start_offset=trace[0],
                stop_offset=trace[1],
                trace=trace[2],
                sample_count=count,
            )
            predictions.extend(batch_predictions)
            labels.extend(batch_labels)
        end_offset = checked_batch_offset(
            batch_stop, serial["contract"]["batch_stride"]
        )
        assert generator.get_offset() == end_offset
        path = root / f"{population}-{profile}-shard-{shard_index}.json"
        contract = {
            key: value for key, value in serial["contract"].items() if key != "trace"
        }
        contract.update(
            trace_sha256=canonical_json_sha256(serial["contract"]["trace"]),
            global_batch_start=batch_start,
            global_batch_stop=batch_stop,
            start_offset=start_offset,
            end_offset=end_offset,
            observed_batches=batch_stop - batch_start,
        )
        correct = sum(a == b for a, b in zip(predictions, labels, strict=True))
        write_shard_result(
            path,
            predictions=predictions,
            interval={
                "shard_index": shard_index,
                "shard_count": 2,
                "sample_start": sample_start,
                "sample_stop": sample_stop,
                "batch_start": batch_start,
                "batch_stop": batch_stop,
                "population": population,
                "batch_size": BATCH_SIZE,
            },
            counts={"correct": correct, "evaluated": sample_stop - sample_start},
            gaussian_stats=normalized_stats(),
            clamp_stats={
                "synthetic/clamp": {
                    "values": sample_stop - sample_start,
                    "underflows": shard_index,
                    "overflows": 0,
                }
            },
            calibration_clamp_stats={},
            run_identity=RUN_IDENTITY,
            rng_contract=contract,
        )
        paths.append(path)
    return merge_shard_results(paths, root / f"{population}-{profile}-aggregate.json")


# @lat: [[evaluation#Evaluation and Verification#Exact ViT Timing Noise Shards#Synthetic Verification]]
def verify_serial_and_merged_cuda_streams() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact random generator verification")
    expected_ranges = {
        512: ((0, 256, 0, 8), (256, 512, 8, 16)),
        5000: ((0, 2528, 0, 79), (2528, 5000, 79, 157)),
    }
    for population, ranges in expected_ranges.items():
        assert tuple(
            exact_batch_shard_bounds(population, BATCH_SIZE, 2, index)
            for index in range(2)
        ) == ranges

    with tempfile.TemporaryDirectory(prefix="verify-exact-vit-") as temporary:
        root = Path(temporary)
        for population in (512, 5000):
            for profile in ("np", "nl", "joint"):
                serial = serial_result(population, profile)
                aggregate = sharded_result(root, population, profile, serial)
                merged = (root / aggregate["predictions"]["path"]).read_bytes()
                assert decode_predictions(merged) == tuple(serial["predictions"])
                assert aggregate["counts"]["correct"] == serial["correct"]
                assert aggregate["counts"]["evaluated"] == population
                assert aggregate["gaussian_stats"] == serial["stats"]

        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            linear_time_std=0.0,
            log_time_std=0.0,
            seed=SEED,
            device="cuda",
        )
        _, _, first = synthetic_batch(0, BATCH_SIZE)
        _, _, second = synthetic_batch(BATCH_SIZE, BATCH_SIZE)
        zero_contract = build_rng_preflight_contract(
            run_identity=RUN_IDENTITY,
            batch_size=BATCH_SIZE,
            batches=(first, second),
        )
        assert zero_contract["batch_stride"] == 0
        assert get_gaussian_time_noise().generator.get_offset() == 0


def verify_merge_rejections_and_offset_guards() -> None:
    for value in (-4, 2**64, 6):
        try:
            checked_batch_offset(1, value)
        except (ValueError, OverflowError):
            pass
        else:
            raise AssertionError(f"invalid CUDA offset stride accepted: {value}")
    try:
        checked_batch_offset(2**63, 4)
    except OverflowError:
        pass
    else:
        raise AssertionError("CUDA offset multiplication overflow was accepted")

    configure("joint")
    _, _, first = synthetic_batch(0, BATCH_SIZE)
    _, _, second = synthetic_batch(BATCH_SIZE, BATCH_SIZE)
    mismatched_stride = copy.deepcopy(second)
    mismatched_stride = (
        mismatched_stride[0],
        mismatched_stride[1] + 4,
        mismatched_stride[2],
    )
    for name, batches in (
        ("stride", (first, mismatched_stride)),
        (
            "trace",
            (
                first,
                (
                    second[0],
                    second[1],
                    (
                        {**second[2][0], "site": "synthetic.unexpected"},
                        *second[2][1:],
                    ),
                ),
            ),
        ),
    ):
        try:
            build_rng_preflight_contract(
                run_identity=RUN_IDENTITY,
                batch_size=BATCH_SIZE,
                batches=batches,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"preflight accepted a mismatched {name}")

    contract = build_rng_preflight_contract(
        run_identity=RUN_IDENTITY,
        batch_size=BATCH_SIZE,
        batches=(first, second),
    )
    try:
        verify_rng_batch_trace(
            contract,
            start_offset=second[0],
            stop_offset=second[1] + 4,
            trace=second[2],
            sample_count=BATCH_SIZE,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a shard batch stride mismatch was accepted")

    with tempfile.TemporaryDirectory(prefix="verify-exact-vit-rejections-") as temporary:
        root = Path(temporary)
        serial = serial_result(512, "joint")
        contract_path = root / "preflight-contract.json"
        contract_path.write_text(json.dumps(serial["contract"]))
        changed_identity = {
            **RUN_IDENTITY,
            "loaded_model_state_sha256": "3" * 64,
        }
        try:
            load_rng_contract(contract_path, run_identity=changed_identity)
        except ValueError:
            pass
        else:
            raise AssertionError("preflight accepted a changed loaded model state")
        base = root / "base"
        base.mkdir()
        sharded_result(base, 512, "joint", serial)
        originals = [base / f"512-joint-shard-{index}.json" for index in range(2)]
        first_payload = json.loads(originals[0].read_text())
        for name, statistic_key in (
            ("Gaussian", "gaussian_stats"),
            ("clamp", "clamp_stats"),
        ):
            impossible_stats = copy.deepcopy(first_payload[statistic_key])
            site_counts = next(iter(impossible_stats.values()))
            if statistic_key == "gaussian_stats":
                site_counts.update(events=0, misses=1)
            else:
                site_counts.update(values=0, underflows=1)
            arguments = {
                "gaussian_stats": first_payload["gaussian_stats"],
                "clamp_stats": first_payload["clamp_stats"],
                "calibration_clamp_stats": first_payload[
                    "calibration_clamp_stats"
                ],
            }
            arguments[statistic_key] = impossible_stats
            rejected_path = root / f"write-impossible-{name}.json"
            try:
                write_shard_result(
                    rejected_path,
                    predictions=range(first_payload["predictions"]["count"]),
                    interval=first_payload["interval"],
                    counts=first_payload["counts"],
                    run_identity=first_payload["run_identity"],
                    rng_contract=first_payload["rng_contract"],
                    **arguments,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"shard writer accepted impossible {name} counters")
            assert not rejected_path.exists()
            assert not rejected_path.with_suffix(".predictions.int64").exists()

        mutations = (
            ("gap", lambda payload: payload["interval"].update(sample_start=257)),
            ("overlap", lambda payload: payload["interval"].update(sample_start=255)),
            ("identity", lambda payload: payload["run_identity"].update(model="other")),
            (
                "activation",
                lambda payload: payload["run_identity"].update(activation="relu"),
            ),
            (
                "loaded-state",
                lambda payload: payload["run_identity"].update(
                    loaded_model_state_sha256="3" * 64
                ),
            ),
            ("rng", lambda payload: payload["rng_contract"].update(batch_stride=12)),
            (
                "gaussian-subset",
                lambda payload: payload["gaussian_stats"].update(
                    {next(iter(payload["gaussian_stats"])): {"events": 1}}
                ),
            ),
            (
                "gaussian-clamp-schema",
                lambda payload: payload["gaussian_stats"].update(
                    {
                        next(iter(payload["gaussian_stats"])): {
                            "values": 1,
                            "underflows": 0,
                            "overflows": 0,
                        }
                    }
                ),
            ),
            (
                "clamp-gaussian-schema",
                lambda payload: payload["clamp_stats"].update(
                    {
                        next(iter(payload["clamp_stats"])): {
                            "events": 1,
                            "misses": 0,
                            "deadline_events": 0,
                            "deadline_ulp_min": 0.0,
                            "deadline_ulp_max": 0.0,
                            "outputs": 1,
                            "output_underflows": 0,
                            "output_overflows": 0,
                        }
                    }
                ),
            ),
            (
                "gaussian-site-omitted",
                lambda payload: payload["gaussian_stats"].pop(
                    next(iter(payload["gaussian_stats"]))
                ),
            ),
            (
                "clamp-site-omitted",
                lambda payload: payload["clamp_stats"].pop(
                    next(iter(payload["clamp_stats"]))
                ),
            ),
            (
                "gaussian-misses-exceed-events",
                lambda payload: next(
                    iter(payload["gaussian_stats"].values())
                ).update(events=0, misses=1),
            ),
            (
                "gaussian-deadline-events-exceed-events",
                lambda payload: next(
                    iter(payload["gaussian_stats"].values())
                ).update(events=0, deadline_events=1),
            ),
            (
                "gaussian-output-underflows-exceed-outputs",
                lambda payload: next(
                    iter(payload["gaussian_stats"].values())
                ).update(outputs=0, output_underflows=1),
            ),
            (
                "gaussian-output-overflows-exceed-outputs",
                lambda payload: next(
                    iter(payload["gaussian_stats"].values())
                ).update(outputs=0, output_overflows=1),
            ),
            (
                "gaussian-output-saturation-counts-exceed-outputs",
                lambda payload: next(
                    iter(payload["gaussian_stats"].values())
                ).update(outputs=1, output_underflows=1, output_overflows=1),
            ),
            (
                "clamp-underflows-exceed-values",
                lambda payload: next(iter(payload["clamp_stats"].values())).update(
                    values=0, underflows=1
                ),
            ),
            (
                "clamp-overflows-exceed-values",
                lambda payload: next(iter(payload["clamp_stats"].values())).update(
                    values=0, overflows=1
                ),
            ),
            (
                "clamp-saturation-counts-exceed-values",
                lambda payload: next(iter(payload["clamp_stats"].values())).update(
                    values=1, underflows=1, overflows=1
                ),
            ),
        )
        for name, mutate in mutations:
            case = root / name
            case.mkdir()
            paths = []
            for index, original in enumerate(originals):
                payload = json.loads(original.read_text())
                raw_name = payload["predictions"]["path"]
                (case / raw_name).write_bytes((base / raw_name).read_bytes())
                if index == 1:
                    mutate(payload)
                path = case / original.name
                path.write_text(json.dumps(payload))
                paths.append(path)
            try:
                merge_shard_results(paths, case / "aggregate.json")
            except ValueError:
                pass
            else:
                raise AssertionError(f"merge accepted {name}")

        malformed = (
            ("interval", lambda payload, value: payload["interval"].update(sample_start=value)),
            ("count", lambda payload, value: payload["counts"].update(correct=value)),
            (
                "statistic",
                lambda payload, value: next(iter(payload["gaussian_stats"].values())).update(
                    events=value
                ),
            ),
        )
        for field, mutate in malformed:
            for value in (True, 1.0, "1", -1):
                case = root / f"malformed-{field}-{type(value).__name__}-{value}"
                case.mkdir()
                paths = []
                for index, original in enumerate(originals):
                    payload = json.loads(original.read_text())
                    raw_name = payload["predictions"]["path"]
                    (case / raw_name).write_bytes((base / raw_name).read_bytes())
                    if index == 1:
                        mutate(payload, value)
                    path = case / original.name
                    path.write_text(json.dumps(payload))
                    paths.append(path)
                try:
                    merge_shard_results(paths, case / "aggregate.json")
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        f"merge accepted malformed {field}: {value!r}"
                    )


def verify_model_provenance_and_process_cleanup() -> None:
    first = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    first.register_buffer("scalar_fixture", torch.tensor(1.0))
    second = copy.deepcopy(first)
    first_digest = identity.model_state_sha256(first)
    assert first_digest == identity.model_state_sha256(second)
    with torch.no_grad():
        second[0].weight[0, 0].add_(1.0)
    assert first_digest != identity.model_state_sha256(second)

    with tempfile.TemporaryDirectory(prefix="verify-exact-provenance-") as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"loaded model fixture")
        digest = identity.artifact_identity(checkpoint)["aggregate_sha256"]
        arguments = [
            "--model_id",
            str(checkpoint),
            "--checkpoint-sha256",
            digest,
        ]
        assert validate_local_checkpoint(arguments)["aggregate_sha256"] == digest
        (checkpoint / "weights.bin").write_bytes(b"mutated model fixture")
        try:
            validate_local_checkpoint(arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("mutated local checkpoint was accepted")

        rejected_output = root / "too-short"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/experiments/run_exact_sharded_vit.py"),
                "--output-dir",
                str(rejected_output),
                "--prefix-samples",
                "63",
                "--batch-size",
                "32",
                "--noise-profile",
                "np",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2
        assert not rejected_output.exists()

    graceful_children: list[subprocess.Popen[Any]] = []
    graceful = _spawn_managed(
        graceful_children,
        [
            sys.executable,
            "-c",
            (
                "import signal,sys,time; "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                "blocked=next(line.split()[1] for line in "
                "open('/proc/self/status') if line.startswith('SigBlk:')); "
                "print(blocked, flush=True); time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert graceful.stdout is not None
        blocked = int(graceful.stdout.readline().strip(), 16)
        assert not blocked & (1 << (signal.SIGINT - 1))
        assert not blocked & (1 << (signal.SIGTERM - 1))
        _signal_process_groups(graceful_children, signal.SIGTERM)
        assert _wait_and_unregister(graceful_children, graceful) == 0
        assert graceful_children == []
        with patch(
            "scripts.experiments.run_exact_sharded_vit.os.killpg"
        ) as killpg:
            _signal_process_groups([graceful], signal.SIGTERM)
            killpg.assert_not_called()
    finally:
        if graceful.poll() is None:
            _terminate(graceful_children, grace_seconds=0.05)
        if graceful.stdout is not None:
            graceful.stdout.close()

    ignored_children: list[subprocess.Popen[Any]] = []
    ignored = _spawn_managed(
        ignored_children,
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert ignored.stdout is not None
        assert ignored.stdout.readline().strip() == "ready"
        assert os.getpgid(ignored.pid) == ignored.pid
        assert ignored_children == [ignored]
        _terminate(ignored_children, grace_seconds=0.05)
    finally:
        if ignored.poll() is None:
            _terminate(ignored_children, grace_seconds=0.05)
        if ignored.stdout is not None:
            ignored.stdout.close()
    assert ignored.returncode == -signal.SIGKILL
    try:
        os.killpg(ignored.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("terminated evaluator process group still exists")

    children: list[subprocess.Popen[Any]] = []
    forwarded = _spawn_managed(
        children,
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    prior = signal.getsignal(signal.SIGINT)
    try:
        with forward_termination_signals(children):
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler) and handler is not prior
            try:
                handler(signal.SIGINT, None)
            except RunnerInterrupted as error:
                assert error.signum == signal.SIGINT
            else:
                raise AssertionError(
                    "runner signal handler did not interrupt supervision"
                )
    finally:
        _terminate(children, grace_seconds=0.05)
    assert forwarded.returncode is not None
    assert signal.getsignal(signal.SIGINT) is prior


def main() -> None:
    verify_serial_and_merged_cuda_streams()
    print("PASS verify_serial_and_merged_cuda_streams")
    verify_merge_rejections_and_offset_guards()
    print("PASS verify_merge_rejections_and_offset_guards")
    verify_model_provenance_and_process_cleanup()
    print("PASS verify_model_provenance_and_process_cleanup")
    print("Exact sharded ViT verification passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        set_gaussian_time_noise(enabled=False)
