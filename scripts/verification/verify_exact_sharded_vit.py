#!/usr/bin/env python3
"""Verify exact CUDA stream sharding and shard artifact reconstruction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_exact_sharded_vit import (
    build_rng_preflight_contract,
    canonical_json_sha256,
    checked_batch_offset,
    decode_predictions,
    exact_batch_shard_bounds,
    merge_shard_results,
    noise_profile_fractions,
    verify_rng_batch_trace,
    write_shard_result,
)
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
        base = root / "base"
        base.mkdir()
        sharded_result(base, 512, "joint", serial)
        originals = [base / f"512-joint-shard-{index}.json" for index in range(2)]
        mutations = (
            ("gap", lambda payload: payload["interval"].update(sample_start=257)),
            ("overlap", lambda payload: payload["interval"].update(sample_start=255)),
            ("identity", lambda payload: payload["run_identity"].update(model="other")),
            ("rng", lambda payload: payload["rng_contract"].update(batch_stride=12)),
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


def main() -> None:
    verify_serial_and_merged_cuda_streams()
    print("PASS verify_serial_and_merged_cuda_streams")
    verify_merge_rejections_and_offset_guards()
    print("PASS verify_merge_rejections_and_offset_guards")
    print("Exact sharded ViT verification passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        set_gaussian_time_noise(enabled=False)
