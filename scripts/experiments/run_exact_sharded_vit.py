#!/usr/bin/env python3
"""Run and merge an exact two-process ViT timing-noise evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu


SCHEMA_VERSION = 1
SHARD_COUNT = 2
DEFAULT_LINEAR_STD_FRAC = 0.03785
DEFAULT_LOG_STD_FRAC = 0.02712
SUM_FIELDS = frozenset(
    {
        "events",
        "misses",
        "deadline_events",
        "outputs",
        "output_underflows",
        "output_overflows",
        "values",
        "underflows",
        "overflows",
    }
)
MIN_FIELDS = frozenset({"deadline_ulp_min"})
MAX_FIELDS = frozenset({"deadline_ulp_max"})
MAX_CUDA_OFFSET = 2**64 - 4
MANAGED_EVALUATOR_OPTIONS = frozenset(
    {
        "--device",
        "--model_backend",
        "--experiment_name",
        "--batch_size",
        "--evaluation-prefix-samples",
        "--evaluation-shard-count",
        "--evaluation-shard-index",
        "--shard-result-path",
        "--gaussian-rng-preflight-path",
        "--gaussian-rng-contract-path",
        "--gaussian-rng-contract-timeout-seconds",
        "--gaussian-time-noise",
        "--no-gaussian-time-noise",
        "--time-noise-std-frac",
        "--linear-time-noise-std-frac",
        "--log-time-noise-std-frac",
        "--time-noise-seed",
        "--time-noise-deadline-margin-std",
        "--source-commit",
        "--tensorboard",
        "--no-tensorboard",
        "--report-clamp-stats",
        "--no-report-clamp-stats",
    }
)


def exact_batch_shard_bounds(
    population: int,
    batch_size: int,
    shard_count: int,
    shard_index: int,
) -> tuple[int, int, int, int]:
    """Return sample and batch bounds without splitting a global batch."""

    values = (population, batch_size, shard_count, shard_index)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("population, batch size, shard count, and index must be integers")
    if population <= 0 or batch_size <= 0 or shard_count <= 0:
        raise ValueError("population, batch size, and shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be inside shard count")

    global_batches = math.ceil(population / batch_size)
    if shard_count > global_batches:
        raise ValueError("each exact shard must contain at least one global batch")
    base, remainder = divmod(global_batches, shard_count)
    batch_start = shard_index * base + min(shard_index, remainder)
    batch_stop = batch_start + base + int(shard_index < remainder)
    sample_start = min(batch_start * batch_size, population)
    sample_stop = min(batch_stop * batch_size, population)
    return sample_start, sample_stop, batch_start, batch_stop


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value with stable key and whitespace rules."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_cuda_offset(offset: int, *, name: str) -> int:
    """Reject offsets that cannot be represented by CUDA's opaque Philox state."""

    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"{name} must be an integer")
    if offset < 0 or offset > MAX_CUDA_OFFSET:
        raise ValueError(f"{name} is outside the supported CUDA offset range")
    if offset % 4:
        raise ValueError(f"{name} must be aligned to four Philox units")
    return offset


def checked_batch_offset(batch_index: int, batch_stride: int) -> int:
    """Multiply a global batch index and stride without unsigned wraparound."""

    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("global batch index must be an integer")
    if batch_index < 0:
        raise ValueError("global batch index must be non-negative")
    validate_cuda_offset(batch_stride, name="batch stride")
    if batch_stride and batch_index > MAX_CUDA_OFFSET // batch_stride:
        raise OverflowError("global batch offset exceeds the CUDA generator range")
    return validate_cuda_offset(
        batch_index * batch_stride, name="global batch offset"
    )


def _relative_trace(
    trace: Iterable[Mapping[str, Any]],
    *,
    batch_start_offset: int,
    batch_stop_offset: int,
) -> list[dict[str, Any]]:
    relative: list[dict[str, Any]] = []
    previous = batch_start_offset
    for call in trace:
        before = validate_cuda_offset(int(call["before_offset"]), name="call before offset")
        after = validate_cuda_offset(int(call["after_offset"]), name="call after offset")
        if before != previous or after < before:
            raise ValueError("Gaussian call offsets are not contiguous and monotone")
        relative.append(
            {
                "site": str(call["site"]),
                "encoding": str(call["encoding"]),
                "shape": [int(value) for value in call["shape"]],
                "dtype": str(call["dtype"]),
                "sampled": bool(call["sampled"]),
                "before_offset": before - batch_start_offset,
                "after_offset": after - batch_start_offset,
            }
        )
        previous = after
    if previous != batch_stop_offset:
        raise ValueError("Gaussian call trace does not cover the batch offset interval")
    return relative


def build_rng_preflight_contract(
    *,
    run_identity: Mapping[str, Any],
    batch_size: int,
    batches: Iterable[tuple[int, int, Iterable[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Validate two distinct full batches before allowing offset extrapolation."""

    observed = list(batches)
    if len(observed) != 2:
        raise ValueError("random generator preflight requires exactly two batches")
    normalized: list[list[dict[str, Any]]] = []
    strides: list[int] = []
    for start, stop, trace in observed:
        validate_cuda_offset(start, name="batch start offset")
        validate_cuda_offset(stop, name="batch stop offset")
        if stop < start:
            raise ValueError("batch random generator offsets are reversed")
        normalized.append(
            _relative_trace(
                trace,
                batch_start_offset=start,
                batch_stop_offset=stop,
            )
        )
        strides.append(stop - start)
    if observed[1][0] != observed[0][1]:
        raise ValueError("preflight batch offsets are not contiguous")
    if strides[0] != strides[1]:
        raise ValueError("full batch random generator strides differ")
    stride = validate_cuda_offset(strides[0], name="batch stride")
    if normalized[0] != normalized[1]:
        raise ValueError("full batch ordered Gaussian call traces differ")
    if not normalized[0]:
        raise ValueError("preflight observed no Gaussian encoder calls")
    any_sampled = any(call["sampled"] for call in normalized[0])
    if any_sampled != (stride > 0):
        raise ValueError("Gaussian sampling coverage and generator stride disagree")
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "torch_cuda_generator_philox_offset",
        "identity_sha256": canonical_json_sha256(run_identity),
        "batch_size": batch_size,
        "batch_stride": stride,
        "preflight_batches": 2,
        "trace": normalized[0],
    }


def verify_rng_batch_trace(
    contract: Mapping[str, Any],
    *,
    start_offset: int,
    stop_offset: int,
    trace: Iterable[Mapping[str, Any]],
    sample_count: int,
) -> None:
    """Require one actual shard batch to satisfy the preflight stride and trace."""

    expected_stride = validate_cuda_offset(
        int(contract["batch_stride"]), name="contract batch stride"
    )
    validate_cuda_offset(start_offset, name="batch start offset")
    validate_cuda_offset(stop_offset, name="batch stop offset")
    if stop_offset - start_offset != expected_stride:
        raise RuntimeError(
            "Gaussian random generator batch stride differs from preflight"
        )
    actual = _relative_trace(
        trace,
        batch_start_offset=start_offset,
        batch_stop_offset=stop_offset,
    )
    expected = contract["trace"]
    if sample_count == int(contract["batch_size"]):
        if actual != expected:
            raise RuntimeError("full batch Gaussian call trace differs from preflight")
        return
    if not 0 < sample_count < int(contract["batch_size"]):
        raise ValueError("partial batch size is invalid")
    if len(actual) != len(expected):
        raise RuntimeError("partial batch Gaussian call count differs from preflight")
    for expected_call, actual_call in zip(expected, actual, strict=True):
        for field in (
            "site",
            "encoding",
            "dtype",
            "sampled",
            "before_offset",
            "after_offset",
        ):
            if actual_call[field] != expected_call[field]:
                raise RuntimeError(
                    f"partial batch Gaussian call field differs: {field}"
                )
        expected_shape = expected_call["shape"]
        actual_shape = actual_call["shape"]
        if expected_shape == actual_shape:
            continue
        if (
            not expected_shape
            or len(expected_shape) != len(actual_shape)
            or expected_shape[0] != int(contract["batch_size"])
            or actual_shape[0] != sample_count
            or expected_shape[1:] != actual_shape[1:]
        ):
            raise RuntimeError("partial batch Gaussian call shape differs unexpectedly")


def load_rng_contract(
    path: Path,
    *,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a completed preflight contract and bind it to one run identity."""

    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("algorithm") != "torch_cuda_generator_philox_offset"
        or payload.get("identity_sha256") != canonical_json_sha256(run_identity)
    ):
        raise ValueError("Gaussian random generator preflight identity differs")
    validate_cuda_offset(int(payload["batch_stride"]), name="contract batch stride")
    if payload.get("preflight_batches") != 2 or not isinstance(payload.get("trace"), list):
        raise ValueError("Gaussian random generator preflight is incomplete")
    return payload


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prediction_bytes(predictions: Iterable[int]) -> tuple[bytes, int]:
    """Serialize predictions as raw signed little endian 64-bit integers."""

    values = tuple(predictions)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("predictions must be integers")
    return b"".join(struct.pack("<q", value) for value in values), len(values)


def decode_predictions(data: bytes) -> tuple[int, ...]:
    """Decode the raw prediction representation after validating its width."""

    if len(data) % 8:
        raise ValueError("raw prediction bytes are not aligned to int64")
    return tuple(item[0] for item in struct.iter_unpack("<q", data))


def write_shard_result(
    result_path: Path,
    *,
    predictions: Iterable[int],
    interval: Mapping[str, int],
    counts: Mapping[str, int],
    gaussian_stats: Mapping[str, Mapping[str, int | float]],
    clamp_stats: Mapping[str, Mapping[str, int]],
    calibration_clamp_stats: Mapping[str, Mapping[str, int]],
    run_identity: Mapping[str, Any],
    rng_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable shard metadata file and its raw predictions."""

    if result_path.exists():
        raise FileExistsError(result_path)
    raw, prediction_count = prediction_bytes(predictions)
    sample_count = int(interval["sample_stop"]) - int(interval["sample_start"])
    if sample_count <= 0 or prediction_count != sample_count:
        raise ValueError("prediction count must equal the nonempty shard interval")
    if int(counts.get("evaluated", -1)) != sample_count:
        raise ValueError("evaluated count must equal the shard interval")
    if not 0 <= int(counts.get("correct", -1)) <= sample_count:
        raise ValueError("correct count must lie inside the evaluated count")

    predictions_path = result_path.with_suffix(".predictions.int64")
    if predictions_path.exists():
        raise FileExistsError(predictions_path)
    _atomic_bytes(predictions_path, raw)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": int(interval["shard_index"]),
        "shard_count": int(interval["shard_count"]),
        "interval": dict(interval),
        "counts": dict(counts),
        "predictions": {
            "path": predictions_path.name,
            "dtype": "int64",
            "byte_order": "little",
            "count": prediction_count,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "gaussian_stats": {
            site: dict(values) for site, values in sorted(gaussian_stats.items())
        },
        "clamp_stats": {
            site: dict(values) for site, values in sorted(clamp_stats.items())
        },
        "calibration_clamp_stats": {
            site: dict(values)
            for site, values in sorted(calibration_clamp_stats.items())
        },
        "run_identity": dict(run_identity),
        "rng_contract": dict(rng_contract),
    }
    runtime_files.new_json(result_path, payload)
    return payload


def load_shard_result(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate one shard result plus its external raw predictions."""

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported shard result schema: {path}")
    prediction = payload.get("predictions")
    if not isinstance(prediction, dict) or set(
        ("path", "dtype", "byte_order", "count", "sha256")
    ) - set(prediction):
        raise ValueError(f"incomplete prediction metadata: {path}")
    if prediction["dtype"] != "int64" or prediction["byte_order"] != "little":
        raise ValueError(f"unsupported prediction representation: {path}")
    relative = Path(prediction["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"prediction path must be local to its shard result: {path}")
    raw = (path.parent / relative).read_bytes()
    if len(raw) != int(prediction["count"]) * 8:
        raise ValueError(f"prediction byte count mismatch: {path}")
    if hashlib.sha256(raw).hexdigest() != prediction["sha256"]:
        raise ValueError(f"prediction digest mismatch: {path}")
    decode_predictions(raw)
    return payload, raw


def _aggregate_stats(
    records: Iterable[Mapping[str, Mapping[str, int | float]]],
) -> dict[str, dict[str, int | float]]:
    aggregate: dict[str, dict[str, int | float]] = {}
    for record in records:
        for site, values in record.items():
            target = aggregate.setdefault(site, {})
            for field, value in values.items():
                if field in SUM_FIELDS:
                    target[field] = int(target.get(field, 0)) + int(value)
                elif field in MIN_FIELDS:
                    numeric = float(value)
                    previous = float(target.get(field, 0.0))
                    if numeric > 0.0 and (previous == 0.0 or numeric < previous):
                        target[field] = numeric
                    else:
                        target.setdefault(field, previous)
                elif field in MAX_FIELDS:
                    target[field] = max(float(target.get(field, 0.0)), float(value))
                else:
                    raise ValueError(f"unsupported counter field {field!r} at {site!r}")
    return {site: aggregate[site] for site in sorted(aggregate)}


def _shared_rng_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {
        "global_batch_start",
        "global_batch_stop",
        "start_offset",
        "end_offset",
        "observed_batches",
    }
    return {key: value for key, value in contract.items() if key not in omitted}


def merge_shard_results(
    shard_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Validate, concatenate, and aggregate exact shard result artifacts."""

    paths = [Path(path) for path in shard_paths]
    loaded = [(*load_shard_result(path), path) for path in paths]
    if len(loaded) != SHARD_COUNT:
        raise ValueError("exact ViT merging requires exactly two shard results")
    loaded.sort(key=lambda item: int(item[0]["shard_index"]))
    payloads = [item[0] for item in loaded]
    if [item["shard_index"] for item in payloads] != [0, 1]:
        raise ValueError("shard indices must be exactly zero and one")
    if any(item.get("shard_count") != SHARD_COUNT for item in payloads):
        raise ValueError("shard count mismatch")
    if payloads[0]["run_identity"] != payloads[1]["run_identity"]:
        raise ValueError("shard run identities differ")
    shared_rng = _shared_rng_identity(payloads[0]["rng_contract"])
    if shared_rng != _shared_rng_identity(payloads[1]["rng_contract"]):
        raise ValueError("shard random generator contracts differ")
    trace_sha256 = str(shared_rng.get("trace_sha256", ""))
    if (
        shared_rng.get("schema_version") != SCHEMA_VERSION
        or shared_rng.get("algorithm") != "torch_cuda_generator_philox_offset"
        or shared_rng.get("preflight_batches") != 2
        or shared_rng.get("identity_sha256")
        != canonical_json_sha256(payloads[0]["run_identity"])
        or len(trace_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trace_sha256)
    ):
        raise ValueError("shard random generator contract is incomplete or unbound")

    first_interval, second_interval = (
        payloads[0]["interval"], payloads[1]["interval"]
    )
    if int(first_interval["sample_start"]) != 0:
        raise ValueError("shard coverage must start at sample zero")
    if int(first_interval["sample_stop"]) != int(second_interval["sample_start"]):
        raise ValueError("shard sample intervals have a gap or overlap")
    if int(first_interval["batch_stop"]) != int(second_interval["batch_start"]):
        raise ValueError("shard batch intervals have a gap or overlap")
    population = int(first_interval["population"])
    if int(second_interval["sample_stop"]) != population:
        raise ValueError("shard coverage does not end at the selected population")
    if any(int(item["interval"]["population"]) != population for item in payloads):
        raise ValueError("shard populations differ")
    batch_size = int(first_interval["batch_size"])
    if int(shared_rng.get("batch_size", -1)) != batch_size:
        raise ValueError("shard random generator batch size differs")
    for index, (payload, raw, _path) in enumerate(loaded):
        interval = payload["interval"]
        if (
            int(interval["shard_index"]) != index
            or int(interval["shard_count"]) != SHARD_COUNT
            or int(interval["batch_size"]) != batch_size
            or tuple(
                int(interval[field])
                for field in (
                    "sample_start",
                    "sample_stop",
                    "batch_start",
                    "batch_stop",
                )
            )
            != exact_batch_shard_bounds(population, batch_size, SHARD_COUNT, index)
        ):
            raise ValueError("shard interval differs from the canonical batch partition")
        sample_count = int(interval["sample_stop"]) - int(interval["sample_start"])
        if (
            len(raw) != sample_count * 8
            or int(payload["predictions"]["count"]) != sample_count
            or int(payload["counts"]["evaluated"]) != sample_count
        ):
            raise ValueError("shard prediction or task count differs from its interval")

    stride = validate_cuda_offset(
        int(payloads[0]["rng_contract"]["batch_stride"]),
        name="shard batch stride",
    )
    for payload in payloads:
        interval = payload["interval"]
        contract = payload["rng_contract"]
        batch_start = int(interval["batch_start"])
        batch_stop = int(interval["batch_stop"])
        if (
            int(contract["global_batch_start"]) != batch_start
            or int(contract["global_batch_stop"]) != batch_stop
            or int(contract["start_offset"])
            != checked_batch_offset(batch_start, stride)
            or int(contract["end_offset"])
            != checked_batch_offset(batch_stop, stride)
            or int(contract["observed_batches"]) != batch_stop - batch_start
        ):
            raise ValueError("shard random generator offset continuity failed")
    if int(payloads[0]["rng_contract"]["end_offset"]) != int(
        payloads[1]["rng_contract"]["start_offset"]
    ):
        raise ValueError("random generator offsets have a gap or overlap")

    raw = b"".join(item[1] for item in loaded)
    predictions = decode_predictions(raw)
    if len(predictions) != population:
        raise ValueError("merged prediction count differs from the population")
    correct = sum(int(item["counts"]["correct"]) for item in payloads)
    evaluated = sum(int(item["counts"]["evaluated"]) for item in payloads)
    if evaluated != population or not 0 <= correct <= evaluated:
        raise ValueError("merged task counts are inconsistent")

    raw_path = output_path.with_suffix(".predictions.int64")
    if output_path.exists() or raw_path.exists():
        raise FileExistsError(output_path if output_path.exists() else raw_path)
    _atomic_bytes(raw_path, raw)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "shard_count": SHARD_COUNT,
        "interval": {
            "sample_start": 0,
            "sample_stop": population,
            "batch_start": 0,
            "batch_stop": int(second_interval["batch_stop"]),
            "population": population,
            "batch_size": int(first_interval["batch_size"]),
        },
        "counts": {
            "correct": correct,
            "evaluated": evaluated,
            "accuracy": correct / evaluated,
        },
        "predictions": {
            "path": raw_path.name,
            "dtype": "int64",
            "byte_order": "little",
            "count": len(predictions),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "gaussian_stats": _aggregate_stats(
            item["gaussian_stats"] for item in payloads
        ),
        "clamp_stats": _aggregate_stats(item["clamp_stats"] for item in payloads),
        "calibration_clamp_stats": _aggregate_stats(
            item["calibration_clamp_stats"] for item in payloads
        ),
        "run_identity": payloads[0]["run_identity"],
        "rng_contract": {
            **shared_rng,
            "global_batch_start": 0,
            "global_batch_stop": int(second_interval["batch_stop"]),
            "start_offset": 0,
            "end_offset": int(payloads[1]["rng_contract"]["end_offset"]),
            "observed_batches": int(second_interval["batch_stop"]),
        },
        "shards": [
            {
                "result_path": str(path.resolve()),
                "result_sha256": identity.sha256_file(path),
                "prediction_sha256": payload["predictions"]["sha256"],
                "interval": payload["interval"],
            }
            for payload, _raw, path in loaded
        ],
    }
    runtime_files.new_json(output_path, aggregate)
    return aggregate


def noise_profile_fractions(
    profile: str,
    *,
    linear_std_frac: float,
    log_std_frac: float,
) -> tuple[float, float]:
    """Resolve the selected encoder injection coverage."""

    for name, value in (
        ("linear standard deviation fraction", linear_std_frac),
        ("logarithmic standard deviation fraction", log_std_frac),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if profile == "np":
        return linear_std_frac, 0.0
    if profile == "nl":
        return 0.0, log_std_frac
    if profile == "joint":
        return linear_std_frac, log_std_frac
    raise ValueError("noise profile must be np, nl, or joint")


def _has_option(arguments: Iterable[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in arguments)


def build_shard_command(
    args: argparse.Namespace,
    *,
    shard_index: int,
    source_commit: str,
    contract_path: Path,
    result_path: Path | None,
    preflight: bool = False,
) -> list[str]:
    """Build one evaluator command with all exact-sharding controls owned here."""

    evaluator_args = list(args.evaluator_args)
    if evaluator_args[:1] == ["--"]:
        evaluator_args = evaluator_args[1:]
    conflicts = sorted(
        option
        for option in MANAGED_EVALUATOR_OPTIONS
        if _has_option(evaluator_args, option)
    )
    if conflicts:
        raise ValueError(f"runner-owned evaluator options were supplied: {conflicts}")
    if not _has_option(evaluator_args, "--checkpoint-sha256"):
        raise ValueError("evaluator arguments must include --checkpoint-sha256")
    linear, logarithmic = noise_profile_fractions(
        args.noise_profile,
        linear_std_frac=args.linear_std_frac,
        log_std_frac=args.log_std_frac,
    )
    command = [
        args.python_bin,
        str(args.source_root / "scripts/evaluation/error_analysis_vit.py"),
        *evaluator_args,
        "--device",
        "cuda",
        "--model_backend",
        "spiking",
        "--experiment_name",
        (
            f"{args.output_dir.name}_rng_preflight"
            if preflight
            else f"{args.output_dir.name}_shard_{shard_index}"
        ),
        "--batch_size",
        str(args.batch_size),
        "--evaluation-prefix-samples",
        str(args.prefix_samples),
        "--evaluation-shard-count",
        str(1 if preflight else SHARD_COUNT),
        "--evaluation-shard-index",
        str(shard_index),
        "--gaussian-rng-contract-path",
        str(contract_path),
        "--gaussian-rng-contract-timeout-seconds",
        str(args.contract_timeout_seconds),
        "--gaussian-time-noise",
        "--time-noise-std-frac",
        "0",
        "--linear-time-noise-std-frac",
        repr(linear),
        "--log-time-noise-std-frac",
        repr(logarithmic),
        "--time-noise-seed",
        str(args.seed),
        "--time-noise-deadline-margin-std",
        repr(args.deadline_margin_std),
        "--source-commit",
        source_commit,
        "--no-tensorboard",
        "--report-clamp-stats",
    ]
    if preflight:
        command.extend(("--gaussian-rng-preflight-path", str(contract_path)))
    else:
        if result_path is None:
            raise ValueError("a shard result path is required outside preflight")
        command.extend(("--shard-result-path", str(result_path)))
    return command


def _terminate(children: Iterable[subprocess.Popen[Any]]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 15.0
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Launch two admitted GPU workers and merge only fully verified results."""

    args.source_root = args.source_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists")
    if len(set(args.gpus)) != SHARD_COUNT:
        raise ValueError("exact evaluation requires two distinct physical GPUs")
    activity = local_gpu.gpu_activity(gpu_ids=tuple(args.gpus))
    unavailable = {
        gpu: sample
        for gpu, sample in activity.items()
        if not local_gpu.gpu_available(sample)
    }
    if unavailable:
        raise RuntimeError(f"requested physical GPUs are not available: {unavailable}")

    source_commit = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    identity.verify_clean_checkout(args.source_root, source_commit)
    args.output_dir.mkdir(parents=True)
    contract_path = args.output_dir / "gaussian-rng-contract.json"
    result_paths = [args.output_dir / f"shard-{index}.json" for index in range(2)]
    log_paths = [args.output_dir / f"shard-{index}.log" for index in range(2)]
    preflight_log = args.output_dir / "rng-preflight.log"
    preflight_command = build_shard_command(
        args,
        shard_index=0,
        source_commit=source_commit,
        contract_path=contract_path,
        result_path=None,
        preflight=True,
    )
    commands = [
        build_shard_command(
            args,
            shard_index=index,
            source_commit=source_commit,
            contract_path=contract_path,
            result_path=result_paths[index],
        )
        for index in range(2)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(args.source_root),
        "source_commit": source_commit,
        "runner_sha256": identity.sha256_file(Path(__file__)),
        "evaluator_sha256": identity.sha256_file(
            args.source_root / "scripts/evaluation/error_analysis_vit.py"
        ),
        "gpus": args.gpus,
        "gpu_activity": activity,
        "prefix_samples": args.prefix_samples,
        "batch_size": args.batch_size,
        "noise_profile": args.noise_profile,
        "seed": args.seed,
        "deadline_margin_std": args.deadline_margin_std,
        "commands": commands,
        "preflight_command": preflight_command,
    }
    runtime_files.new_json(args.output_dir / "run.json", manifest)

    preflight_environment = os.environ.copy()
    preflight_environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpus[0]),
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(
            str(path)
            for path in (
                args.source_root,
                args.source_root / "src/transformers/src",
                args.source_root / "src/spikingjelly",
            )
        ),
    )
    with preflight_log.open("xb") as handle:
        completed = subprocess.run(
            preflight_command,
            cwd=args.source_root,
            env=preflight_environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"random generator preflight failed with code {completed.returncode}; "
            f"see {preflight_log}"
        )
    if not contract_path.is_file():
        raise RuntimeError("random generator preflight did not publish its contract")

    children: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        for index, command in enumerate(commands):
            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES=str(args.gpus[index]),
                WANDB_MODE="disabled",
                PYTHONUNBUFFERED="1",
                PYTHONPATH=os.pathsep.join(
                    str(path)
                    for path in (
                        args.source_root,
                        args.source_root / "src/transformers/src",
                        args.source_root / "src/spikingjelly",
                    )
                ),
            )
            handle = log_paths[index].open("xb")
            handles.append(handle)
            children.append(
                subprocess.Popen(
                    command,
                    cwd=args.source_root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        while True:
            codes = [child.poll() for child in children]
            failures = [code for code in codes if code not in (None, 0)]
            if failures:
                _terminate(children)
                raise RuntimeError(
                    f"shard process failed with codes {codes}; see {log_paths}"
                )
            if all(code == 0 for code in codes):
                break
            time.sleep(2.0)
    except BaseException:
        _terminate(children)
        raise
    finally:
        for handle in handles:
            handle.close()

    aggregate = merge_shard_results(
        result_paths, args.output_dir / "aggregate.json"
    )
    print(
        "Exact merged result — "
        f"correct: {aggregate['counts']['correct']}, "
        f"evaluated: {aggregate['counts']['evaluated']}, "
        f"accuracy: {aggregate['counts']['accuracy']:.8f}, "
        f"prediction_sha256: {aggregate['predictions']['sha256']}",
        flush=True,
    )
    return aggregate


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--gpus", type=int, nargs=2, default=(4, 5))
    parser.add_argument("--prefix-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-profile", choices=("np", "nl", "joint"), required=True)
    parser.add_argument("--linear-std-frac", type=float, default=DEFAULT_LINEAR_STD_FRAC)
    parser.add_argument("--log-std-frac", type=float, default=DEFAULT_LOG_STD_FRAC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deadline-margin-std", type=float, default=0.0)
    parser.add_argument("--contract-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "evaluator_args",
        nargs=argparse.REMAINDER,
        help="ordinary evaluator arguments after --; runner-owned options are rejected",
    )
    args = parser.parse_args()
    if args.prefix_samples <= 0 or args.batch_size <= 0:
        parser.error("prefix samples and batch size must be positive")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if (
        not math.isfinite(args.deadline_margin_std)
        or args.deadline_margin_std < 0.0
        or not math.isfinite(args.contract_timeout_seconds)
        or args.contract_timeout_seconds <= 0.0
    ):
        parser.error("margin and contract timeout must be finite and valid")
    return args


# @lat: [[evaluation#Evaluation and Verification#Exact ViT Timing Noise Shards]]
def main() -> None:
    execute(parse_arguments())


if __name__ == "__main__":
    main()
