#!/usr/bin/env python3
"""Run and merge an exact two-process ViT timing-noise evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping


SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from utils.transforms.calibration import (
    CALIBRATION_COMPATIBLE_SOURCE_COMMIT_ENV,
    CALIBRATION_COMPATIBLE_VIT_EVALUATOR_SHA256_ENV,
)


SCHEMA_VERSION = 1
SHARD_COUNT = 2
DEFAULT_LINEAR_STD_FRAC = 0.03785
DEFAULT_LOG_STD_FRAC = 0.02712
GAUSSIAN_INTEGER_FIELDS = frozenset(
    {
        "events",
        "misses",
        "deadline_events",
        "outputs",
        "output_underflows",
        "output_overflows",
    }
)
GAUSSIAN_MIN_FIELDS = frozenset({"deadline_ulp_min"})
GAUSSIAN_MAX_FIELDS = frozenset({"deadline_ulp_max"})
GAUSSIAN_NUMBER_FIELDS = GAUSSIAN_MIN_FIELDS | GAUSSIAN_MAX_FIELDS
CLAMP_STAT_FIELDS = frozenset({"values", "underflows", "overflows"})
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
        "--source-root",
        "--source-commit",
        "--tensorboard",
        "--no-tensorboard",
        "--report-clamp-stats",
        "--no-report-clamp-stats",
    }
)
INTERVAL_FIELDS = (
    "shard_index",
    "shard_count",
    "sample_start",
    "sample_stop",
    "batch_start",
    "batch_stop",
    "population",
    "batch_size",
)


def exact_nonnegative_int(value: Any, *, name: str) -> int:
    """Accept only a native nonnegative integer without lossy coercion."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def exact_positive_int(value: Any, *, name: str) -> int:
    """Accept only a native positive integer without lossy coercion."""

    result = exact_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def nonnegative_number(value: Any, *, name: str) -> int | float:
    """Accept a finite nonnegative integer or real statistic, excluding booleans."""

    if type(value) is int:
        if value >= 0:
            return value
    elif type(value) is float and math.isfinite(value) and value >= 0.0:
        return value
    raise ValueError(f"{name} must be a finite nonnegative number")


def exact_batch_shard_bounds(
    population: int,
    batch_size: int,
    shard_count: int,
    shard_index: int,
) -> tuple[int, int, int, int]:
    """Return sample and batch bounds without splitting a global batch."""

    population = exact_positive_int(population, name="population")
    batch_size = exact_positive_int(batch_size, name="batch size")
    shard_count = exact_positive_int(shard_count, name="shard count")
    shard_index = exact_nonnegative_int(shard_index, name="shard index")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be inside shard count")

    global_batches = math.ceil(population / batch_size)
    if shard_count > global_batches:
        raise ValueError("each exact shard must contain at least one global batch")
    base, remainder = divmod(global_batches, shard_count)
    batch_start = shard_index * base + min(shard_index, remainder)
    batch_stop = batch_start + base + (1 if shard_index < remainder else 0)
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

    offset = exact_nonnegative_int(offset, name=name)
    if offset > MAX_CUDA_OFFSET:
        raise ValueError(f"{name} is outside the supported CUDA offset range")
    if offset % 4:
        raise ValueError(f"{name} must be aligned to four Philox units")
    return offset


def checked_batch_offset(batch_index: int, batch_stride: int) -> int:
    """Multiply a global batch index and stride without unsigned wraparound."""

    batch_index = exact_nonnegative_int(batch_index, name="global batch index")
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
        if (
            not isinstance(call.get("site"), str)
            or call.get("encoding") not in ("linear", "log")
            or not isinstance(call.get("dtype"), str)
            or type(call.get("sampled")) is not bool
            or not isinstance(call.get("shape"), list)
        ):
            raise ValueError("Gaussian call trace contains malformed fields")
        before = validate_cuda_offset(call["before_offset"], name="call before offset")
        after = validate_cuda_offset(call["after_offset"], name="call after offset")
        if before != previous or after < before:
            raise ValueError("Gaussian call offsets are not contiguous and monotone")
        relative.append(
            {
                "site": call["site"],
                "encoding": call["encoding"],
                "shape": [
                    exact_nonnegative_int(value, name="Gaussian call shape")
                    for value in call["shape"]
                ],
                "dtype": call["dtype"],
                "sampled": call["sampled"],
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

    batch_size = exact_positive_int(batch_size, name="preflight batch size")
    observed = list(batches)
    if len(observed) != 2:
        raise ValueError("random generator preflight requires exactly two batches")
    normalized: list[list[dict[str, Any]]] = []
    strides: list[int] = []
    for start, stop, trace in observed:
        start = validate_cuda_offset(start, name="batch start offset")
        stop = validate_cuda_offset(stop, name="batch stop offset")
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
        contract["batch_stride"], name="contract batch stride"
    )
    contract_batch_size = exact_positive_int(
        contract["batch_size"], name="contract batch size"
    )
    sample_count = exact_positive_int(sample_count, name="sample count")
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
    if sample_count == contract_batch_size:
        if actual != expected:
            raise RuntimeError("full batch Gaussian call trace differs from preflight")
        return
    if sample_count >= contract_batch_size:
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
            or expected_shape[0] != contract_batch_size
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
        exact_nonnegative_int(
            payload.get("schema_version"), name="contract schema version"
        )
        != SCHEMA_VERSION
        or payload.get("algorithm") != "torch_cuda_generator_philox_offset"
        or payload.get("identity_sha256") != canonical_json_sha256(run_identity)
    ):
        raise ValueError("Gaussian random generator preflight identity differs")
    validate_cuda_offset(payload["batch_stride"], name="contract batch stride")
    exact_positive_int(payload.get("batch_size"), name="contract batch size")
    if (
        exact_positive_int(
            payload.get("preflight_batches"), name="contract preflight batches"
        )
        != 2
        or not isinstance(payload.get("trace"), list)
    ):
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


def validated_interval(interval: Mapping[str, Any]) -> dict[str, int]:
    """Validate the complete exact shard interval representation."""

    if not isinstance(interval, Mapping):
        raise ValueError("shard interval must be a mapping")
    result = {
        field: exact_nonnegative_int(interval.get(field), name=f"interval {field}")
        for field in INTERVAL_FIELDS
    }
    exact_positive_int(result["shard_count"], name="interval shard count")
    exact_positive_int(result["population"], name="interval population")
    exact_positive_int(result["batch_size"], name="interval batch size")
    return result


def validated_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    """Validate exact task counters without accepting numeric lookalikes."""

    if not isinstance(counts, Mapping):
        raise ValueError("task counts must be a mapping")
    return {
        field: exact_nonnegative_int(counts.get(field), name=f"task count {field}")
        for field in ("correct", "evaluated")
    }


def _validate_gaussian_site(
    site: str, values: Mapping[str, int | float]
) -> None:
    """Reject Gaussian counters that the instrumentation cannot produce."""

    events = values["events"]
    outputs = values["outputs"]
    if values["misses"] > events:
        raise ValueError(f"Gaussian misses exceed events at {site!r}")
    if values["deadline_events"] > events:
        raise ValueError(f"Gaussian deadline events exceed events at {site!r}")
    if values["output_underflows"] > outputs:
        raise ValueError(f"Gaussian output underflows exceed outputs at {site!r}")
    if values["output_overflows"] > outputs:
        raise ValueError(f"Gaussian output overflows exceed outputs at {site!r}")
    if values["output_underflows"] + values["output_overflows"] > outputs:
        raise ValueError(
            f"Gaussian output saturation counts exceed outputs at {site!r}"
        )


def _validate_clamp_site(site: str, values: Mapping[str, int | float]) -> None:
    """Reject clamp counters that the instrumentation cannot produce."""

    total = values["values"]
    if values["underflows"] > total:
        raise ValueError(f"clamp underflows exceed values at {site!r}")
    if values["overflows"] > total:
        raise ValueError(f"clamp overflows exceed values at {site!r}")
    if values["underflows"] + values["overflows"] > total:
        raise ValueError(f"clamp saturation counts exceed values at {site!r}")


def validated_stats(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    integer_fields: frozenset[str],
    number_fields: frozenset[str] = frozenset(),
    validate_site: Callable[[str, Mapping[str, int | float]], None] | None = None,
) -> dict[str, dict[str, int | float]]:
    """Validate every site against one complete counter schema."""

    if not isinstance(stats, Mapping):
        raise ValueError("statistics must be a mapping")
    if any(not isinstance(site, str) or not site for site in stats):
        raise ValueError("statistic sites must be nonempty strings")
    result: dict[str, dict[str, int | float]] = {}
    for site in sorted(stats):
        values = stats[site]
        if not isinstance(values, Mapping):
            raise ValueError("statistic sites must name counter mappings")
        expected_fields = integer_fields | number_fields
        if set(values) != expected_fields:
            raise ValueError(f"statistic schema differs at {site!r}")
        validated: dict[str, int | float] = {}
        for field, value in values.items():
            if field in integer_fields:
                validated[field] = exact_nonnegative_int(
                    value, name=f"statistic {site}/{field}"
                )
            elif field in number_fields:
                validated[field] = nonnegative_number(
                    value, name=f"statistic {site}/{field}"
                )
        if validate_site is not None:
            validate_site(site, validated)
        result[site] = validated
    return result


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
    checked_interval = validated_interval(interval)
    checked_counts = validated_counts(counts)
    checked_gaussian_stats = validated_stats(
        gaussian_stats,
        integer_fields=GAUSSIAN_INTEGER_FIELDS,
        number_fields=GAUSSIAN_NUMBER_FIELDS,
        validate_site=_validate_gaussian_site,
    )
    checked_clamp_stats = validated_stats(
        clamp_stats,
        integer_fields=CLAMP_STAT_FIELDS,
        validate_site=_validate_clamp_site,
    )
    checked_calibration_stats = validated_stats(
        calibration_clamp_stats,
        integer_fields=CLAMP_STAT_FIELDS,
        validate_site=_validate_clamp_site,
    )
    sample_count = checked_interval["sample_stop"] - checked_interval["sample_start"]
    if sample_count <= 0 or prediction_count != sample_count:
        raise ValueError("prediction count must equal the nonempty shard interval")
    if checked_counts["evaluated"] != sample_count:
        raise ValueError("evaluated count must equal the shard interval")
    if checked_counts["correct"] > sample_count:
        raise ValueError("correct count must lie inside the evaluated count")

    predictions_path = result_path.with_suffix(".predictions.int64")
    if predictions_path.exists():
        raise FileExistsError(predictions_path)
    _atomic_bytes(predictions_path, raw)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": checked_interval["shard_index"],
        "shard_count": checked_interval["shard_count"],
        "interval": checked_interval,
        "counts": checked_counts,
        "predictions": {
            "path": predictions_path.name,
            "dtype": "int64",
            "byte_order": "little",
            "count": prediction_count,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "gaussian_stats": checked_gaussian_stats,
        "clamp_stats": checked_clamp_stats,
        "calibration_clamp_stats": checked_calibration_stats,
        "run_identity": dict(run_identity),
        "rng_contract": dict(rng_contract),
    }
    runtime_files.new_json(result_path, payload)
    return payload


def load_shard_result(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate one shard result plus its external raw predictions."""

    payload = json.loads(path.read_text())
    if (
        exact_nonnegative_int(
            payload.get("schema_version"), name="shard schema version"
        )
        != SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported shard result schema: {path}")
    prediction = payload.get("predictions")
    if not isinstance(prediction, dict) or set(
        ("path", "dtype", "byte_order", "count", "sha256")
    ) - set(prediction):
        raise ValueError(f"incomplete prediction metadata: {path}")
    if prediction["dtype"] != "int64" or prediction["byte_order"] != "little":
        raise ValueError(f"unsupported prediction representation: {path}")
    prediction_count = exact_nonnegative_int(
        prediction["count"], name="prediction count"
    )
    if not isinstance(prediction["path"], str):
        raise ValueError(f"prediction path must be a string: {path}")
    relative = Path(prediction["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"prediction path must be local to its shard result: {path}")
    raw = (path.parent / relative).read_bytes()
    if len(raw) != prediction_count * 8:
        raise ValueError(f"prediction byte count mismatch: {path}")
    if hashlib.sha256(raw).hexdigest() != prediction["sha256"]:
        raise ValueError(f"prediction digest mismatch: {path}")
    decode_predictions(raw)
    return payload, raw


def _aggregate_stats(
    records: Iterable[Mapping[str, Mapping[str, int | float]]],
    *,
    integer_fields: frozenset[str],
    min_fields: frozenset[str] = frozenset(),
    max_fields: frozenset[str] = frozenset(),
    statistic_name: str,
    validate_site: Callable[[str, Mapping[str, int | float]], None],
) -> dict[str, dict[str, int | float]]:
    validated_records = [
        validated_stats(
            unchecked_record,
            integer_fields=integer_fields,
            number_fields=min_fields | max_fields,
            validate_site=validate_site,
        )
        for unchecked_record in records
    ]
    if validated_records:
        expected_sites = set(validated_records[0])
        if any(set(record) != expected_sites for record in validated_records[1:]):
            raise ValueError(f"{statistic_name} statistic sites differ across shards")

    aggregate: dict[str, dict[str, int | float]] = {}
    for record in validated_records:
        for site, values in record.items():
            target = aggregate.setdefault(site, {})
            for field, value in values.items():
                if field in integer_fields:
                    target[field] = exact_nonnegative_int(
                        target.get(field, 0), name=f"aggregate {site}/{field}"
                    ) + exact_nonnegative_int(value, name=f"statistic {site}/{field}")
                elif field in min_fields:
                    numeric = nonnegative_number(value, name=f"statistic {site}/{field}")
                    previous = nonnegative_number(
                        target.get(field, 0.0), name=f"aggregate {site}/{field}"
                    )
                    if numeric > 0.0 and (previous == 0.0 or numeric < previous):
                        target[field] = numeric
                    else:
                        target.setdefault(field, previous)
                elif field in max_fields:
                    target[field] = max(
                        nonnegative_number(
                            target.get(field, 0.0), name=f"aggregate {site}/{field}"
                        ),
                        nonnegative_number(value, name=f"statistic {site}/{field}"),
                    )
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
    for payload, _raw, _path in loaded:
        exact_nonnegative_int(payload.get("shard_index"), name="shard index")
        exact_positive_int(payload.get("shard_count"), name="shard count")
    loaded.sort(key=lambda item: item[0]["shard_index"])
    payloads = [item[0] for item in loaded]
    if [item["shard_index"] for item in payloads] != [0, 1]:
        raise ValueError("shard indices must be exactly zero and one")
    if any(item.get("shard_count") != SHARD_COUNT for item in payloads):
        raise ValueError("shard count mismatch")
    for payload in payloads:
        run_identity = payload.get("run_identity")
        if not isinstance(run_identity, Mapping):
            raise ValueError("shard run identity must be a mapping")
        identity.checked_hash(run_identity.get("checkpoint_sha256"))
        identity.checked_hash(run_identity.get("loaded_model_state_sha256"))
    if payloads[0]["run_identity"] != payloads[1]["run_identity"]:
        raise ValueError("shard run identities differ")
    if any(not isinstance(item.get("rng_contract"), Mapping) for item in payloads):
        raise ValueError("shard random generator contract must be a mapping")
    shared_rng = _shared_rng_identity(payloads[0]["rng_contract"])
    if shared_rng != _shared_rng_identity(payloads[1]["rng_contract"]):
        raise ValueError("shard random generator contracts differ")
    trace_sha256 = shared_rng.get("trace_sha256")
    if (
        exact_nonnegative_int(
            shared_rng.get("schema_version"), name="contract schema version"
        )
        != SCHEMA_VERSION
        or shared_rng.get("algorithm") != "torch_cuda_generator_philox_offset"
        or exact_positive_int(
            shared_rng.get("preflight_batches"), name="contract preflight batches"
        )
        != 2
        or shared_rng.get("identity_sha256")
        != canonical_json_sha256(payloads[0]["run_identity"])
        or not isinstance(trace_sha256, str)
        or len(trace_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trace_sha256)
    ):
        raise ValueError("shard random generator contract is incomplete or unbound")

    first_interval, second_interval = (
        validated_interval(payloads[0].get("interval")),
        validated_interval(payloads[1].get("interval")),
    )
    if first_interval["sample_start"] != 0:
        raise ValueError("shard coverage must start at sample zero")
    if first_interval["sample_stop"] != second_interval["sample_start"]:
        raise ValueError("shard sample intervals have a gap or overlap")
    if first_interval["batch_stop"] != second_interval["batch_start"]:
        raise ValueError("shard batch intervals have a gap or overlap")
    population = first_interval["population"]
    if second_interval["sample_stop"] != population:
        raise ValueError("shard coverage does not end at the selected population")
    if second_interval["population"] != population:
        raise ValueError("shard populations differ")
    batch_size = first_interval["batch_size"]
    if exact_positive_int(
        shared_rng.get("batch_size"), name="contract batch size"
    ) != batch_size:
        raise ValueError("shard random generator batch size differs")
    for index, (payload, raw, _path) in enumerate(loaded):
        interval = (first_interval, second_interval)[index]
        if (
            interval["shard_index"] != index
            or interval["shard_count"] != SHARD_COUNT
            or interval["batch_size"] != batch_size
            or tuple(
                interval[field]
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
        sample_count = interval["sample_stop"] - interval["sample_start"]
        prediction_count = exact_nonnegative_int(
            payload["predictions"].get("count"), name="prediction count"
        )
        counts = validated_counts(payload.get("counts"))
        if (
            len(raw) != sample_count * 8
            or prediction_count != sample_count
            or counts["evaluated"] != sample_count
            or counts["correct"] > counts["evaluated"]
        ):
            raise ValueError("shard prediction or task count differs from its interval")

    stride = validate_cuda_offset(
        payloads[0]["rng_contract"]["batch_stride"],
        name="shard batch stride",
    )
    for index, payload in enumerate(payloads):
        interval = (first_interval, second_interval)[index]
        contract = payload["rng_contract"]
        batch_start = interval["batch_start"]
        batch_stop = interval["batch_stop"]
        if (
            exact_nonnegative_int(
                contract.get("global_batch_start"), name="contract batch start"
            )
            != batch_start
            or exact_nonnegative_int(
                contract.get("global_batch_stop"), name="contract batch stop"
            )
            != batch_stop
            or validate_cuda_offset(
                contract.get("start_offset"), name="contract start offset"
            )
            != checked_batch_offset(batch_start, stride)
            or validate_cuda_offset(
                contract.get("end_offset"), name="contract end offset"
            )
            != checked_batch_offset(batch_stop, stride)
            or exact_nonnegative_int(
                contract.get("observed_batches"), name="contract observed batches"
            )
            != batch_stop - batch_start
        ):
            raise ValueError("shard random generator offset continuity failed")
    if (
        payloads[0]["rng_contract"]["end_offset"]
        != payloads[1]["rng_contract"]["start_offset"]
    ):
        raise ValueError("random generator offsets have a gap or overlap")

    raw = b"".join(item[1] for item in loaded)
    predictions = decode_predictions(raw)
    if len(predictions) != population:
        raise ValueError("merged prediction count differs from the population")
    checked_counts = [validated_counts(item.get("counts")) for item in payloads]
    correct = sum(item["correct"] for item in checked_counts)
    evaluated = sum(item["evaluated"] for item in checked_counts)
    if evaluated != population or not 0 <= correct <= evaluated:
        raise ValueError("merged task counts are inconsistent")
    gaussian_stats = _aggregate_stats(
        (item["gaussian_stats"] for item in payloads),
        integer_fields=GAUSSIAN_INTEGER_FIELDS,
        min_fields=GAUSSIAN_MIN_FIELDS,
        max_fields=GAUSSIAN_MAX_FIELDS,
        statistic_name="Gaussian",
        validate_site=_validate_gaussian_site,
    )
    clamp_stats = _aggregate_stats(
        (item["clamp_stats"] for item in payloads),
        integer_fields=CLAMP_STAT_FIELDS,
        statistic_name="clamp",
        validate_site=_validate_clamp_site,
    )
    calibration_clamp_stats = _aggregate_stats(
        (item["calibration_clamp_stats"] for item in payloads),
        integer_fields=CLAMP_STAT_FIELDS,
        statistic_name="calibration clamp",
        validate_site=_validate_clamp_site,
    )

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
            "batch_stop": second_interval["batch_stop"],
            "population": population,
            "batch_size": first_interval["batch_size"],
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
        "gaussian_stats": gaussian_stats,
        "clamp_stats": clamp_stats,
        "calibration_clamp_stats": calibration_clamp_stats,
        "run_identity": payloads[0]["run_identity"],
        "rng_contract": {
            **shared_rng,
            "global_batch_start": 0,
            "global_batch_stop": second_interval["batch_stop"],
            "start_offset": 0,
            "end_offset": payloads[1]["rng_contract"]["end_offset"],
            "observed_batches": second_interval["batch_stop"],
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


def evaluator_arguments(args: argparse.Namespace) -> list[str]:
    """Return evaluator arguments after rejecting runner-owned controls."""

    arguments = list(args.evaluator_args)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    conflicts = sorted(
        option for option in MANAGED_EVALUATOR_OPTIONS if _has_option(arguments, option)
    )
    if conflicts:
        raise ValueError(f"runner-owned evaluator options were supplied: {conflicts}")
    return arguments


def evaluator_option(arguments: list[str], option: str) -> str:
    """Read one required evaluator option and reject missing or duplicate values."""

    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith(option + "="):
            values.append(argument.partition("=")[2])
        elif argument == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"evaluator option has no value: {option}")
            values.append(arguments[index + 1])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"evaluator arguments require exactly one {option}")
    return values[0]


def optional_evaluator_option(arguments: list[str], option: str) -> str | None:
    """Read one optional evaluator option while rejecting duplicate values."""

    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith(option + "="):
            values.append(argument.partition("=")[2])
        elif argument == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"evaluator option has no value: {option}")
            values.append(arguments[index + 1])
    if len(values) > 1:
        raise ValueError(f"evaluator arguments contain duplicate {option}")
    if values and not values[0]:
        raise ValueError(f"evaluator option has no value: {option}")
    return values[0] if values else None


def evaluation_entrypoint(source_root: Path, arguments: list[str]) -> Path:
    """Select the sole evaluator entrypoint required by the calibration mode."""

    mode = optional_evaluator_option(arguments, "--calibration-mode")
    if mode not in {"validate", "inference"}:
        raise ValueError(
            "exact evaluation requires calibration mode validate or inference"
        )
    evaluator_option(arguments, "--calibration-dataset-path")
    evaluator_option(arguments, "--calibration-dataset-fingerprint")
    return source_root / "scripts/analysis/evaluate_calibrated_vit.py"


def calibration_compatibility_identity(arguments: list[str]) -> dict[str, str] | None:
    """Record an active frozen-calibration compatibility gate without interpreting it."""

    mode = optional_evaluator_option(arguments, "--calibration-mode")
    if mode not in {"validate", "inference"}:
        return None
    values = {
        "source_commit": os.environ.get(CALIBRATION_COMPATIBLE_SOURCE_COMMIT_ENV),
        "vit_evaluator_sha256": os.environ.get(
            CALIBRATION_COMPATIBLE_VIT_EVALUATOR_SHA256_ENV
        ),
    }
    recorded = {key: value for key, value in values.items() if value is not None}
    return recorded or None


def validate_local_checkpoint(arguments: list[str]) -> dict[str, Any] | None:
    """Bind an explicit local model path to its supplied artifact identity."""

    expected = identity.checked_hash(evaluator_option(arguments, "--checkpoint-sha256"))
    model_id = evaluator_option(arguments, "--model_id")
    path = Path(model_id).expanduser()
    if not path.exists():
        if path.is_absolute():
            raise FileNotFoundError(path)
        return None
    artifact = identity.artifact_identity(path)
    if artifact["aggregate_sha256"] != expected:
        raise ValueError("local checkpoint artifact differs from checkpoint SHA-256")
    return artifact


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

    evaluator_args = evaluator_arguments(args)
    evaluator_option(evaluator_args, "--checkpoint-sha256")
    evaluator_option(evaluator_args, "--model_id")
    linear, logarithmic = noise_profile_fractions(
        args.noise_profile,
        linear_std_frac=args.linear_std_frac,
        log_std_frac=args.log_std_frac,
    )
    entrypoint = evaluation_entrypoint(args.source_root, evaluator_args)
    command = [
        args.python_bin,
        str(entrypoint),
    ]
    if entrypoint.name == "evaluate_calibrated_vit.py":
        command.extend(("--source-root", str(args.source_root)))
    command.extend([
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
    ])
    if preflight:
        command.extend(("--gaussian-rng-preflight-path", str(contract_path)))
    else:
        if result_path is None:
            raise ValueError("a shard result path is required outside preflight")
        command.extend(("--shard-result-path", str(result_path)))
    return command


class RunnerInterrupted(Exception):
    """Record a termination signal forwarded to managed evaluator groups."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"runner received signal {signum}")
        self.signum = signum


def _signal_process_groups(
    children: Iterable[subprocess.Popen[Any]], signum: int
) -> None:
    for child in children:
        if child.poll() is not None:
            continue
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass


def _spawn_managed(
    children: list[subprocess.Popen[Any]],
    command: list[str],
    **kwargs: Any,
) -> subprocess.Popen[Any]:
    """Start and register one process group without a signal-forwarding race."""

    forwarded = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {signum: signal.getsignal(signum) for signum in forwarded}
    pending: list[int] = []

    def defer(signum: int, _frame: Any) -> None:
        pending.append(signum)

    for signum in forwarded:
        signal.signal(signum, defer)
    child: subprocess.Popen[Any] | None = None
    failure: BaseException | None = None
    try:
        child = subprocess.Popen(
            command,
            start_new_session=True,
            **kwargs,
        )
        children.append(child)
    except BaseException as error:
        failure = error
    finally:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded)
        try:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if pending:
        _signal_process_groups(children, pending[0])
        raise RunnerInterrupted(pending[0])
    if failure is not None:
        raise failure
    if child is None:
        raise RuntimeError("managed evaluator process was not created")
    return child


def _wait_and_unregister(
    children: list[subprocess.Popen[Any]], child: subprocess.Popen[Any]
) -> int:
    """Wait for one evaluator and remove its process group from the live registry."""

    returncode = child.wait()
    children.remove(child)
    return returncode


def _terminate(
    children: Iterable[subprocess.Popen[Any]], *, grace_seconds: float = 15.0
) -> None:
    """Terminate complete evaluator groups and reap every direct child."""

    processes = list(children)
    forwarded = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded)
    try:
        _signal_process_groups(processes, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        for child in processes:
            if child.poll() is None:
                try:
                    child.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
        survivors = [child for child in processes if child.poll() is None]
        _signal_process_groups(survivors, signal.SIGKILL)
        for child in processes:
            child.wait()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@contextmanager
def forward_termination_signals(
    children: list[subprocess.Popen[Any]],
) -> Iterator[None]:
    """Forward SIGTERM and SIGINT to evaluator groups during supervision."""

    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def forward(signum: int, _frame: Any) -> None:
        _signal_process_groups(children, signum)
        raise RunnerInterrupted(signum)

    for signum in previous:
        signal.signal(signum, forward)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def require_available_gpus(
    gpus: Iterable[int],
    *,
    max_attempts: int = 1,
    retry_interval_s: float = 0.0,
) -> dict[int, dict[str, Any]]:
    """Require idle physical GPUs, with optional bounded telemetry resampling."""

    requested = tuple(gpus)
    max_attempts = exact_positive_int(max_attempts, name="GPU admission attempts")
    retry_interval_s = float(
        nonnegative_number(retry_interval_s, name="GPU admission retry interval")
    )
    unavailable: dict[int, dict[str, Any]] = {}
    for attempt in range(max_attempts):
        activity = local_gpu.gpu_activity(gpu_ids=requested)
        unavailable = {
            gpu: sample
            for gpu, sample in activity.items()
            if not local_gpu.gpu_available(sample)
        }
        if not unavailable:
            return activity
        if attempt + 1 < max_attempts:
            time.sleep(retry_interval_s)
    raise RuntimeError(f"requested physical GPUs are not available: {unavailable}")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Launch two admitted GPU workers and merge only fully verified results."""

    args.source_root = args.source_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.prefix_samples = exact_positive_int(
        args.prefix_samples, name="evaluation prefix samples"
    )
    args.batch_size = exact_positive_int(args.batch_size, name="batch size")
    if args.prefix_samples < 2 * args.batch_size:
        raise ValueError("evaluation prefix must contain at least two full batches")
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists")
    if not isinstance(args.gpus, (list, tuple)) or len(args.gpus) != SHARD_COUNT:
        raise ValueError("exact evaluation requires exactly two physical GPUs")
    args.gpus = tuple(
        exact_nonnegative_int(gpu, name="physical GPU index") for gpu in args.gpus
    )
    if len(set(args.gpus)) != SHARD_COUNT:
        raise ValueError("exact evaluation requires two distinct physical GPUs")
    evaluator_args = evaluator_arguments(args)
    entrypoint = evaluation_entrypoint(args.source_root, evaluator_args)
    calibration_compatibility = calibration_compatibility_identity(evaluator_args)
    checkpoint_artifact = validate_local_checkpoint(evaluator_args)
    activity = require_available_gpus(args.gpus)

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
        "evaluation_entrypoint": str(entrypoint.relative_to(args.source_root)),
        "evaluation_entrypoint_sha256": identity.sha256_file(entrypoint),
        "evaluator_sha256": identity.sha256_file(
            args.source_root / "scripts/evaluation/error_analysis_vit.py"
        ),
        "calibration_compatibility": calibration_compatibility,
        "gpus": args.gpus,
        "gpu_activity": activity,
        "checkpoint_artifact": checkpoint_artifact,
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
    children: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        with forward_termination_signals(children):
            preflight_handle = preflight_log.open("xb")
            handles.append(preflight_handle)
            preflight = _spawn_managed(
                children,
                preflight_command,
                cwd=args.source_root,
                env=preflight_environment,
                stdout=preflight_handle,
                stderr=subprocess.STDOUT,
            )
            preflight_code = _wait_and_unregister(children, preflight)
            if preflight_code:
                raise RuntimeError(
                    f"random generator preflight failed with code {preflight_code}; "
                    f"see {preflight_log}"
                )
            if not contract_path.is_file():
                raise RuntimeError(
                    "random generator preflight did not publish its contract"
                )

            post_preflight_activity = require_available_gpus(
                args.gpus,
                max_attempts=6,
                retry_interval_s=1.0,
            )
            if validate_local_checkpoint(evaluator_args) != checkpoint_artifact:
                raise RuntimeError("local checkpoint artifact changed during preflight")
            runtime_files.new_json(
                args.output_dir / "post-preflight-admission.json",
                {
                    "gpu_activity": post_preflight_activity,
                    "checkpoint_artifact": checkpoint_artifact,
                },
            )

            shard_children: list[subprocess.Popen[Any]] = []
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
                child = _spawn_managed(
                    children,
                    command,
                    cwd=args.source_root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                shard_children.append(child)
            while True:
                codes = [child.poll() for child in shard_children]
                failures = [code for code in codes if code not in (None, 0)]
                if failures:
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
    if args.prefix_samples < 2 * args.batch_size:
        parser.error("prefix samples must contain at least two full batches")
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
    try:
        execute(parse_arguments())
    except RunnerInterrupted as error:
        raise SystemExit(128 + error.signum) from None


if __name__ == "__main__":
    main()
