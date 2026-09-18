#!/usr/bin/env python3
"""Sweep clock resolution on a fixed calibrated ViT-B subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from utils.transforms.calibration import (
    CALIBRATION_COMPATIBLE_SOURCE_COMMIT_ENV,
    CALIBRATION_COMPATIBLE_VIT_EVALUATOR_SHA256_ENV,
)


default_tag = "vit_base_clock_driven_imagenet500_theta20_float64_fine_v1"
default_window_steps_tag = (
    "vit_base_clock_driven_window_steps_imagenet500_theta20_float64_v1"
)
default_evaluation_samples = 500
default_shards = 21
ALLOWED_GPUS = (4, 5, 6, 7)
EXTENDED_GPUS = tuple(range(8))
CALIBRATION_VIT_EVALUATOR_SHA256 = (
    "b2039cd53c0142b468886552caf393edcdde7eaca4513ac49124c54fe0e9c85d"
)
calibration_safe_patch_sha256 = {
    "scripts/evaluation/error_analysis_vit.py": (
        "43905a2fcff7456db80b1c26c8fb31e287904703d5c3720282ec1ed5dd763584"
    ),
    "utils/transformers/integrations/spiking_sdpa_attention.py": (
        "4a9ebcc268b5d43de3faa5498c60bad83d6b111c76b20b33d0bc1512e1019548"
    ),
    "utils/transformers/models/spiking_ops.py": (
        "10fc31cff8d361767f28b62c81f24e3ec0a6d028d41d2b0e8f26a48ea2a59a61"
    ),
    "utils/transforms/clock.py": (
        "df5265dc323ac8a1255c0367a00d8e77607af1ac1c183040d1402eb8959a7066"
    ),
    "utils/transforms/primitive.py": (
        "2433001e0485a2b48ba40d28f97f4b0b67570150d1788104a288da7ac0b85985"
    ),
    "utils/transforms/spike_to_potential.py": (
        "35b288bf3bbc07bc4b01c59330da8b5cf9ffa4318b6172ea358c51d61af0dbbd"
    ),
}
default_time_steps = tuple(index / 100.0 for index in range(1, 11))
default_time_steps_per_window = (64, 128, 256, 512, 1024, 2048)
PYTHON = Path("/opt/conda/envs/dt/bin/python")
CHECKPOINT = Path(
    "/data/delayed-temporal/artifacts/assets/theta-selection-v1/checkpoints/"
    "vit_base_patch16_224.augreg2_in21k_ft_in1k"
)
CHECKPOINT_SHA256 = "596ea1f22f56761c30661c87310c670e4ff296729bc5de349af41ac6ef6286ff"
CALIBRATION_DATASET = Path(
    "/data/delayed-temporal/artifacts/assets/theta-selection-v1/datasets/"
    "imagenet_theta_selection_v1/train_seed0_5000"
)
CALIBRATION_FINGERPRINT = "cabf903d14d1b1ac"
EVALUATION_DATASET = Path(
    "/data/delayed-temporal/artifacts/assets/theta-selection-v1/datasets/"
    "imagenet_theta_selection_v1/validation_50000"
)
PREPROCESSING = Path("scripts/configs/vit_timm_preprocessing.json")


def prepare_evaluation_subset(
    source_path: Path,
    subset_path: Path,
    metadata_path: Path,
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Materialize and identify the first N examples of a saved validation set."""

    from datasets import Dataset, DatasetDict, load_from_disk

    if sample_count <= 0:
        raise ValueError("evaluation sample count must be positive")
    if subset_path.exists() or metadata_path.exists():
        if not subset_path.is_dir() or not metadata_path.is_file():
            raise ValueError("evaluation subset and metadata must either both exist or not")
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "source_path": str(source_path.resolve()),
            "sample_count": sample_count,
            "selection": f"validation[0:{sample_count}]",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("stored evaluation subset identity differs")
        saved = load_from_disk(str(subset_path))
        if isinstance(saved, DatasetDict):
            saved = saved["validation"]
        if not isinstance(saved, Dataset) or len(saved) != sample_count:
            raise ValueError("stored evaluation subset population differs")
        if metadata.get("fingerprint") != saved._fingerprint:
            raise ValueError("stored evaluation subset fingerprint differs")
        return metadata

    source = load_from_disk(str(source_path))
    if isinstance(source, DatasetDict):
        source = source["validation"]
    if not isinstance(source, Dataset):
        raise TypeError("evaluation source must contain a Dataset validation split")
    if sample_count > len(source):
        raise ValueError("evaluation sample count exceeds the saved validation set")
    subset = source.select(range(sample_count))
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = subset_path.with_name(f".{subset_path.name}.staging-{os.getpid()}")
    subset.save_to_disk(str(staging_path))
    staging_path.replace(subset_path)
    saved = load_from_disk(str(subset_path))
    metadata = {
        "source_path": str(source_path.resolve()),
        "source_fingerprint": source._fingerprint,
        "sample_count": sample_count,
        "selection": f"validation[0:{sample_count}]",
        "fingerprint": saved._fingerprint,
    }
    runtime_files.new_json(metadata_path, metadata)
    return metadata


def source_commit(source: Path) -> str:
    """Return the exact clean source revision used by every phase."""

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    identity.verify_clean_checkout(source, commit)
    return commit


def tasks(
    time_steps: tuple[float, ...],
    shard_count: int,
    shard_indices: tuple[int, ...] | None = None,
    *,
    time_steps_per_window: tuple[int, ...] = (),
) -> tuple[tuple[str, float | None, int | None, int], ...]:
    """Return selected shards of the baseline and one clock resolution axis."""

    if time_steps and time_steps_per_window:
        raise ValueError("clock sweep axes are mutually exclusive")
    conditions = (("continuous", None, None),) + tuple(
        (f"dt_{time_step:g}", time_step, None) for time_step in time_steps
    ) + tuple(
        (f"steps_{steps}", None, steps) for steps in time_steps_per_window
    )
    selected = normalize_shard_indices(shard_count, shard_indices)
    return tuple(
        (
            f"{condition}_shard_{shard_index:02d}",
            time_step,
            steps,
            shard_index,
        )
        for condition, time_step, steps in conditions
        for shard_index in selected
    )


def normalize_shard_indices(
    shard_count: int,
    requested: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Validate an optional deterministic subset of global shard indices."""

    if requested is None:
        return tuple(range(shard_count))
    if not requested:
        raise ValueError("selected shard indices must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("selected shard indices must be unique")
    if any(index < 0 or index >= shard_count for index in requested):
        raise ValueError("selected shard index is outside the global shard range")
    return tuple(sorted(requested))


def common_arguments(
    source: Path,
    commit: str,
    calibration_path: Path,
    evaluation_dataset_path: Path,
) -> list[str]:
    """Build the invariant noise-off calibrated ViT-B evaluation arguments."""

    return [
        "--device", "cuda",
        "--model_backend", "spiking",
        "--model_id", str(CHECKPOINT),
        "--dataset_id", "imagenet-1k",
        "--evaluation-dataset-path", str(evaluation_dataset_path),
        "--evaluation-split", "validation",
        "--image-preprocessing-config", str(source / PREPROCESSING),
        "--batch_size", "32",
        "--theta", "20",
        "--precision", "float64",
        "--source-commit", commit,
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--no-tensorboard",
        "--report-clamp-stats",
        "--spiking-layernorm",
        "--spiking-ln-mul",
        "--spiking-ln-log",
        "--spiking-ln-expdiff",
        "--spiking-attention",
        "--spiking-mlp",
        "--no-spiking-mlp-exact-gelu",
        "--no-gaussian-time-noise",
        "--time-noise-seed", "0",
        "--time-noise-std-frac", "0",
        "--time-noise-mean", "0",
        "--time-noise-deadline-margin-std", "0",
        "--no-mismatch-enabled",
        "--mismatch-theta-std", "0",
        "--mismatch-seed", "0",
        "--weight-noise-std", "0",
        "--bias-noise-std", "0",
        "--calibration-path", str(calibration_path),
        "--calibration-samples", "5000",
        "--calibration-seed", "0",
        "--calibration-bins", "2048",
        "--calibration-lower-quantile", "0",
        "--calibration-upper-quantile", "1",
        "--calibration-margin-fraction", "0.05",
    ]


def calibrated_command(
    source: Path,
    commit: str,
    calibration_path: Path,
    evaluation_dataset_path: Path,
    *,
    phase: str,
    run_id: str,
    time_step: float | None = None,
    time_steps_per_window: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[str]:
    """Build one source-frozen calibration or validation command."""

    if phase not in {"collect", "validate"}:
        raise ValueError("clock-driven campaign phase must be collect or validate")
    arguments = common_arguments(
        source, commit, calibration_path, evaluation_dataset_path
    )
    arguments += [
        "--experiment_name", run_id,
        "--calibration-mode", phase,
    ]
    if phase == "validate":
        arguments += [
            "--quick-test",
            "--evaluation-shard-index", str(shard_index),
            "--evaluation-shard-count", str(shard_count),
        ]
    if time_step is not None and time_steps_per_window is not None:
        raise ValueError("clock execution modes are mutually exclusive")
    if time_step is None and time_steps_per_window is None:
        arguments += [
            "--no-clock-driven",
            "--clock-time-step", "0",
            "--clock-time-steps-per-window", "0",
        ]
    else:
        arguments += [
            "--clock-driven",
            "--clock-time-step", (
                format(time_step, ".17g") if time_step is not None else "0"
            ),
            "--clock-time-steps-per-window", (
                str(time_steps_per_window)
                if time_steps_per_window is not None
                else "0"
            ),
        ]
    return [
        str(PYTHON),
        "-u",
        str(source / "scripts/analysis/evaluate_calibrated_vit.py"),
        "--source-root", str(source),
        "--calibration-dataset-path", str(CALIBRATION_DATASET),
        "--calibration-dataset-fingerprint", CALIBRATION_FINGERPRINT,
        "--gelu-cubic-implementation", "phi_nl_psi_ed",
        "--gelu-cubic-floor", "1e-5",
        *arguments,
    ]


def build_experiment(
    source: Path,
    commit: str,
    time_steps: tuple[float, ...],
    time_steps_per_window: tuple[int, ...],
    shard_count: int,
    *,
    tag: str,
    evaluation_dataset_path: Path,
    evaluation_metadata: dict[str, Any],
    calibration_sha256: str,
    calibration_source_commit: str,
    calibration_compatibility_paths: list[str],
    controller_commit: str,
    requested_gpus: tuple[int, ...],
    extended_gpu_pool: bool,
    selected_shards: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Create the immutable campaign identity from source and local artifacts."""

    experiment = {
        "tag": tag,
        "controller_source_root": str(SOURCE),
        "controller_source_commit": controller_commit,
        "controller_sha256": identity.sha256_file(Path(__file__)),
        "source_root": str(source),
        "source_commit": commit,
        "python_bin": str(PYTHON),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "calibration_dataset_path": str(CALIBRATION_DATASET),
        "calibration_dataset_fingerprint": CALIBRATION_FINGERPRINT,
        "calibration_sha256": calibration_sha256,
        "calibration_source_commit": calibration_source_commit,
        "calibration_compatibility_paths": calibration_compatibility_paths,
        "evaluation_dataset_path": str(evaluation_dataset_path),
        "evaluation_dataset_fingerprint": evaluation_metadata["fingerprint"],
        "evaluation_source_fingerprint": evaluation_metadata["source_fingerprint"],
        "evaluation_population": evaluation_metadata["sample_count"],
        "evaluation_selection": evaluation_metadata["selection"],
        "preprocessing_path": str(source / PREPROCESSING),
        "preprocessing_sha256": identity.sha256_file(source / PREPROCESSING),
        "theta": 20.0,
        "precision": "float64",
        "batch_size": 32,
        "time_steps": list(time_steps),
        "time_steps_per_window": list(time_steps_per_window),
        "evaluation_shards": shard_count,
        "simulation": "explicit_sequential_state_updates",
        "clock_rounding": "first_non_earlier_edge",
        "gaussian_time_noise": False,
        "mismatch": False,
        "parameter_noise": False,
        "requested_gpus": list(requested_gpus),
        "extended_gpu_pool": extended_gpu_pool,
        "evaluator_sha256": identity.sha256_file(
            source / "scripts/evaluation/error_analysis_vit.py"
        ),
        "calibration_wrapper_sha256": identity.sha256_file(
            source / "scripts/analysis/evaluate_calibrated_vit.py"
        ),
        "clock_module_sha256": identity.sha256_file(
            source / "utils/transforms/clock.py"
        ),
    }
    if selected_shards is not None:
        experiment["selected_shards"] = list(selected_shards)
    return experiment


def gpu_snapshot() -> dict[int, tuple[int, int]]:
    """Return GPU memory MiB and utilization without relying on foreign PIDs."""

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    snapshot: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        index, memory, utilization = (int(field.strip()) for field in line.split(","))
        snapshot[index] = (memory, utilization)
    return snapshot


def select_fixed_gpu_pool(
    requested: tuple[int, ...],
    *,
    allowed: tuple[int, ...],
) -> tuple[int, ...]:
    """Freeze the worker pool after two low-memory, low-utilization observations."""

    if not requested or any(gpu not in allowed for gpu in requested):
        raise ValueError(
            "clock-driven campaign GPUs must belong to the active GPU policy"
        )
    first = gpu_snapshot()
    time.sleep(10.0)
    second = gpu_snapshot()
    return tuple(
        gpu
        for gpu in requested
        if all(
            snapshot.get(gpu, (10**9, 100))[0] <= 1024
            and snapshot.get(gpu, (10**9, 100))[1] <= 5
            for snapshot in (first, second)
        )
    )


def calibration_compatibility_paths(
    source: Path,
    calibration_commit: str,
    execution_commit: str,
) -> list[str]:
    """Accept reuse only across changes unable to affect calibration values."""

    if calibration_commit == execution_commit:
        return []
    ancestor = subprocess.run(
        [
            "git", "-C", str(source), "merge-base", "--is-ancestor",
            calibration_commit, execution_commit,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("calibration source is not an ancestor of execution source")
    changed = subprocess.check_output(
        [
            "git", "-C", str(source), "diff", "--name-only",
            calibration_commit, execution_commit,
        ],
        text=True,
    ).splitlines()
    allowed = {
        "lat.md/clock-driven.md",
        "scripts/analysis/plot_clock_time_step_sweep.py",
        "scripts/experiments/run_clock_driven_vit.py",
        "scripts/verification/verify_calibration.py",
        "scripts/verification/verify_clock_driven.py",
        "utils/transforms/calibration.py",
        "utils/transforms/clock.py",
        *calibration_safe_patch_sha256,
    }
    disallowed = sorted(set(changed) - allowed)
    if disallowed:
        raise ValueError(
            "calibration reuse crosses calibration-relevant changes: "
            + ", ".join(disallowed)
        )
    for path in sorted(set(changed) & calibration_safe_patch_sha256.keys()):
        patch_bytes = subprocess.check_output(
            [
                "git", "-C", str(source), "diff",
                calibration_commit, execution_commit, "--", path,
            ]
        )
        if hashlib.sha256(patch_bytes).hexdigest() != (
            calibration_safe_patch_sha256[path]
        ):
            raise ValueError(
                f"calibration reuse has an unapproved patch for clock execution: {path}"
            )
    return sorted(changed)


def validate_calibration(
    path: Path,
    commit: str,
    *,
    source: Path,
    calibration_source_commit: str | None = None,
) -> tuple[str, str, list[str]]:
    """Validate an exact or explicitly compatible ViT-B calibration table."""

    table = json.loads(path.read_text())
    if len(table.get("layers", {})) != 109:
        raise ValueError("clock-driven ViT-B calibration requires 109 active sites")
    metadata = table["metadata"]
    if metadata.get("theta") != 20.0 or metadata.get("dtype") != "float64":
        raise ValueError("clock-driven calibration theta or dtype differs")
    options = dict(metadata["model_options"])
    table_commit = options.get("source_commit")
    expected_table_commit = calibration_source_commit or commit
    if table_commit != expected_table_commit:
        raise ValueError("clock-driven calibration source commit differs")
    compatibility_paths = calibration_compatibility_paths(
        source, expected_table_commit, commit
    )
    if options.get("calibration_dataset_fingerprint") != CALIBRATION_FINGERPRINT:
        raise ValueError("clock-driven calibration population differs")
    if options.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("clock-driven calibration checkpoint differs")
    if options.get("vit_evaluator_sha256") != CALIBRATION_VIT_EVALUATOR_SHA256:
        raise ValueError("clock-driven calibration evaluator identity differs")
    return identity.sha256_file(path), expected_table_commit, compatibility_paths


def _single(pattern: str, text: str, name: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def parse_result(
    log_path: Path,
    *,
    run_id: str,
    time_step: float | None,
    time_steps_per_window: int | None,
    shard_index: int,
    shard_count: int,
    expected_population: int,
    evaluation_dataset_path: Path,
    gpu: int,
    commit: str,
    calibration_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Accept one complete evaluator log and extract its official result."""

    text = log_path.read_text(errors="strict")
    if "Traceback (most recent call last)" in text:
        raise ValueError("clock-driven evaluator log contains a traceback")
    if f"source: disk:{evaluation_dataset_path}," not in text:
        raise ValueError("clock-driven evaluation dataset path differs")
    correct = int(_single(r"^Correct: (\d+)$", text, "correct count"))
    samples = int(_single(r"^Evaluated samples: (\d+)$", text, "sample count"))
    shard_match = re.search(
        r"^Evaluation shard — index: (?P<index>\d+), count: (?P<count>\d+), "
        r"start: (?P<start>\d+), stop: (?P<stop>\d+), population: (?P<population>\d+)$",
        text,
        flags=re.MULTILINE,
    )
    if shard_match is None:
        raise ValueError("clock-driven evaluation omitted shard identity")
    logged_shard = {name: int(value) for name, value in shard_match.groupdict().items()}
    if (
        logged_shard["index"] != shard_index
        or logged_shard["count"] != shard_count
        or logged_shard["population"] != expected_population
        or logged_shard["stop"] - logged_shard["start"] != samples
    ):
        raise ValueError("clock-driven evaluation shard identity differs")
    logged_accuracy = float(_single(r"^Accuracy: ([0-9.]+)$", text, "accuracy"))
    accuracy = correct / samples
    if (
        not math.isfinite(logged_accuracy)
        or logged_accuracy != float(f"{accuracy:.8f}")
    ):
        raise ValueError("clock-driven accuracy does not match correct/total")
    prediction_sha256 = _single(
        r"^Prediction SHA256: ([0-9a-f]{64})$", text, "prediction digest"
    )
    clock_config = re.search(
        r"^Clock-driven execution — enabled: (?P<enabled>True|False), "
        r"time_step: (?P<time_step>[0-9.eE+-]+), "
        r"time_steps_per_window: (?P<window_steps>\d+), "
        r"gaussian_time_noise: false$",
        text,
        flags=re.MULTILINE,
    )
    if clock_config is None:
        raise ValueError("clock-driven evaluation omitted its execution mode")
    expected_clock_enabled = (
        time_step is not None or time_steps_per_window is not None
    )
    if (clock_config.group("enabled") == "True") != expected_clock_enabled:
        raise ValueError("clock-driven enabled state differs")
    if (
        float(clock_config.group("time_step"))
        != (0.0 if time_step is None else time_step)
        or int(clock_config.group("window_steps"))
        != (0 if time_steps_per_window is None else time_steps_per_window)
    ):
        raise ValueError("clock-driven resolution differs")
    clock_sites: dict[str, dict[str, float | int]] = {}
    for match in re.finditer(
        r"^Clock\[(?P<site>[^]]+)\] events=(?P<events>\d+), "
        r"rounded_events=(?P<rounded>\d+), "
        r"mean_absolute_error=(?P<mean>[0-9.eE+-]+), "
        r"maximum_absolute_error=(?P<maximum>[0-9.eE+-]+), "
        r"windows=(?P<windows>\d+), "
        r"minimum_window_steps=(?P<minimum_steps>\d+), "
        r"maximum_window_steps=(?P<maximum_steps>\d+)$",
        text,
        flags=re.MULTILINE,
    ):
        clock_sites[match.group("site")] = {
            "events": int(match.group("events")),
            "rounded_events": int(match.group("rounded")),
            "mean_absolute_error": float(match.group("mean")),
            "maximum_absolute_error": float(match.group("maximum")),
            "windows": int(match.group("windows")),
            "minimum_window_steps": int(match.group("minimum_steps")),
            "maximum_window_steps": int(match.group("maximum_steps")),
        }
    clock_updates: dict[str, dict[str, int]] = {}
    for match in re.finditer(
        r"^ClockUpdates\[(?P<kind>[^]]+)\] calls=(?P<calls>\d+), "
        r"time_steps=(?P<steps>\d+), element_updates=(?P<elements>\d+)$",
        text,
        flags=re.MULTILINE,
    ):
        clock_updates[match.group("kind")] = {
            "calls": int(match.group("calls")),
            "time_steps": int(match.group("steps")),
            "element_updates": int(match.group("elements")),
        }
    if not expected_clock_enabled and clock_sites:
        raise ValueError("continuous baseline unexpectedly emitted clock statistics")
    if not expected_clock_enabled and clock_updates:
        raise ValueError("continuous baseline unexpectedly emitted clock updates")
    required_clock_sites = {"neg_linear_transform", "neg_log_transform"}
    if expected_clock_enabled and not required_clock_sites.issubset(clock_sites):
        raise ValueError("clock-driven evaluation has incomplete encoder statistics")
    if expected_clock_enabled and set(clock_updates) != {
        "encoder", "exponential", "pwm"
    }:
        raise ValueError("clock-driven evaluation has incomplete state-update counters")
    if expected_clock_enabled and any(
        counts["calls"] <= 0
        or counts["time_steps"] <= 0
        or counts["element_updates"] <= 0
        for counts in clock_updates.values()
    ):
        raise ValueError("clock-driven evaluation did not execute every state loop")
    if time_steps_per_window is not None:
        if any(
            counts["minimum_window_steps"] != time_steps_per_window
            or counts["maximum_window_steps"] != time_steps_per_window
            for counts in clock_sites.values()
        ):
            raise ValueError("encoder time window did not use the fixed step count")
        expected_steps_per_call = {
            "encoder": time_steps_per_window + 1,
            "exponential": time_steps_per_window,
            "pwm": time_steps_per_window,
        }
        if any(
            clock_updates[kind]["time_steps"]
            != clock_updates[kind]["calls"] * steps_per_call
            for kind, steps_per_call in expected_steps_per_call.items()
        ):
            raise ValueError("state update loop did not use the fixed step count")
    return {
        "run_id": run_id,
        "time_step": time_step,
        "time_steps_per_window": time_steps_per_window,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "evaluation_population": expected_population,
        "evaluation_dataset_path": str(evaluation_dataset_path),
        "shard_start": logged_shard["start"],
        "shard_stop": logged_shard["stop"],
        "clock_driven": expected_clock_enabled,
        "correct": correct,
        "samples": samples,
        "accuracy": accuracy,
        "prediction_sha256": prediction_sha256,
        "clock_sites": clock_sites,
        "clock_updates": clock_updates,
        "gpu": gpu,
        "gpu_model": _single(r"^GPU model: (.+)$", text, "GPU model"),
        "elapsed_seconds": elapsed_seconds,
        "source_commit": commit,
        "calibration_sha256": calibration_sha256,
        "log_path": str(log_path),
        "log_sha256": identity.sha256_file(log_path),
        "success": True,
    }


def write_summary(
    root: Path,
    results: list[dict[str, Any]],
    *,
    tag: str,
    time_steps: tuple[float, ...],
    time_steps_per_window: tuple[int, ...] = (),
    shard_count: int,
    expected_population: int,
) -> None:
    """Write shard evidence and summaries for every complete condition."""

    ordered = sorted(
        results,
        key=lambda row: (
            0 if not row["clock_driven"] else 1,
            row["time_step"] if row["time_step"] is not None else math.inf,
            (
                row.get("time_steps_per_window")
                if row.get("time_steps_per_window") is not None
                else math.inf
            ),
            row["shard_index"],
        ),
    )
    shard_fields = (
        "run_id", "clock_driven", "time_step", "time_steps_per_window",
        "shard_index", "shard_count",
        "shard_start", "shard_stop", "correct", "samples", "accuracy",
        "prediction_sha256", "gpu", "gpu_model", "elapsed_seconds",
        "source_commit", "calibration_sha256", "log_sha256",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=shard_fields)
    writer.writeheader()
    for row in ordered:
        writer.writerow({field: row.get(field) for field in shard_fields})
    runtime_files.atomic_text(root / "raw_shards.csv", buffer.getvalue())

    complete_conditions: list[dict[str, Any]] = []
    conditions = (("continuous", None, None),) + tuple(
        (f"dt_{value:g}", value, None) for value in time_steps
    ) + tuple(
        (f"steps_{value}", None, value) for value in time_steps_per_window
    )
    for condition, time_step, window_steps in conditions:
        shards = sorted(
            (
                row
                for row in results
                if row["time_step"] == time_step
                and row.get("time_steps_per_window") == window_steps
            ),
            key=lambda row: row["shard_index"],
        )
        if len(shards) != shard_count:
            continue
        if [row["shard_index"] for row in shards] != list(range(shard_count)):
            raise ValueError(f"duplicate or missing shard for {condition}")
        if (
            shards[0]["shard_start"] != 0
            or shards[-1]["shard_stop"] != expected_population
        ):
            raise ValueError(f"incomplete population coverage for {condition}")
        if any(
            left["shard_stop"] != right["shard_start"]
            for left, right in zip(shards, shards[1:])
        ):
            raise ValueError(f"non-contiguous shard coverage for {condition}")
        correct = sum(row["correct"] for row in shards)
        samples = sum(row["samples"] for row in shards)
        if samples != expected_population:
            raise ValueError(
                f"condition {condition} did not cover {expected_population} images"
            )
        digest_payload = "\n".join(row["prediction_sha256"] for row in shards)
        complete_conditions.append({
            "condition": condition,
            "clock_driven": time_step is not None or window_steps is not None,
            "time_step": time_step,
            "time_steps_per_window": window_steps,
            "correct": correct,
            "samples": samples,
            "accuracy": correct / samples,
            "ordered_shard_digest_sha256": hashlib.sha256(
                digest_payload.encode("ascii")
            ).hexdigest(),
            "elapsed_seconds_max": max(row["elapsed_seconds"] for row in shards),
            "shards": shard_count,
        })

    summary_fields = (
        "condition", "clock_driven", "time_step", "time_steps_per_window",
        "correct", "samples", "accuracy",
        "ordered_shard_digest_sha256", "elapsed_seconds_max", "shards",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=summary_fields)
    writer.writeheader()
    for row in complete_conditions:
        writer.writerow(row)
    runtime_files.atomic_text(root / "summary.csv", buffer.getvalue())
    runtime_files.atomic_json(
        root / "summary.json",
        {"tag": tag, "conditions": complete_conditions, "shard_runs": ordered},
    )


def next_attempt_log(root: Path, run_id: str) -> Path:
    """Return a new append-only log path without replacing prior attempts."""

    attempts = sorted((root / "logs").glob(f"{run_id}.attempt-*.log"))
    return root / "logs" / f"{run_id}.attempt-{len(attempts):02d}.log"


def recover_result_from_logs(
    root: Path,
    *,
    run_id: str,
    time_step: float | None,
    time_steps_per_window: int | None,
    shard_index: int,
    shard_count: int,
    expected_population: int,
    evaluation_dataset_path: Path,
    commit: str,
    calibration_sha256: str,
) -> dict[str, Any] | None:
    """Recover a completed evaluator whose controller stopped before acceptance."""

    for log_path in reversed(
        sorted((root / "logs").glob(f"{run_id}.attempt-*.log"))
    ):
        text = log_path.read_text(errors="strict")
        if not re.search(r"^Correct: \d+$", text, flags=re.MULTILINE):
            continue
        gpu = int(_single(r"^Controller GPU: (\d+)$", text, "controller GPU"))
        progress = re.findall(r"^Evaluation progress — (\{.+\})$", text, re.MULTILINE)
        elapsed_seconds = (
            float(json.loads(progress[-1])["elapsed_seconds"]) if progress else 0.0
        )
        return parse_result(
            log_path,
            run_id=run_id,
            time_step=time_step,
            time_steps_per_window=time_steps_per_window,
            shard_index=shard_index,
            shard_count=shard_count,
            expected_population=expected_population,
            evaluation_dataset_path=evaluation_dataset_path,
            gpu=gpu,
            commit=commit,
            calibration_sha256=calibration_sha256,
            elapsed_seconds=elapsed_seconds,
        )
    return None


def run_command(
    command: list[str],
    log_path: Path,
    *,
    gpu: int,
    source: Path,
    calibration_source_commit: str | None = None,
) -> subprocess.Popen[bytes]:
    """Launch one evaluator with one physical GPU and durable ordinary logs."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("xb")
    handle.write(f"Controller GPU: {gpu}\n".encode("utf-8"))
    handle.flush()
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if calibration_source_commit is not None:
        environment[CALIBRATION_COMPATIBLE_SOURCE_COMMIT_ENV] = (
            calibration_source_commit
        )
        environment[CALIBRATION_COMPATIBLE_VIT_EVALUATOR_SHA256_ENV] = (
            CALIBRATION_VIT_EVALUATOR_SHA256
        )
    process = subprocess.Popen(
        command,
        cwd=source,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    handle.close()
    return process


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Evaluation Contract]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--tag")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--calibration-source-commit")
    parser.add_argument(
        "--evaluation-source-path",
        type=Path,
        default=EVALUATION_DATASET,
    )
    parser.add_argument(
        "--evaluation-samples", type=int, default=default_evaluation_samples
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=list(ALLOWED_GPUS))
    parser.add_argument(
        "--allow-gpus-0-3",
        action="store_true",
        help="Campaign-only override that permits physical GPUs 0 through 3.",
    )
    parser.add_argument(
        "--time-steps",
        "--time-bins",
        dest="time_steps",
        type=float,
        nargs="+",
        default=None,
        help="Positive time-bin widths to evaluate after the continuous baseline.",
    )
    parser.add_argument(
        "--time-steps-per-window",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Positive counts of equal time steps applied independently to every "
            "declared time window."
        ),
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=default_shards,
        help="Contiguous validation shards per condition (default: 21).",
    )
    parser.add_argument(
        "--shard-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional global shard indices to run; the global shard count and "
            "range definition remain unchanged."
        ),
    )
    args = parser.parse_args()

    source = args.source_root.resolve()
    if args.time_steps is not None and args.time_steps_per_window is not None:
        raise ValueError("time-bin and steps-per-window sweeps are mutually exclusive")
    time_steps = tuple(
        float(value)
        for value in (
            default_time_steps if args.time_steps is None
            and args.time_steps_per_window is None else (args.time_steps or ())
        )
    )
    window_steps = tuple(int(value) for value in (args.time_steps_per_window or ()))
    if (
        time_steps
        and (len(set(time_steps)) != len(time_steps)
        or any(not math.isfinite(value) or value <= 0.0 for value in time_steps)
        )
    ):
        raise ValueError("time bins must be unique, finite, and strictly positive")
    if window_steps and (
        len(set(window_steps)) != len(window_steps)
        or any(value <= 0 for value in window_steps)
    ):
        raise ValueError("time steps per window must be unique and strictly positive")
    tag = args.tag or (default_window_steps_tag if window_steps else default_tag)
    root = (
        args.output_root
        or Path("/data/delayed-temporal/artifacts/logs/clock_driven") / tag
    ).resolve()
    if args.shards <= 0:
        raise ValueError("shards must be positive")
    if args.evaluation_samples < args.shards:
        raise ValueError("evaluation samples must be no smaller than shard count")
    selected_shards = normalize_shard_indices(
        args.shards,
        None if args.shard_indices is None else tuple(args.shard_indices),
    )
    explicit_shard_selection = (
        None if args.shard_indices is None else selected_shards
    )

    controller_commit = source_commit(SOURCE)
    commit = source_commit(source)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "results", "calibration", "assets"):
        (root / directory).mkdir(exist_ok=True)
    evaluation_dataset_path = (
        root / "assets" / f"validation_first_{args.evaluation_samples}"
    )
    evaluation_metadata = prepare_evaluation_subset(
        args.evaluation_source_path.resolve(),
        evaluation_dataset_path,
        root / "evaluation_subset.json",
        sample_count=args.evaluation_samples,
    )

    calibration_path = (
        args.calibration_path.resolve()
        if args.calibration_path is not None
        else root / "calibration" / "vit_base_theta20.json"
    )
    requested_gpus = tuple(args.gpus)
    allowed_gpus = EXTENDED_GPUS if args.allow_gpus_0_3 else ALLOWED_GPUS
    available = select_fixed_gpu_pool(requested_gpus, allowed=allowed_gpus)
    if not available:
        raise RuntimeError("no allowed idle GPU is available for clock-driven evaluation")
    if calibration_path.exists():
        (
            calibration_sha256,
            calibration_source_commit,
            calibration_compatibility,
        ) = validate_calibration(
            calibration_path,
            commit,
            source=source,
            calibration_source_commit=args.calibration_source_commit,
        )
    elif args.calibration_path is not None:
        raise FileNotFoundError(f"calibration table does not exist: {calibration_path}")
    else:
        if args.calibration_source_commit is not None:
            raise ValueError(
                "calibration-source-commit requires an existing calibration path"
            )
        calibration_log = next_attempt_log(root, "calibration")
        command = calibrated_command(
            source,
            commit,
            calibration_path,
            evaluation_dataset_path,
            phase="collect",
            run_id="clock_driven_calibration",
        )
        print(f"Launching calibration on GPU {available[0]}", flush=True)
        process = run_command(
            command,
            calibration_log,
            gpu=available[0],
            source=source,
        )
        if process.wait() != 0:
            raise RuntimeError(f"calibration failed; inspect {calibration_log}")
        (
            calibration_sha256,
            calibration_source_commit,
            calibration_compatibility,
        ) = validate_calibration(calibration_path, commit, source=source)

    experiment = build_experiment(
        source,
        commit,
        time_steps,
        window_steps,
        args.shards,
        tag=tag,
        evaluation_dataset_path=evaluation_dataset_path,
        evaluation_metadata=evaluation_metadata,
        calibration_sha256=calibration_sha256,
        calibration_source_commit=calibration_source_commit,
        calibration_compatibility_paths=calibration_compatibility,
        controller_commit=controller_commit,
        requested_gpus=requested_gpus,
        extended_gpu_pool=args.allow_gpus_0_3,
        selected_shards=explicit_shard_selection,
    )
    runtime_files.immutable_json(root / "experiment.json", experiment)

    accepted: list[dict[str, Any]] = []
    pending: list[tuple[str, float | None, int | None, int]] = []
    selected_tasks = tasks(
        time_steps,
        args.shards,
        selected_shards,
        time_steps_per_window=window_steps,
    )
    for run_id, time_step, steps_per_window, shard_index in selected_tasks:
        result_path = root / "results" / f"{run_id}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (
                result.get("success") is True
                and result.get("source_commit") == commit
                and result.get("calibration_sha256") == calibration_sha256
                and result.get("time_step") == time_step
                and result.get("time_steps_per_window") == steps_per_window
                and result.get("shard_index") == shard_index
                and result.get("shard_count") == args.shards
                and result.get("evaluation_population") == args.evaluation_samples
                and result.get("evaluation_dataset_path")
                == str(evaluation_dataset_path)
            ):
                accepted.append(result)
                continue
            raise ValueError(f"stored result identity differs: {result_path}")
        recovered = recover_result_from_logs(
            root,
            run_id=run_id,
            time_step=time_step,
            time_steps_per_window=steps_per_window,
            shard_index=shard_index,
            shard_count=args.shards,
            expected_population=args.evaluation_samples,
            evaluation_dataset_path=evaluation_dataset_path,
            commit=commit,
            calibration_sha256=calibration_sha256,
        )
        if recovered is not None:
            runtime_files.new_json(result_path, recovered)
            accepted.append(recovered)
            print(f"Recovered completed {run_id}", flush=True)
            continue
        pending.append((run_id, time_step, steps_per_window, shard_index))

    write_summary(
        root,
        accepted,
        tag=tag,
        time_steps=time_steps,
        time_steps_per_window=window_steps,
        shard_count=args.shards,
        expected_population=args.evaluation_samples,
    )
    running: dict[int, dict[str, Any]] = {}
    free = list(available)
    while pending or running:
        while pending and free:
            gpu = free.pop(0)
            run_id, time_step, steps_per_window, shard_index = pending.pop(0)
            log_path = next_attempt_log(root, run_id)
            command = calibrated_command(
                source,
                commit,
                calibration_path,
                evaluation_dataset_path,
                phase="validate",
                run_id=run_id,
                time_step=time_step,
                time_steps_per_window=steps_per_window,
                shard_index=shard_index,
                shard_count=args.shards,
            )
            print(f"Launching {run_id} on GPU {gpu}", flush=True)
            running[gpu] = {
                "process": run_command(
                    command,
                    log_path,
                    gpu=gpu,
                    source=source,
                    calibration_source_commit=(
                        calibration_source_commit
                        if calibration_source_commit != commit
                        else None
                    ),
                ),
                "run_id": run_id,
                "time_step": time_step,
                "time_steps_per_window": steps_per_window,
                "shard_index": shard_index,
                "log_path": log_path,
                "started": time.monotonic(),
            }
        if not running:
            continue
        time.sleep(5.0)
        for gpu, state in list(running.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise RuntimeError(
                    f"{state['run_id']} failed; inspect {state['log_path']}"
                )
            result = parse_result(
                state["log_path"],
                run_id=state["run_id"],
                time_step=state["time_step"],
                time_steps_per_window=state["time_steps_per_window"],
                shard_index=state["shard_index"],
                shard_count=args.shards,
                expected_population=args.evaluation_samples,
                evaluation_dataset_path=evaluation_dataset_path,
                gpu=gpu,
                commit=commit,
                calibration_sha256=calibration_sha256,
                elapsed_seconds=time.monotonic() - state["started"],
            )
            runtime_files.new_json(root / "results" / f"{state['run_id']}.json", result)
            accepted.append(result)
            free.append(gpu)
            del running[gpu]
            write_summary(
                root,
                accepted,
                tag=tag,
                time_steps=time_steps,
                time_steps_per_window=window_steps,
                shard_count=args.shards,
                expected_population=args.evaluation_samples,
            )
            print(
                f"Completed {state['run_id']}: {result['correct']}/{result['samples']} "
                f"({result['accuracy']:.6f})",
                flush=True,
            )

    if len(accepted) != len(selected_tasks):
        raise RuntimeError("clock-driven campaign ended without every condition")
    write_summary(
        root,
        accepted,
        tag=tag,
        time_steps=time_steps,
        time_steps_per_window=window_steps,
        shard_count=args.shards,
        expected_population=args.evaluation_samples,
    )
    print(f"Completed {tag}: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
