"""Content-addressed cache for dense ANN evaluation evidence.

The cache identity deliberately excludes the converted-model source commit.  A
change confined to the SNN implementation must not spend another GPU pass on
an unchanged dense model, dataset, preprocessing, and metric contract.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Callable

from scripts.runtime import files as runtime_files
from scripts.runtime import identity


SCHEMA_VERSION = 1
DENSE_EVALUATION_CONTRACT_VERSION = 1


def portable_dataset_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove only the host-local path from an authenticated dataset snapshot."""
    result = {key: value for key, value in snapshot.items() if key != "path"}
    if not result.get("fingerprint") or not isinstance(result.get("samples"), int):
        raise ValueError("ANN baseline dataset identity is incomplete")
    return result


def build_identity(
    *,
    model_key: str,
    model_family: str,
    checkpoint_identity: Any,
    evaluation_dataset: dict[str, Any],
    evaluation_settings: dict[str, Any],
) -> dict[str, Any]:
    """Build the source-commit-independent identity of one dense evaluation."""
    if not model_key or not model_family or not checkpoint_identity:
        raise ValueError("ANN baseline model identity is incomplete")
    if not evaluation_settings:
        raise ValueError("ANN baseline evaluation settings are incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "dense_evaluation_contract_version": DENSE_EVALUATION_CONTRACT_VERSION,
        "model_key": model_key,
        "model_family": model_family,
        "checkpoint_identity": checkpoint_identity,
        "evaluation_dataset": portable_dataset_identity(evaluation_dataset),
        "evaluation_settings": evaluation_settings,
    }


def identity_sha256(baseline_identity: dict[str, Any]) -> str:
    """Return the content address for a dense evaluation identity."""
    if baseline_identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("ANN baseline schema differs")
    if (
        baseline_identity.get("dense_evaluation_contract_version")
        != DENSE_EVALUATION_CONTRACT_VERSION
    ):
        raise ValueError("ANN dense evaluation contract differs")
    return identity.json_sha256(baseline_identity)


def baseline_directory(cache_root: Path, baseline_identity: dict[str, Any]) -> Path:
    """Resolve the immutable directory for one dense evaluation identity."""
    return cache_root / identity_sha256(baseline_identity)


def publish(
    *,
    cache_root: Path,
    baseline_identity: dict[str, Any],
    source_log: Path,
    metrics: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Publish an authenticated completed ANN phase, accepting identical races."""
    directory = baseline_directory(cache_root, baseline_identity)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "evaluation.log"
    record_path = directory / "baseline.json"
    with (directory / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (
                record.get("identity") != baseline_identity
                or record.get("metrics") != metrics
                or not log_path.is_file()
                or identity.sha256_file(log_path) != record.get("log_sha256")
            ):
                raise ValueError("existing ANN baseline evidence differs")
            return record
        log_bytes = source_log.read_bytes()
        runtime_files.immutable(log_path, log_bytes)
        record = {
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "identity": baseline_identity,
            "identity_sha256": identity_sha256(baseline_identity),
            "log_file": "evaluation.log",
            "log_sha256": identity.sha256_bytes(log_bytes),
            "metrics": metrics,
            "elapsed_seconds": float(elapsed_seconds),
        }
        runtime_files.immutable_json(record_path, record)
        return record


def load(
    *,
    cache_root: Path,
    baseline_identity: dict[str, Any],
    parse_metrics: Callable[[str], dict[str, Any]],
) -> tuple[dict[str, Any], Path] | None:
    """Load and re-parse a matching immutable ANN baseline when it exists."""
    directory = baseline_directory(cache_root, baseline_identity)
    record_path = directory / "baseline.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("state") != "complete"
        or record.get("identity") != baseline_identity
        or record.get("identity_sha256") != identity_sha256(baseline_identity)
        or record.get("log_file") != "evaluation.log"
    ):
        raise ValueError("cached ANN baseline identity differs")
    log_path = directory / record["log_file"]
    if not log_path.is_file() or identity.sha256_file(log_path) != record.get("log_sha256"):
        raise ValueError("cached ANN baseline log differs")
    parsed = parse_metrics(log_path.read_text(errors="replace"))
    if parsed != record.get("metrics"):
        raise ValueError("cached ANN baseline metrics differ from its log")
    return record, log_path


def materialize_phase(
    *,
    record: dict[str, Any],
    source_log: Path,
    output: Path,
) -> dict[str, Any]:
    """Copy cached evidence into a result bundle without performing inference."""
    if identity.sha256_file(source_log) != record.get("log_sha256"):
        raise ValueError("materialized ANN baseline log differs")
    digest = record["identity_sha256"]
    relative = Path("logs") / f"ann.reused-{digest[:16]}.log"
    destination = output / relative
    runtime_files.immutable(destination, source_log.read_bytes())
    return {
        "phase": "ann",
        "elapsed_seconds": 0.0,
        "source_elapsed_seconds": record["elapsed_seconds"],
        "log_file": str(relative),
        "log_sha256": record["log_sha256"],
        "metrics": record["metrics"],
        "ann_baseline_reuse": {
            "identity_sha256": digest,
            "baseline_record_sha256": identity.json_sha256(record),
        },
    }
