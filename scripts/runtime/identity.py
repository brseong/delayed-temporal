"""Content identities shared by setup commands and experiment controllers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


IGNORED_SOURCE_DIRECTORIES = {".git", "__pycache__", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 identity of an in-memory byte sequence."""
    return hashlib.sha256(content).hexdigest()


def checked_hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA-256 identity")
    return value


def json_sha256(value: Any) -> str:
    """Hash one JSON-compatible identity with its canonical compact encoding."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def model_state_sha256(model: Any) -> str:
    """Hash the exact named tensor state loaded into one PyTorch model."""

    import torch

    state = model.state_dict()
    if not isinstance(state, dict):
        raise TypeError("model state must be a dictionary")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("model state names and values must be strings and tensors")
        if tensor.layout is not torch.strided or tensor.device.type == "meta":
            raise ValueError("model state hash requires materialized strided tensors")
        value = tensor.detach().cpu().contiguous()
        descriptor = json.dumps(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    """Hash relative path names, sizes, and contents in deterministic order."""
    resolved = path.resolve()
    if resolved.is_file():
        files, root = [resolved], resolved.parent
    elif resolved.is_dir():
        files, root = sorted(item for item in resolved.rglob("*") if item.is_file()), resolved
    else:
        raise FileNotFoundError(resolved)
    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        digest = sha256_file(item)
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
        records.append({"path": relative, "bytes": size, "sha256": digest})
    return {
        "format_version": 1,
        "path": str(resolved),
        "aggregate_sha256": aggregate.hexdigest(),
        "bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def artifact_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Return the portable artifact hash with absolute immutable-file metadata."""
    identity = artifact_identity(path)
    if not identity["files"]:
        raise ValueError(f"Empty artifact: {identity['path']}")
    resolved = Path(identity["path"])
    root = resolved if resolved.is_dir() else resolved.parent
    records = []
    for record in identity["files"]:
        item = root / record["path"]
        stat = item.stat()
        records.append({
            "path": str(item),
            "bytes": record["bytes"],
            "mtime_ns": stat.st_mtime_ns,
            "sha256": record["sha256"],
        })
    return identity["aggregate_sha256"], records


def package_source_files(path: Path) -> list[Path]:
    """Enumerate importable source without transient caches or Git metadata."""
    if not path.is_dir():
        raise ValueError(f"Package source must be a directory: {path}")
    files = []
    for current, directories, names in os.walk(path, followlinks=False):
        root = Path(current)
        directories[:] = sorted(name for name in directories if name not in IGNORED_SOURCE_DIRECTORIES)
        if any((root / name).is_symlink() for name in directories):
            raise ValueError("Package source directory symlinks require an explicit source root")
        files.extend(
            root / name
            for name in names
            if name not in IGNORED_SOURCE_DIRECTORIES and not name.endswith(".pyc")
        )
    return sorted(files)


def package_source_identity(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash source names, sizes, and bytes independently of installation path."""
    files = package_source_files(path)
    if not files:
        raise ValueError(f"Empty package source: {path}")
    aggregate = hashlib.sha256()
    records = []
    for item in files:
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest = sha256_file(item)
        aggregate.update(f"{relative}\0{stat.st_size}\0{digest}\n".encode())
        records.append({
            "path": str(item),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        })
    return aggregate.hexdigest(), records


def verify_clean_checkout(source: Path, expected_commit: str) -> None:
    """Require one Git checkout at the recorded commit without tracked edits."""
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if head != expected_commit or dirty.strip():
        raise ValueError("Source root must have its frozen HEAD and no tracked modifications")


def verify_file_identities(files: dict[Path, str]) -> None:
    """Reject missing or changed files from a frozen experiment identity."""
    for path, expected in files.items():
        if sha256_file(path) != checked_hash(expected):
            raise ValueError(f"Frozen file content differs: {path}")


def verify_package_identities(packages: dict[Path, str]) -> None:
    """Reject editable package trees whose portable source identity changed."""
    for path, expected in packages.items():
        if package_source_identity(path)[0] != checked_hash(expected):
            raise ValueError(f"Editable dependency source differs: {path.name}")
