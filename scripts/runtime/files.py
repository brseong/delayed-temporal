"""Filesystem primitives shared by independent experiment controllers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def absolute_path(value: Any) -> Path:
    """Validate an absolute path received from an experiment manifest."""
    if not isinstance(value, str):
        raise ValueError("An absolute path is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or any(c in value for c in "\n\r\0,:"):
        raise ValueError(f"Unsafe absolute path: {value!r}")
    return path


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def immutable(path: Path, content: bytes) -> None:
    """Create one immutable evidence file or accept identical existing bytes."""
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f"Refusing to replace a different preparation file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != content:
                raise ValueError(f"Refusing to replace a different preparation file: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable_json(path: Path, value: Any) -> None:
    """Create immutable canonical JSON evidence."""
    immutable(path, json_bytes(value))


def new_json(path: Path, value: Any) -> None:
    """Create one durable JSON record and reject every existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def safe_output(root: Path, relative: str) -> Path:
    """Resolve a manifest-relative output while containing it below its root."""
    path = Path(relative)
    target = (root / path).resolve()
    if path.is_absolute() or not target.is_relative_to(root.resolve()) or ".." in path.parts:
        raise ValueError("Task output must remain inside its experiment directory")
    return target


def atomic_json(path: Path, value: Any) -> None:
    """Durably replace one JSON file without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, content: str) -> None:
    """Durably replace one UTF-8 text file without exposing partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
