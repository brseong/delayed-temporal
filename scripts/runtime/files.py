"""Filesystem primitives shared by independent experiment controllers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    """Durably replace one JSON file without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

