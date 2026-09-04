#!/usr/bin/env python3
"""Create a stable file-level identity manifest for a file or directory artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    """Hash path names, sizes, and contents in deterministic relative order."""

    resolved = path.resolve()
    if resolved.is_file():
        files = [resolved]
        root = resolved.parent
    elif resolved.is_dir():
        files = sorted(item for item in resolved.rglob("*") if item.is_file())
        root = resolved
    else:
        raise FileNotFoundError(resolved)
    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        digest = hash_file(item)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        records.append({"path": relative, "bytes": size, "sha256": digest})
    return {
        "format_version": 1,
        "path": str(resolved),
        "aggregate_sha256": aggregate.hexdigest(),
        "bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identity = artifact_identity(args.path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(identity["aggregate_sha256"])


if __name__ == "__main__":
    main()
