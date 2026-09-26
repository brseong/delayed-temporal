#!/usr/bin/env python3
"""Create immutable text datasets used by the full calibrated comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset, load_from_disk

from utils.transformers.calibration import select_calibration_subset
from scripts.runtime import identity


def tree_identity(path: Path) -> tuple[str, dict[str, str]]:
    files = {
        str(item.relative_to(path)): identity.sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(json.dumps([name, value], separators=(",", ":")).encode())
    return digest.hexdigest(), files


def save_dataset(dataset: Any, root: Path, relative: str, expected_rows: int) -> dict[str, Any]:
    if len(dataset) != expected_rows:
        raise ValueError(f"{relative} has {len(dataset)} rows, expected {expected_rows}")
    path = root / relative
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(path)
    restored = load_from_disk(path)
    if len(restored) != expected_rows or restored.column_names != dataset.column_names:
        raise ValueError(f"{relative} changed while being saved")
    digest, files = tree_identity(path)
    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "source_fingerprint": str(dataset._fingerprint),
        "fingerprint": str(restored._fingerprint),
        "aggregate_sha256": digest,
        "files_sha256": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", default="/root/.cache/huggingface/datasets")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    sst2_train = load_dataset("glue", "sst2", split="train", cache_dir=args.cache_dir)
    sst2_validation = load_dataset("glue", "sst2", split="validation", cache_dir=args.cache_dir)
    wiki_train = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=args.cache_dir,
    ).filter(lambda row: bool(row["text"].strip()))
    wiki_test = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=args.cache_dir,
    ).filter(lambda row: bool(row["text"].strip()))

    datasets = {
        "sst2_train_seed0_5000": save_dataset(
            select_calibration_subset(sst2_train, sample_count=5000, seed=0),
            root, "sst2/train_seed0_5000", 5000,
        ),
        "sst2_validation_872": save_dataset(
            sst2_validation, root, "sst2/validation_872", 872,
        ),
        "wikitext2_train_nonempty_seed0_5000": save_dataset(
            select_calibration_subset(wiki_train, sample_count=5000, seed=0),
            root, "wikitext2/train_nonempty_seed0_5000", 5000,
        ),
        "wikitext2_test_nonempty_2891": save_dataset(
            wiki_test, root, "wikitext2/test_nonempty_2891", 2891,
        ),
    }
    manifest = {
        "schema_version": 1,
        "calibration_seed": 0,
        "selection": "prefix of seeded training permutation",
        "wikitext_filter": "strip length greater than zero before selection",
        "datasets": datasets,
    }
    destination = root / "manifest.json"
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"manifest": str(destination), "datasets": len(datasets)}, sort_keys=True))


if __name__ == "__main__":
    main()
