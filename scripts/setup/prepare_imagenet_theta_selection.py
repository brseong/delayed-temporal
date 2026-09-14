#!/usr/bin/env python3
"""Export self-contained ImageNet artifacts for the UBAI theta selection sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets import Dataset, load_dataset, load_from_disk

from utils.transformers.calibration import select_calibration_subset


def directory_sha256(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash every regular file in a saved dataset with stable relative ordering."""

    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
        size = item.stat().st_size
        value = file_digest.hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
        files.append({"path": relative, "bytes": size, "sha256": value})
    return digest.hexdigest(), files


def label_sha256(dataset: Dataset) -> str:
    """Hash the ordered ImageNet label stream without decoding image payloads."""

    digest = hashlib.sha256()
    for label in dataset["label"]:
        digest.update(int(label).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def save_artifact(dataset: Dataset, path: Path) -> dict[str, Any]:
    """Save one immutable dataset and return its identity metadata."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite dataset artifact: {path}")
    dataset.save_to_disk(str(path))
    reloaded = load_from_disk(str(path))
    if not isinstance(reloaded, Dataset) or len(reloaded) != len(dataset):
        raise RuntimeError(f"saved dataset did not replay exactly: {path}")
    source_label_sha256 = label_sha256(dataset)
    if label_sha256(reloaded) != source_label_sha256:
        raise RuntimeError(f"saved dataset changed label order: {path}")
    aggregate_sha256, files = directory_sha256(path)
    return {
        "path": str(path.resolve()),
        "samples": len(dataset),
        "source_fingerprint": dataset._fingerprint,
        "fingerprint": reloaded._fingerprint,
        "label_sha256": source_label_sha256,
        "aggregate_sha256": aggregate_sha256,
        "bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/nas/datasets"))
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--train-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    if args.train_samples <= 0 or args.train_seed < 0:
        raise ValueError("train samples must be positive and seed non-negative")

    args.output_root.mkdir(parents=True)
    train = load_dataset(
        "imagenet-1k",
        split="train",
        cache_dir=str(args.cache_dir),
    )
    train_selection = select_calibration_subset(
        train,
        sample_count=args.train_samples,
        seed=args.train_seed,
    )
    validation = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir=str(args.cache_dir),
    )

    validation_record = save_artifact(
        validation,
        args.output_root / "validation_50000",
    )
    validation_reloaded = load_from_disk(str(args.output_root / "validation_50000"))
    if not isinstance(validation_reloaded, Dataset):
        raise TypeError("saved validation artifact is not a Dataset")
    validation_record["quick_prefix_fingerprint"] = validation_reloaded.select(
        range(min(5000, len(validation_reloaded)))
    )._fingerprint

    manifest = {
        "format_version": 1,
        "dataset_id": "imagenet-1k",
        "train_selection": {
            "source_split": "train",
            "selection_seed": args.train_seed,
            **save_artifact(train_selection, args.output_root / "train_seed0_5000"),
        },
        "validation": {
            "source_split": "validation",
            "quick_prefix_samples": 5000,
            **validation_record,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
