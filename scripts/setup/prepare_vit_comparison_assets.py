#!/usr/bin/env python3
"""Export cached CIFAR data and identify existing ViT comparison artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import Dataset, Image, load_from_disk

from scripts.setup.hash_artifact import artifact_identity, hash_file
from scripts.setup.prepare_imagenet_theta_selection import label_sha256, save_artifact
from utils.transformers.calibration import select_calibration_subset


CIFAR_ID = "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
CIFAR_REVISION = "ff3c9332ba8094d9ed467a959eb683647f87127d"
CIFAR_CACHE = Path(
    "/root/.cache/huggingface/datasets/cifar10/plain_text/0.0.0/"
    "0b2714987fa478483af9968de7c934580d0bb9a2"
)


def image_label_sha256(dataset: Dataset, image_key: str) -> str:
    """Hash decoded RGB pixels and ordered labels, independent of image storage paths."""
    digest = hashlib.sha256()
    for row in dataset:
        value = row[image_key].convert("RGB")
        digest.update(int(row["label"]).to_bytes(8, "little", signed=True))
        digest.update(value.width.to_bytes(8, "little"))
        digest.update(value.height.to_bytes(8, "little"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_cifar(dataset: Dataset, *, expected_samples: int) -> None:
    if len(dataset) != expected_samples or not {"img", "label"}.issubset(dataset.column_names):
        raise ValueError("cached CIFAR split has unexpected columns or sample count")
    if any(not 0 <= int(label) < 10 for label in dataset["label"]):
        raise ValueError("CIFAR label is outside the ten classes")


def save_cifar(dataset: Dataset, path: Path, *, split: str) -> dict[str, Any]:
    """Save embedded image payloads and prove that the exact image/label order replays."""
    original_digest = image_label_sha256(dataset, "img")
    record = save_artifact(dataset, path)
    reloaded = load_from_disk(str(path))
    if image_label_sha256(reloaded, "img") != original_digest:
        raise ValueError("saved CIFAR images or labels changed")
    # An exported artifact must not rely on external paths in the old cache.
    raw = reloaded.cast_column("img", Image(decode=False))
    if any(not row["img"]["bytes"] for row in raw):
        raise ValueError("CIFAR export contains an external image reference")
    return {**record, "split": split, "image_key": "img", "image_label_sha256": original_digest}


def export_cifar(cache: Path, output: Path) -> dict[str, Any]:
    train_file = cache / "cifar10-train.arrow"
    test_file = cache / "cifar10-test.arrow"
    train = Dataset.from_file(str(train_file))
    test = Dataset.from_file(str(test_file))
    validate_cifar(train, expected_samples=50000)
    validate_cifar(test, expected_samples=10000)
    selected = select_calibration_subset(train, sample_count=5000, seed=0)
    # Replay from the source once to check the exact seeded selection contract.
    replay = train.shuffle(seed=0).select(range(5000))
    if selected._fingerprint != replay._fingerprint or label_sha256(selected) != label_sha256(replay):
        raise ValueError("CIFAR training selection differs from the maintained selector")
    return {
        "source_files": {
            "train": {"path": str(train_file), "sha256": hash_file(train_file)},
            "test": {"path": str(test_file), "sha256": hash_file(test_file)},
        },
        "train": {**save_cifar(selected, output / "train_seed0_5000", split="train"), "selection_seed": 0},
        "evaluation": {
            **save_cifar(test, output / "test_10000", split="test"),
            "evaluated_samples": 10000, "quick_test": False,
        },
    }


def identify_imagenet(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    result = {}
    for name, old_key, directory, count in (
        ("train", "train_selection", "train_seed0_5000", 5000),
        ("evaluation", "validation", "validation_50000", 50000),
    ):
        path = root / directory
        dataset = load_from_disk(str(path))
        old = manifest[old_key]
        if not isinstance(dataset, Dataset) or len(dataset) != count or dataset._fingerprint != old["fingerprint"]:
            raise ValueError("existing ImageNet dataset identity changed")
        if label_sha256(dataset) != old["label_sha256"]:
            raise ValueError("existing ImageNet label order changed")
        identity = artifact_identity(path)
        if identity["aggregate_sha256"] != old["aggregate_sha256"]:
            raise ValueError("existing ImageNet artifact SHA-256 changed")
        result[name] = {**old, **identity, "image_key": "image", "split": old["source_split"]}
        if name == "evaluation":
            prefix = dataset.select(range(5000))
            if prefix._fingerprint != old["quick_prefix_fingerprint"]:
                raise ValueError("existing ImageNet validation prefix changed")
            result[name].update(
                evaluated_samples=5000, quick_test=True,
                evaluated_fingerprint=prefix._fingerprint,
                evaluated_label_sha256=label_sha256(prefix),
            )
    return result


def identify_checkpoint(path: Path, *, classes: int, width: int, layers: int) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text())
    if (config["hidden_size"], config["num_hidden_layers"]) != (width, layers):
        raise ValueError("checkpoint model dimensions do not match the declared architecture")
    if len(config.get("id2label", {})) != classes:
        raise ValueError("checkpoint class count does not match the dataset")
    if (config.get("image_size"), config.get("patch_size"), config.get("num_channels")) != (224, 16, 3):
        raise ValueError("comparison checkpoint must use 224-pixel RGB images and 16-pixel patches")
    if not (path / "preprocessor_config.json").is_file():
        raise ValueError("checkpoint image processor is missing")
    if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("checkpoint weights are missing")
    return {**artifact_identity(path), "config": config}


def build_model_record(
    key: str, task: str, architecture: str, checkpoint_id: str,
    checkpoint: dict[str, Any], datasets: dict[str, Any],
) -> dict[str, Any]:
    train, evaluation = datasets["train"], datasets["evaluation"]
    return {
        "model_key": key, "task": task, "architecture": architecture,
        "checkpoint_id": checkpoint_id, "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["aggregate_sha256"], "checkpoint_config": checkpoint["config"],
        "calibration_dataset_path": train["path"],
        "calibration_dataset_fingerprint": train["fingerprint"],
        "calibration_dataset_sha256": train["aggregate_sha256"],
        "dataset_path": evaluation["path"],
        "dataset_fingerprint": evaluation.get("evaluated_fingerprint", evaluation["fingerprint"]),
        "dataset_sha256": evaluation["aggregate_sha256"],
        "expected_samples": evaluation["evaluated_samples"],
        "evaluation_split": evaluation["split"], "evaluation_quick_test": evaluation["quick_test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/assets/vit-conversion-comparison-v1")
    parser.add_argument("--cifar-cache", type=Path, default=CIFAR_CACHE)
    parser.add_argument(
        "--cifar-checkpoint", type=Path,
        default=Path("/root/.cache/huggingface/hub/models--MF21377197--vit-small-patch16-224-finetuned-Cifar10/snapshots") / CIFAR_REVISION,
    )
    parser.add_argument(
        "--imagenet-root", type=Path,
        default=ROOT / "artifacts/assets/theta-selection-v1/datasets/imagenet_theta_selection_v1",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/data/nas"))
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("refusing to replace existing comparison assets")
    # Finish read-only verification of shared assets before creating any output.
    imagenet = identify_imagenet(args.imagenet_root.resolve())
    specifications = (
        ("imagenet_vit_small", "ViT-S/16", "vit_small_patch16_224.augreg_in21k_ft_in1k", 384, 12),
        ("imagenet_vit_base", "ViT-B/16", "vit_base_patch16_224.augreg2_in21k_ft_in1k", 768, 12),
        ("imagenet_vit_large", "ViT-L/16", "vit_large_patch16_224.augreg_in21k_ft_in1k", 1024, 24),
    )
    checkpoints = {
        key: identify_checkpoint(args.checkpoint_root / filename, classes=1000, width=width, layers=layers)
        for key, _, filename, width, layers in specifications
    }
    identify_checkpoint(args.cifar_checkpoint, classes=10, width=384, layers=12)
    output.mkdir(parents=True)
    cifar = export_cifar(args.cifar_cache.resolve(), output / "datasets/cifar10")
    cifar_path = output / "checkpoints/vit_small_patch16_224_cifar10"
    shutil.copytree(args.cifar_checkpoint, cifar_path, symlinks=False)
    cifar_checkpoint = identify_checkpoint(cifar_path, classes=10, width=384, layers=12)
    checkpoints["cifar10_vit_small"] = cifar_checkpoint
    models = [build_model_record(
        "cifar10_vit_small", "cifar10", "ViT-S/16", CIFAR_ID, cifar_checkpoint, cifar,
    )]
    for key, architecture, filename, _, _ in specifications:
        models.append(build_model_record(key, "imagenet-1k", architecture, filename, checkpoints[key], imagenet))
    manifest = {
        "format_version": 1, "cifar_checkpoint_revision": CIFAR_REVISION,
        "datasets": {"imagenet-1k": imagenet, "cifar10": cifar},
        "checkpoints": checkpoints, "models": models,
    }
    path = output / "manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(path)


if __name__ == "__main__":
    main()
