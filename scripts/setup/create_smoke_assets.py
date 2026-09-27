#!/usr/bin/env python3
"""Create a tiny local ViT checkpoint and image dataset for evaluator smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from PIL import Image
import torch
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor


def create_smoke_assets(output_dir: Path) -> None:
    """Write deterministic synthetic inputs without fetching a model or dataset."""

    checkpoint_dir = output_dir / "checkpoint"
    dataset_dir = output_dir / "dataset"
    if checkpoint_dir.exists() or dataset_dir.exists():
        raise FileExistsError(f"smoke assets already exist under {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = ViTConfig(
        image_size=4,
        patch_size=2,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_labels=2,
    )
    ViTForImageClassification(config).save_pretrained(checkpoint_dir)
    ViTImageProcessor(size={"height": 4, "width": 4}).save_pretrained(checkpoint_dir)

    images = (
        Image.new("RGB", (4, 4), (60, 100, 140)),
        Image.new("RGB", (4, 4), (130, 90, 50)),
    )
    Dataset.from_dict({"image": list(images), "label": [0, 1]}).save_to_disk(
        str(dataset_dir)
    )
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Dataset: {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New output directory")
    args = parser.parse_args()
    create_smoke_assets(args.output_dir)


if __name__ == "__main__":
    main()
