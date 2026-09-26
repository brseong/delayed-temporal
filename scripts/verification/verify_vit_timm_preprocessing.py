#!/usr/bin/env python3
"""Verify the immutable timm transform used by the ImageNet ViT comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from timm.data import create_transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.error_analysis_vit import load_vit_image_processor
from utils.transformers.models.spiking_vit.calibration import image_processor_pixel_bounds


CONFIG = ROOT / "scripts/configs/vit_timm_preprocessing.json"
MODELS = (
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
)


def reject(payload: dict, model: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(payload))
        try:
            load_vit_image_processor(f"/checkpoint/{model}", str(path))
        except ValueError:
            return
    raise AssertionError("invalid preprocessing configuration was accepted")


def main() -> None:
    payload = json.loads(CONFIG.read_text())
    assert payload["schema_version"] == 1
    assert set(payload["models"]) == set(MODELS)
    image = Image.fromarray(
        np.arange(300 * 280 * 3, dtype=np.uint8).reshape(300, 280, 3)
    )
    for model in MODELS:
        processor = load_vit_image_processor(f"/checkpoint/{model}", str(CONFIG))
        actual = processor([image], return_tensors="pt")["pixel_values"]
        expected = create_transform(
            input_size=(3, 224, 224), is_training=False,
            interpolation="bicubic", mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5), crop_pct=0.9, crop_mode="center",
        )(image).unsqueeze(0)
        assert actual.shape == (1, 3, 224, 224)
        assert torch.equal(actual, expected)
        bounds = image_processor_pixel_bounds(processor, num_channels=3)
        assert bounds.min == -1.0 and bounds.max == 1.0

    malformed = json.loads(CONFIG.read_text())
    malformed["models"][MODELS[0]]["crop_pct"] = 1.0
    reject(malformed, MODELS[0])
    malformed = json.loads(CONFIG.read_text())
    del malformed["models"][MODELS[0]]
    reject(malformed, MODELS[0])
    print("ViT timm preprocessing verification passed")


if __name__ == "__main__":
    main()
