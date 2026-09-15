"""Verify comparison dataset identity and calibration smoke isolation on CPU."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src/transformers/src"), str(ROOT / "src/spikingjelly")]

from datasets import Dataset, Features, Image, Value, load_from_disk
from PIL import Image as PILImage

from scripts.analysis.evaluate_calibrated_vit import (
    bind_metadata_identity, calibration_dataset_view, local_training_accessors,
    validate_arguments,
)
from scripts.setup.prepare_vit_comparison_assets import (
    build_model_record, image_label_sha256, save_cifar, validate_cifar,
)
from scripts.evaluation.error_analysis_vit import require_finite_logits
from utils.transforms.calibration import CalibrationMetadata, validate_calibration_metadata
from utils.transformers.models.spiking_vit.calibration import vit_calibration_specs
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification


def reject(action) -> None:
    try:
        action()
    except (ValueError, TypeError, FileExistsError):
        return
    raise AssertionError("invalid comparison input was accepted")


def verify_views() -> None:
    data = Dataset.from_dict({"img": list(range(5000)), "label": [i % 10 for i in range(5000)]})
    assert calibration_dataset_view(data, dataset_id="cifar10", expected_fingerprint=data._fingerprint) is data
    load, select = local_training_accessors(data, dataset_id="cifar10", expected_fingerprint=data._fingerprint)
    assert load("cifar10", split="train") is data
    assert select(data, sample_count=5000, seed=0) is data
    reject(lambda: load("imagenet-1k", split="train"))
    reject(lambda: load("cifar10", split="test"))
    reject(lambda: select(data, sample_count=64, seed=0))
    reject(lambda: calibration_dataset_view(data, dataset_id="imagenet-1k", expected_fingerprint=data._fingerprint))
    reject(lambda: calibration_dataset_view(data, dataset_id="cifar10", expected_fingerprint="wrong"))
    for count in (-1, 5000, True):
        reject(lambda: calibration_dataset_view(data, dataset_id="cifar10", expected_fingerprint=data._fingerprint, smoke_samples=count))
    smoke = calibration_dataset_view(data, dataset_id="cifar10", expected_fingerprint=data._fingerprint, smoke_samples=64)
    assert smoke["img"] == list(range(64))
    assert smoke._fingerprint != data._fingerprint
    load_smoke, select_smoke = local_training_accessors(smoke, dataset_id="cifar10", expected_fingerprint=smoke._fingerprint, sample_count=64)
    assert select_smoke(load_smoke("cifar10", split="train"), sample_count=64, seed=0) is smoke
    reject(lambda: calibration_dataset_view(smoke, dataset_id="cifar10", expected_fingerprint=smoke._fingerprint))


def verify_smoke_metadata() -> None:
    base = CalibrationMetadata(
        model_family="vit", model_id="model", dataset_id="cifar10", dataset_split="train",
        preprocessing="{}", dtype="float64", theta=40.0, tau_s=1.0, tau_m=1.0,
        clip_margin=1e-5, max_sequence_length=None, input_shape=(3, 224, 224), model_options=(),
    )
    final = bind_metadata_identity(base, {"calibration_purpose": "final", "calibration_source_fingerprint": "source"})
    smoke = bind_metadata_identity(base, {"calibration_purpose": "smoke", "calibration_source_fingerprint": "source"})
    reject(lambda: validate_calibration_metadata(final, smoke))
    for key, value in (
        ("vit_evaluator_sha256", "a" * 64), ("calibration_wrapper_sha256", "b" * 64),
        ("calibration_dataset_fingerprint", "dataset"), ("calibration_image_key", "img"),
    ):
        value_a = bind_metadata_identity(base, {key: value})
        value_b = bind_metadata_identity(base, {key: "different"})
        reject(lambda: validate_calibration_metadata(value_a, value_b))
    args = SimpleNamespace(
        model_backend="spiking", dataset_id="cifar10", calibration_mode="inference",
        calibration_path="table.json", calibration_samples=5000, calibration_seed=0,
        model_id=str(ROOT), evaluation_dataset_path="data", checkpoint_sha256="a" * 64,
        max_eval_batches=0,
    )
    validate_arguments(args, "phi_nl_psi_ed", 1e-5)
    reject(lambda: validate_arguments(args, "phi_nl_psi_ed", 1e-5, smoke_samples=64))
    args.calibration_samples = 64
    reject(lambda: validate_arguments(args, "phi_nl_psi_ed", 1e-5, smoke_samples=64))
    args.max_eval_batches = 2
    validate_arguments(args, "phi_nl_psi_ed", 1e-5, smoke_samples=64)
    reject(lambda: validate_arguments(args, "phi_nl_psi_ed", 1e-5))


def verify_saved_order() -> None:
    features = Features({"img": Image(), "label": Value("int64")})
    images = [PILImage.new("RGB", (4, 4), (i * 19, 0, 0)) for i in range(10)]
    dataset = Dataset.from_dict({"img": images, "label": list(range(10))}, features=features)
    validate_cifar(dataset, expected_samples=10)
    digest = image_label_sha256(dataset, "img")
    assert image_label_sha256(dataset.select(list(reversed(range(10)))), "img") != digest
    with tempfile.TemporaryDirectory(prefix="vit-comparison-data-test-") as temporary:
        path = Path(temporary) / "dataset"
        record = save_cifar(dataset, path, split="test")
        reloaded = load_from_disk(str(path))
        assert record["fingerprint"] == reloaded._fingerprint
        assert record["image_label_sha256"] == digest
        assert reloaded["label"] == list(range(10))
        reject(lambda: save_cifar(dataset, path, split="test"))
    reject(lambda: validate_cifar(dataset, expected_samples=9))


def verify_dynamic_sites() -> None:
    for depth, classes in ((12, 10), (12, 1000), (24, 1000)):
        config = ViTConfig(
            hidden_size=12, num_hidden_layers=depth, num_attention_heads=3,
            intermediate_size=24, image_size=16, patch_size=8, num_labels=classes,
            theta=40.0, use_spiking_attention=True, use_spiking_mlp=True,
            use_spiking_layernorm=True,
        )
        model = ViTForImageClassification(config)
        # Discovery follows the installed evaluator attention backend, not a flag.
        config._attn_implementation = "spiking_sdpa"
        specs = vit_calibration_specs(model, lower_quantile=0.0, upper_quantile=1.0, margin_fraction=0.05)
        assert len(specs) == 4 * depth, (depth, len(specs))


def verify_model_record() -> None:
    checkpoint = {"path": "model", "aggregate_sha256": "a" * 64, "config": {"num_hidden_layers": 24}}
    train = {"path": "train", "aggregate_sha256": "b" * 64, "fingerprint": "train-fp"}
    evaluation = {
        "path": "val50k", "aggregate_sha256": "c" * 64, "fingerprint": "val50k-fp",
        "evaluated_fingerprint": "val5k-fp", "evaluated_samples": 5000,
        "split": "validation", "quick_test": True,
    }
    row = build_model_record("imagenet_vit_large", "imagenet-1k", "ViT-L/16", "model-id", checkpoint, {"train": train, "evaluation": evaluation})
    assert row["dataset_fingerprint"] == "val5k-fp"
    assert row["dataset_path"] == "val50k" and row["evaluation_quick_test"]
    assert row["expected_samples"] == 5000


def verify_finite_logits() -> None:
    import torch
    logits = torch.tensor([[0.0, -1.0, 3.0], [-10.0, 2.0, 1.0]], dtype=torch.float64)
    original = logits.clone()
    require_finite_logits(logits)
    assert torch.equal(logits, original)
    assert logits.argmax(-1).tolist() == [2, 1]
    for value in (float("nan"), float("inf"), -float("inf")):
        invalid = logits.clone()
        invalid[1, 1] = value
        reject(lambda: require_finite_logits(invalid))


def main() -> None:
    for check in (verify_views, verify_smoke_metadata, verify_saved_order, verify_dynamic_sites, verify_model_record, verify_finite_logits):
        check()
    print("Comparison data and evaluator verification passed (6 groups).")


if __name__ == "__main__":
    main()
