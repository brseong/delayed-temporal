"""Evaluate ViT with local training calibration data and a frozen source checkout."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from scripts.evaluation.error_analysis_vit import Arguments
    from utils.transforms.calibration import CalibrationMetadata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import identity


CALIBRATION_SAMPLES = 5000
CALIBRATION_SEED = 0
DATASET_IMAGE_KEYS = {"imagenet-1k": "image", "cifar10": "img"}


def validate_source(source: Path, expected_commit: str) -> None:
    """Refuse a different commit or tracked changes before importing its modules."""
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != expected_commit:
        raise ValueError("source commit does not match the frozen checkout")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty.strip():
        raise ValueError("frozen source has tracked modifications")


def validate_local_training_dataset(
    dataset: Dataset, *, expected_fingerprint: str,
    dataset_id: str = "imagenet-1k", sample_count: int = CALIBRATION_SAMPLES,
) -> Dataset:
    """Validate an already selected training population without another shuffle."""
    if dataset_id not in DATASET_IMAGE_KEYS:
        raise ValueError("unsupported local calibration dataset")
    if len(dataset) != sample_count:
        raise ValueError("calibration artifact sample count mismatch")
    if not expected_fingerprint or dataset._fingerprint != expected_fingerprint:
        raise ValueError("calibration dataset fingerprint mismatch")
    return dataset


def calibration_dataset_view(
    dataset: Dataset, *, dataset_id: str, expected_fingerprint: str, smoke_samples: int = 0,
) -> Dataset:
    """Verify the complete artifact before taking an explicitly requested smoke prefix."""
    if dataset_id not in DATASET_IMAGE_KEYS:
        raise ValueError("unsupported local calibration dataset")
    if isinstance(smoke_samples, bool) or not isinstance(smoke_samples, int):
        raise TypeError("calibration smoke samples must be an integer")
    if not 0 <= smoke_samples < CALIBRATION_SAMPLES:
        raise ValueError("calibration smoke samples must be zero or between 1 and 4999")
    if len(dataset) != CALIBRATION_SAMPLES or dataset._fingerprint != expected_fingerprint:
        raise ValueError("calibration requires the verified complete training 5000 artifact")
    if not {DATASET_IMAGE_KEYS[dataset_id], "label"}.issubset(dataset.column_names):
        raise ValueError("calibration artifact has incorrect image or label columns")
    return dataset.select(range(smoke_samples)) if smoke_samples else dataset


def bind_metadata_identity(
    metadata: CalibrationMetadata,
    identity: Mapping[str, str | int | float | bool | None],
) -> CalibrationMetadata:
    """Extend the frozen metadata with the implementation and artifact identities."""
    options = dict(metadata.model_options)
    for key, value in identity.items():
        if key in options and options[key] != value:
            raise ValueError(f"calibration metadata identity conflict: {key}")
        options[key] = value
    return replace(metadata, model_options=tuple(sorted(options.items())))


def make_progress_reporter(batches_per_pass: int) -> Callable[[int, int], None]:
    """Report explicit calibration batch completions without model hooks."""
    if batches_per_pass <= 0:
        raise ValueError("calibration loader must contain batches")
    total_batches = 2 * batches_per_pass
    completed_batches = 0
    completed_samples = 0
    started = time.monotonic()

    def report(pass_index: int, batch_size: int) -> None:
        nonlocal completed_batches, completed_samples
        completed_batches += 1
        completed_samples += batch_size
        if completed_batches % 10 == 0 or completed_batches == total_batches:
            print("Calibration progress — " + json.dumps({
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "completed_samples": completed_samples,
                "pass": pass_index + 1,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }, sort_keys=True), flush=True)

    return report


def validate_arguments(
    args: Arguments, *, smoke_samples: int = 0,
) -> None:
    # Import after the wrapper has selected its frozen source checkout.
    from scripts.evaluation.error_analysis_vit import GeluCubicImplementation, ModelBackend
    from utils.transforms.calibration import CalibrationMode

    if args.model_backend is not ModelBackend.SPIKING or args.dataset_id not in DATASET_IMAGE_KEYS:
        raise ValueError("calibrated evaluation requires supported spiking ViT data")
    if not isinstance(args.calibration_mode, CalibrationMode):
        raise ValueError("calibration mode must be collect, validate, or inference")
    if not args.calibration_path:
        raise ValueError("calibration path is required")
    if isinstance(smoke_samples, bool) or not isinstance(smoke_samples, int):
        raise TypeError("calibration smoke samples must be an integer")
    if not 0 <= smoke_samples < CALIBRATION_SAMPLES:
        raise ValueError("calibration smoke samples must be zero or between 1 and 4999")
    if args.calibration_samples != (smoke_samples or CALIBRATION_SAMPLES) or args.calibration_seed != CALIBRATION_SEED:
        raise ValueError("calibration sample count or seed does not match the explicit mode")
    if smoke_samples and args.calibration_mode is not CalibrationMode.COLLECT and args.max_eval_batches <= 0:
        raise ValueError("smoke calibration must not be used for an unlimited evaluation")
    if args.gelu_cubic_implementation is not GeluCubicImplementation.PHI_NL_PSI_ED or args.gelu_cubic_floor != 1e-5:
        raise ValueError("calibration requires the fixed GELU cubic implementation and floor")
    if not Path(args.model_id).is_dir():
        raise ValueError("checkpoint must be a local directory")
    if args.calibration_mode is not CalibrationMode.COLLECT and not args.evaluation_dataset_path:
        raise ValueError("frozen calibration evaluation requires a local evaluation dataset")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint SHA-256 is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--calibration-dataset-path", type=Path, required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
    parser.add_argument(
        "--calibration-smoke-samples", type=int, default=0,
        help="Explicitly use a prefix for memory checks; its table is invalid for final runs.",
    )
    own, remaining = parser.parse_known_args()
    identity_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    identity_parser.add_argument("--source-commit", required=True)
    identity_args, _ = identity_parser.parse_known_args(remaining)
    source = own.source_root.resolve()
    validate_source(source, identity_args.source_commit)

    # No registry fallback, network tracking, or event-file writer is used here.
    os.environ.update(
        WANDB_MODE="disabled", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    sys.path[:0] = [
        str(source), str(source / "src/transformers/src"),
        str(source / "src/spikingjelly"),
    ]
    from datasets import Dataset, load_from_disk
    from utils.transforms.calibration import CalibrationMode
    from scripts.analysis import gelu_cubic_phi_nl_vit as gelu
    from scripts.evaluation import error_analysis_vit as evaluator

    vit_args = gelu.parse_arguments([*remaining, "--no-tensorboard"])
    validate_arguments(vit_args, smoke_samples=own.calibration_smoke_samples)
    implementation = vit_args.gelu_cubic_implementation
    floor = vit_args.gelu_cubic_floor
    if implementation is None or floor is None:
        raise ValueError("GELU cubic configuration is required")
    if identity.artifact_identity(Path(vit_args.model_id))["aggregate_sha256"] != vit_args.checkpoint_sha256:
        raise ValueError("checkpoint contents do not match the expected SHA-256")
    dataset = load_from_disk(str(own.calibration_dataset_path.resolve()))
    if not isinstance(dataset, Dataset):
        raise TypeError("calibration artifact must be one saved training Dataset")
    dataset = calibration_dataset_view(
        dataset, dataset_id=vit_args.dataset_id,
        expected_fingerprint=own.calibration_dataset_fingerprint,
        smoke_samples=own.calibration_smoke_samples,
    )
    dataset = validate_local_training_dataset(
        dataset, expected_fingerprint=dataset._fingerprint,
        dataset_id=vit_args.dataset_id, sample_count=vit_args.calibration_samples,
    )
    bound_identity: dict[str, str | int | float | bool | None] = {
        "source_commit": vit_args.source_commit,
        "checkpoint_sha256": vit_args.checkpoint_sha256,
        "gelu_cubic_implementation": implementation.value,
        "gelu_cubic_floor": floor,
        "gelu_evaluator_sha256": identity.sha256_file(Path(gelu.__file__)),
        "vit_evaluator_sha256": identity.sha256_file(Path(evaluator.__file__)),
        "calibration_wrapper_sha256": identity.sha256_file(Path(__file__)),
        "calibration_dataset_fingerprint": dataset._fingerprint,
        "calibration_source_fingerprint": own.calibration_dataset_fingerprint,
        "calibration_dataset_id": vit_args.dataset_id,
        "calibration_image_key": DATASET_IMAGE_KEYS[vit_args.dataset_id],
        "calibration_purpose": "smoke" if own.calibration_smoke_samples else "final",
    }

    def add_identity(metadata: CalibrationMetadata) -> CalibrationMetadata:
        return bind_metadata_identity(metadata, bound_identity)

    table_path = Path(vit_args.calibration_path)
    initial_table_hash = None
    if vit_args.calibration_mode is CalibrationMode.COLLECT:
        if table_path.exists():
            raise FileExistsError("refusing to replace an existing calibration table")
    else:
        initial_table_hash = identity.sha256_file(table_path)
        print(
            f"Calibration input — mode: {vit_args.calibration_mode}, sha256: {initial_table_hash}",
            flush=True,
        )
    print("Calibration dataset identity — " + json.dumps({
        "path": str(own.calibration_dataset_path.resolve()),
        "fingerprint": dataset._fingerprint,
        "samples": len(dataset), "seed": CALIBRATION_SEED,
        "source_fingerprint": own.calibration_dataset_fingerprint,
        "dataset_id": vit_args.dataset_id, "split": "train",
        "image_key": DATASET_IMAGE_KEYS[vit_args.dataset_id],
        "purpose": "smoke" if own.calibration_smoke_samples else "final",
    }, sort_keys=True), flush=True)
    print(f"GELU cubic implementation: {implementation}")
    print(f"GELU cubic magnitude floor: {floor:.9g}")
    evaluator.evaluate_vit_model(
        vit_args,
        gelu_operator=gelu.make_gelu_cubic_implementation(
            implementation, magnitude_floor=floor,
        ),
        calibration_inputs=evaluator.ViTCalibrationInputs(
            dataset=dataset,
            metadata_transform=add_identity,
            on_batch=make_progress_reporter(
                (len(dataset) + vit_args.batch_size - 1) // vit_args.batch_size,
            ),
        ),
    )
    final_table_hash = identity.sha256_file(table_path)
    if initial_table_hash is not None and final_table_hash != initial_table_hash:
        raise ValueError("calibration table changed during evaluation")
    print(
        f"Calibration identity — mode: {vit_args.calibration_mode}, sha256: {final_table_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
