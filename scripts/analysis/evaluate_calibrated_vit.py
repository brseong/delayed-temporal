"""Evaluate ViT with local training calibration data and a frozen source checkout."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


CALIBRATION_SAMPLES = 5000
CALIBRATION_SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def local_training_accessors(
    dataset: Any, *, expected_fingerprint: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return the already selected training population without another shuffle."""
    if len(dataset) != CALIBRATION_SAMPLES:
        raise ValueError("calibration artifact must contain exactly 5000 samples")
    if not expected_fingerprint or dataset._fingerprint != expected_fingerprint:
        raise ValueError("calibration dataset fingerprint mismatch")

    def load_training(dataset_id: str, *, split: str, **_: Any) -> Any:
        if dataset_id != "imagenet-1k" or split != "train":
            raise ValueError("local calibration loader accepts only ImageNet training data")
        return dataset

    def select_training(loaded: Any, *, sample_count: int, seed: int) -> Any:
        if loaded is not dataset:
            raise ValueError("calibration subset must come from the verified local artifact")
        if sample_count != CALIBRATION_SAMPLES or seed != CALIBRATION_SEED:
            raise ValueError("local calibration requires exactly 5000 samples and seed 0")
        if loaded._fingerprint != expected_fingerprint:
            raise ValueError("calibration dataset fingerprint changed")
        return loaded

    return load_training, select_training


def bind_metadata_identity(metadata: Any, identity: dict[str, Any]) -> Any:
    """Extend the frozen metadata with the implementation and artifact identities."""
    options = dict(metadata.model_options)
    for key, value in identity.items():
        if key in options and options[key] != value:
            raise ValueError(f"calibration metadata identity conflict: {key}")
        options[key] = value
    return replace(metadata, model_options=tuple(sorted(options.items())))


def progress_collector(original: Callable[..., Any]) -> Callable[..., Any]:
    """Report progress without replacing tensors, outputs, or the two collection passes."""
    def collect(model: Any, dataloader: Any, *args: Any, **kwargs: Any) -> Any:
        batches_per_pass = len(dataloader)
        total_batches = 2 * batches_per_pass
        completed_batches = 0
        completed_samples = 0
        started = time.monotonic()

        def observe(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            nonlocal completed_batches, completed_samples
            completed_batches += 1
            completed_samples += int(inputs[0].shape[0])
            if completed_batches % 10 == 0 or completed_batches == total_batches:
                print("Calibration progress — " + json.dumps({
                    "completed_batches": completed_batches,
                    "total_batches": total_batches,
                    "completed_samples": completed_samples,
                    "pass": (completed_batches - 1) // batches_per_pass + 1,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }, sort_keys=True), flush=True)

        handle = model.register_forward_hook(observe)
        try:
            return original(model, dataloader, *args, **kwargs)
        finally:
            handle.remove()

    return collect


def validate_arguments(args: Any, implementation: str, magnitude_floor: float) -> None:
    if args.model_backend != "spiking" or args.dataset_id != "imagenet-1k":
        raise ValueError("calibrated evaluation requires spiking ImageNet ViT")
    if args.calibration_mode not in {"collect", "validate", "inference"}:
        raise ValueError("calibration mode must be collect, validate, or inference")
    if not args.calibration_path:
        raise ValueError("calibration path is required")
    if args.calibration_samples != CALIBRATION_SAMPLES or args.calibration_seed != CALIBRATION_SEED:
        raise ValueError("calibration requires exactly 5000 samples and seed 0")
    if implementation != "phi_nl_psi_ed" or magnitude_floor != 1e-5:
        raise ValueError("calibration requires the fixed GELU cubic implementation and floor")
    if not Path(args.model_id).is_dir():
        raise ValueError("checkpoint must be a local directory")
    if args.calibration_mode != "collect" and not args.evaluation_dataset_path:
        raise ValueError("frozen calibration evaluation requires a local evaluation dataset")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint SHA-256 is required")


# @lat: [[evaluation#Evaluation and Verification#Calibrated ViT Noise Comparison]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--calibration-dataset-path", type=Path, required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
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
    from scripts.analysis import gelu_cubic_phi_nl_vit as gelu
    from scripts.evaluation import error_analysis_vit as evaluator
    from scripts.setup.hash_artifact import artifact_identity

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining, "--no-tensorboard"]
    originals: dict[str, Any] = {}
    try:
        vit_args, implementation, floor = gelu.parse_arguments()
        validate_arguments(vit_args, implementation, floor)
        if artifact_identity(Path(vit_args.model_id))["aggregate_sha256"] != vit_args.checkpoint_sha256:
            raise ValueError("checkpoint contents do not match the expected SHA-256")
        dataset = load_from_disk(str(own.calibration_dataset_path.resolve()))
        if not isinstance(dataset, Dataset):
            raise TypeError("calibration artifact must be one saved training Dataset")
        if not {"image", "label"}.issubset(dataset.column_names):
            raise ValueError("calibration artifact requires image and label columns")
        load_training, select_training = local_training_accessors(
            dataset, expected_fingerprint=own.calibration_dataset_fingerprint,
        )
        original_metadata = evaluator.build_vit_calibration_metadata
        bound_identity = {
            "source_commit": vit_args.source_commit,
            "checkpoint_sha256": vit_args.checkpoint_sha256,
            "gelu_cubic_implementation": implementation,
            "gelu_cubic_floor": floor,
            "gelu_evaluator_sha256": sha256_file(Path(gelu.__file__)),
            "calibration_dataset_fingerprint": own.calibration_dataset_fingerprint,
        }

        def build_metadata(**kwargs: Any) -> Any:
            return bind_metadata_identity(original_metadata(**kwargs), bound_identity)

        for name, replacement in (
            ("load_dataset", load_training),
            ("select_calibration_subset", select_training),
            ("build_vit_calibration_metadata", build_metadata),
            ("collect_vit_calibration_table", progress_collector(evaluator.collect_vit_calibration_table)),
        ):
            originals[name] = getattr(evaluator, name)
            setattr(evaluator, name, replacement)
        table_path = Path(vit_args.calibration_path)
        initial_table_hash = None
        if vit_args.calibration_mode == "collect":
            if table_path.exists():
                raise FileExistsError("refusing to replace an existing calibration table")
        else:
            initial_table_hash = sha256_file(table_path)
            print(
                f"Calibration input — mode: {vit_args.calibration_mode}, sha256: {initial_table_hash}",
                flush=True,
            )
        print("Calibration dataset identity — " + json.dumps({
            "path": str(own.calibration_dataset_path.resolve()),
            "fingerprint": dataset._fingerprint,
            "samples": len(dataset), "seed": CALIBRATION_SEED,
        }, sort_keys=True), flush=True)
        gelu.main()
        final_table_hash = sha256_file(table_path)
        if initial_table_hash is not None and final_table_hash != initial_table_hash:
            raise ValueError("calibration table changed during evaluation")
        print(
            f"Calibration identity — mode: {vit_args.calibration_mode}, sha256: {final_table_hash}",
            flush=True,
        )
    finally:
        for name, original in originals.items():
            setattr(evaluator, name, original)
        sys.argv = original_argv


if __name__ == "__main__":
    main()
