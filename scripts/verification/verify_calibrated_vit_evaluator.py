"""Verify local calibration loading and metadata without model evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.evaluate_calibrated_vit import (
    bind_metadata_identity,
    make_progress_reporter,
    validate_local_training_dataset,
    validate_source,
)


def must_reject(callable_: Callable[[], object]) -> None:
    try:
        callable_()
    except (ValueError, TypeError):
        return
    raise AssertionError("invalid calibration input was accepted")


def verify_local_dataset() -> None:
    from datasets import Dataset

    dataset = Dataset.from_dict({"index": list(range(5000))})
    fingerprint = dataset._fingerprint
    assert validate_local_training_dataset(
        dataset, expected_fingerprint=dataset._fingerprint,
    ) is dataset
    must_reject(lambda: validate_local_training_dataset(dataset, expected_fingerprint="wrong"))
    must_reject(lambda: validate_local_training_dataset(
        dataset.select(range(4999)), expected_fingerprint=dataset._fingerprint,
    ))
    dataset._fingerprint = "changed"
    must_reject(lambda: validate_local_training_dataset(
        dataset, expected_fingerprint=fingerprint,
    ))


def verify_source_identity() -> None:
    with patch("subprocess.check_output", side_effect=["commit-a\n", ""]):
        validate_source(Path("/frozen"), "commit-a")
    with patch("subprocess.check_output", return_value="commit-b\n"):
        must_reject(lambda: validate_source(Path("/frozen"), "commit-a"))
    with patch("subprocess.check_output", side_effect=["commit-a\n", " M code.py\n"]):
        must_reject(lambda: validate_source(Path("/frozen"), "commit-a"))


def verify_metadata(source: Path) -> None:
    sys.path[:0] = [str(source), str(source / "src/transformers/src"), str(source / "src/spikingjelly")]
    from utils.transforms.calibration import CalibrationMetadata, validate_calibration_metadata

    base = CalibrationMetadata(
        model_family="vit", model_id="/local/model", dataset_id="imagenet-1k",
        dataset_split="train", preprocessing="{}", dtype="float64",
        tau_s=1.0, tau_m=1.0, clip_margin=1e-5, max_sequence_length=None,
        input_shape=(3, 224, 224), model_options=(("use_spiking_mlp", True),),
    )
    identity: dict[str, str | int | float | bool | None] = {
        "source_commit": "source-a", "checkpoint_sha256": "a" * 64,
        "gelu_cubic_implementation": "phi_nl_psi_ed", "gelu_cubic_floor": 1e-5,
        "gelu_evaluator_sha256": "b" * 64,
        "calibration_dataset_fingerprint": "training-fingerprint",
    }
    collected = bind_metadata_identity(base, identity)
    assert len(collected.model_options) == len(base.model_options) + len(identity)
    assert list(dict(collected.model_options)) == sorted(dict(collected.model_options))
    validate_calibration_metadata(collected, bind_metadata_identity(base, dict(identity)))
    for key, changed in (
        ("source_commit", "source-b"), ("checkpoint_sha256", "c" * 64),
        ("gelu_cubic_implementation", "multiplication"), ("gelu_cubic_floor", 2e-5),
        ("gelu_evaluator_sha256", "d" * 64),
        ("calibration_dataset_fingerprint", "different-training-data"),
    ):
        other = bind_metadata_identity(base, {**identity, key: changed})
        must_reject(lambda: validate_calibration_metadata(collected, other))
        must_reject(lambda: bind_metadata_identity(collected, {key: changed}))
    must_reject(
        lambda: validate_calibration_metadata(
            collected,
            replace(collected, tau_s=2.0),
        )
    )
    assert not any("noise" in key for key, _ in collected.model_options)


def verify_progress_reporting() -> None:
    printed = io.StringIO()
    with redirect_stdout(printed):
        report = make_progress_reporter(6)
        for pass_index in range(2):
            for _ in range(6):
                report(pass_index, 32)
    assert '"completed_batches": 12' in printed.getvalue()
    assert '"completed_samples": 384' in printed.getvalue()
    must_reject(lambda: make_progress_reporter(0))


def verify_enum_configuration() -> None:
    from scripts.evaluation.error_analysis_vit import (
        Activation, DeviceKind, GeluCubicImplementation, GeluDenseOperator,
        ModelBackend, Precision, parse_arguments, validate_vit_calibration_arguments,
    )
    from scripts.analysis.gelu_cubic_phi_nl_vit import parse_arguments as parse_cubic
    from scripts.analysis.gelu_operator_ablation_vit import parse_arguments as parse_ablation
    from scripts.analysis.evaluate_calibrated_vit import validate_arguments
    from utils.transforms.calibration import CalibrationMode

    defaults = parse_arguments([])
    assert defaults.model_backend is ModelBackend.HF
    assert defaults.device is DeviceKind.CUDA
    assert defaults.precision is Precision.FLOAT32
    assert defaults.activation is Activation.GELU
    assert defaults.calibration_mode is None
    assert defaults.logging_config()["calibration_mode"] == "none"
    for enum_type, flag, field in (
        (ModelBackend, "--model_backend", "model_backend"),
        (DeviceKind, "--device", "device"),
        (Precision, "--precision", "precision"),
        (Activation, "--activation", "activation"),
    ):
        for member in enum_type:
            args = parse_arguments([flag, member.value])
            assert args.logging_config()[field] == member.value
        with redirect_stderr(io.StringIO()):
            try:
                parse_arguments([flag, "invalid"])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("invalid option was accepted")

    # An internal caller must supply enum members, even for valid string spellings.
    must_reject(lambda: replace(defaults, model_backend="spiking"))
    must_reject(lambda: replace(defaults, calibration_mode="collect"))
    must_reject(lambda: replace(defaults, gelu_dense_operators=("division",)))

    for implementation in GeluCubicImplementation:
        args = parse_cubic(["--gelu-cubic-implementation", implementation.value])
        assert args.gelu_cubic_implementation is implementation
        assert args.logging_config()["gelu_cubic_implementation"] == implementation.value
    args = parse_ablation([
        "--gelu-dense-operators", *(operator.value for operator in GeluDenseOperator),
    ])
    assert frozenset(args.gelu_dense_operators) == frozenset(GeluDenseOperator)
    assert all(isinstance(operator, GeluDenseOperator) for operator in args.gelu_dense_operators)
    assert args.logging_config()["gelu_dense_operators"] == ("division", "exponential", "multiplication")

    with tempfile.TemporaryDirectory() as checkpoint:
        for mode in CalibrationMode:
            args = parse_arguments([
                "--model_backend", "spiking", "--model_id", checkpoint,
                "--dataset_id", "imagenet-1k", "--calibration-mode", mode.value,
                "--calibration-path", "local.json", "--calibration-samples", "5000",
                "--evaluation-dataset-path", checkpoint, "--checkpoint-sha256", "a" * 64,
            ])
            assert args.calibration_mode is mode
            assert validate_vit_calibration_arguments(args) is mode
            assert args.logging_config()["calibration_mode"] == mode.value
            # Validation consumes exactly the fields that are recorded and used
            # for construction; no independent selection can disagree with them.
            must_reject(lambda: validate_arguments(args))
            args = replace(
                args, gelu_cubic_implementation=GeluCubicImplementation.PHI_NL_PSI_ED,
                gelu_cubic_floor=1e-5,
            )
            validate_arguments(args)
            must_reject(lambda: validate_arguments(replace(
                args, gelu_cubic_implementation=GeluCubicImplementation.MULTIPLICATION,
            )))
            must_reject(lambda: validate_arguments(replace(args, gelu_cubic_floor=2e-5)))
        must_reject(lambda: validate_arguments(
            replace(args, calibration_mode=None),
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    verify_local_dataset()
    verify_source_identity()
    verify_metadata(args.source_root.resolve())
    verify_progress_reporting()
    verify_enum_configuration()
    print("Calibrated ViT evaluator verification passed.")


if __name__ == "__main__":
    main()
