"""Verify local calibration loading and metadata without model evaluation."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.evaluate_calibrated_vit import (
    bind_metadata_identity,
    local_training_accessors,
    progress_collector,
    validate_source,
)


def must_reject(callable_):
    try:
        callable_()
    except (ValueError, TypeError):
        return
    raise AssertionError("invalid calibration input was accepted")


class FakeDataset:
    _fingerprint = "training-fingerprint"

    def __init__(self, count=5000):
        self.count = count

    def __len__(self):
        return self.count

    def shuffle(self, **_):
        raise AssertionError("preselected training data must not be shuffled")

    def select(self, *_):
        raise AssertionError("preselected training data must not be selected again")


def verify_local_accessors():
    dataset = FakeDataset()
    load, select = local_training_accessors(dataset, expected_fingerprint=dataset._fingerprint)
    assert load("imagenet-1k", split="train", cache_dir="unused") is dataset
    assert select(dataset, sample_count=5000, seed=0) is dataset
    must_reject(lambda: load("imagenet-1k", split="validation"))
    must_reject(lambda: load("cifar10", split="train"))
    must_reject(lambda: select(FakeDataset(), sample_count=5000, seed=0))
    must_reject(lambda: select(dataset, sample_count=1000, seed=0))
    must_reject(lambda: select(dataset, sample_count=5000, seed=1))
    must_reject(lambda: local_training_accessors(dataset, expected_fingerprint="wrong"))
    must_reject(lambda: local_training_accessors(FakeDataset(4999), expected_fingerprint=dataset._fingerprint))
    dataset._fingerprint = "changed"
    must_reject(lambda: select(dataset, sample_count=5000, seed=0))


def verify_source_identity():
    with patch("subprocess.check_output", side_effect=["commit-a\n", ""]):
        validate_source(Path("/frozen"), "commit-a")
    with patch("subprocess.check_output", return_value="commit-b\n"):
        must_reject(lambda: validate_source(Path("/frozen"), "commit-a"))
    with patch("subprocess.check_output", side_effect=["commit-a\n", " M code.py\n"]):
        must_reject(lambda: validate_source(Path("/frozen"), "commit-a"))


def verify_metadata(source):
    sys.path[:0] = [str(source), str(source / "src/transformers/src"), str(source / "src/spikingjelly")]
    from utils.transforms.calibration import CalibrationMetadata, validate_calibration_metadata

    base = CalibrationMetadata(
        model_family="vit", model_id="/local/model", dataset_id="imagenet-1k",
        dataset_split="train", preprocessing="{}", dtype="float64", theta=40.0,
        tau_s=1.0, tau_m=1.0, clip_margin=1e-5, max_sequence_length=None,
        input_shape=(3, 224, 224), model_options=(("use_spiking_mlp", True),),
    )
    identity = {
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
    must_reject(lambda: validate_calibration_metadata(collected, replace(collected, theta=80.0)))
    assert not any("noise" in key for key, _ in collected.model_options)


def verify_progress_is_observational():
    class Model:
        observer = None
        removed = False

        def register_forward_hook(self, observer):
            self.observer = observer
            return SimpleNamespace(remove=lambda: setattr(self, "removed", True))

    model = Model()
    original_output = object()
    def original(model, dataloader):
        for _ in range(2):
            for item in dataloader:
                assert model.observer(model, (item,), original_output) is None
        return original_output

    printed = io.StringIO()
    with redirect_stdout(printed):
        result = progress_collector(original)(model, [SimpleNamespace(shape=(32,))] * 6)
    assert result is original_output and model.removed
    assert '"completed_batches": 12' in printed.getvalue()
    assert '"completed_samples": 384' in printed.getvalue()
    failing_model = Model()
    def fail(*_):
        raise ValueError("collection failed")
    must_reject(lambda: progress_collector(fail)(failing_model, [1]))
    assert failing_model.removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    verify_local_accessors()
    verify_source_identity()
    verify_metadata(args.source_root.resolve())
    verify_progress_is_observational()
    print("Calibrated ViT evaluator verification passed.")


if __name__ == "__main__":
    main()
