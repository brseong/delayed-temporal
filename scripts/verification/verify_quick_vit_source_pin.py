"""Verify explicit source changes for the small ViT check without loading assets."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.quick_vit_check import derive_experiment, require_frozen_source


def fixture() -> dict:
    return {
        "source_root": "/data/old-source", "source_commit": "a" * 40,
        "evaluator_path": "scripts/evaluation/main.py", "evaluator_sha256": "a" * 64,
        "calibration_evaluator_path": "scripts/evaluation/calibration.py",
        "calibration_evaluator_sha256": "b" * 64,
        "gelu_evaluator_path": "scripts/evaluation/gelu.py", "gelu_evaluator_sha256": "c" * 64,
        "runtime_sha256": {"scripts/experiments/run.py": "d" * 64},
        "dependency_sha256": {"transformers": "e" * 64, "spikingjelly": "f" * 64},
        "models": [{"checkpoint_sha256": "1" * 64, "dataset_sha256": "2" * 64}],
        "theta": 40, "precision": "float64", "calibration_samples": 5000,
    }


def must_reject(action) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Invalid source change accepted")


# @lat: [[quick-family-checks#Quick Model Family Checks#Small ViT Comparison]]
def verify_explicit_identity_change() -> None:
    original = fixture()
    snapshot = deepcopy(original)
    source = Path("/data/new-source")
    prefixes = ("evaluator", "calibration_evaluator", "gelu_evaluator")
    relative = [original[prefix + "_path"] for prefix in prefixes] + list(original["runtime_sha256"])
    contents = {source / name: name.encode() for name in relative}
    with patch("scripts.experiments.quick_vit_check.require_frozen_source") as frozen, \
            patch.object(Path, "read_bytes", autospec=True, side_effect=lambda path: contents[path]):
        derived = derive_experiment(original, source, "b" * 40)
    frozen.assert_called_once_with(source, "b" * 40)
    assert original == snapshot
    assert derived["source_root"] == str(source) and derived["source_commit"] == "b" * 40
    changed = {"source_root", "source_commit", "runtime_sha256"}
    for prefix in prefixes:
        field = prefix + "_sha256"
        changed.add(field)
        assert derived[field] == hashlib.sha256(contents[source / original[prefix + "_path"]]).hexdigest()
    assert derived["runtime_sha256"] == {
        name: hashlib.sha256(contents[source / name]).hexdigest()
        for name in original["runtime_sha256"]
    }
    assert {k: v for k, v in original.items() if k not in changed} == {
        k: v for k, v in derived.items() if k not in changed
    }
    derived["models"][0]["checkpoint_sha256"] = "changed"
    assert original == snapshot


def verify_legacy_and_partial_options() -> None:
    original = fixture()
    with patch("scripts.experiments.quick_vit_check.require_frozen_source") as frozen, \
            patch.object(Path, "read_bytes") as reader:
        derived = derive_experiment(original, None, None)
        assert derived == original and derived is not original
        frozen.assert_called_once_with(Path(original["source_root"]), original["source_commit"])
        reader.assert_not_called()
    with patch("scripts.experiments.quick_vit_check.require_frozen_source") as frozen:
        must_reject(lambda: derive_experiment(original, Path("/data/new-source"), None))
        must_reject(lambda: derive_experiment(original, None, "b" * 40))
        frozen.assert_not_called()


def verify_checkout_and_path_rejections() -> None:
    source = Path("/data/new-source")
    commit = "b" * 40
    with patch("scripts.experiments.quick_vit_check.subprocess.check_output") as process:
        must_reject(lambda: require_frozen_source(source, "short"))
        process.assert_not_called()
    for head, dirty in (("a" * 40, ""), (commit, " M tracked.py\n")):
        with patch("scripts.experiments.quick_vit_check.subprocess.check_output", side_effect=[head, dirty]):
            must_reject(lambda: require_frozen_source(source, commit))
    with patch("scripts.experiments.quick_vit_check.subprocess.check_output", side_effect=[commit + "\n", ""]):
        require_frozen_source(source, commit)
    for invalid in ("/etc/passwd", "../other.py"):
        for field in ("evaluator_path", "runtime_sha256"):
            original = fixture()
            if field == "runtime_sha256":
                original[field] = {invalid: "0" * 64}
            else:
                original[field] = invalid
            with patch("scripts.experiments.quick_vit_check.require_frozen_source"), \
                    patch.object(Path, "read_bytes", return_value=b"test"):
                must_reject(lambda: derive_experiment(original, source, commit))


if __name__ == "__main__":
    verify_explicit_identity_change()
    verify_legacy_and_partial_options()
    verify_checkout_and_path_rejections()
    print("Quick ViT source checks passed")
