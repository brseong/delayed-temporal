#!/usr/bin/env python3
"""Verify source-independent reuse of authenticated dense ANN evaluations."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import ann_baseline


def reject(function, error=Exception) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError("invalid ANN baseline evidence was accepted")


def parse_metrics(text: str) -> dict[str, int]:
    value = json.loads(text)
    return {"correct": int(value["correct"]), "total": int(value["total"])}


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    runtime = Path("/data/delayed-temporal/artifacts/runtime/verification-ann-baseline-cache")
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        root = Path(temporary)
        cache = root / "cache"
        source_log = root / "ann.log"
        source_log.write_text('{"correct": 9, "total": 10}\n')
        baseline_identity = ann_baseline.build_identity(
            model_key="model",
            model_family="vit",
            checkpoint_identity="a" * 64,
            evaluation_dataset={"path": "/host/a", "fingerprint": "data", "samples": 10},
            evaluation_settings={"metric": "top1_accuracy", "precision": "float64"},
        )
        relocated_identity = ann_baseline.build_identity(
            model_key="model",
            model_family="vit",
            checkpoint_identity="a" * 64,
            evaluation_dataset={"path": "/host/b", "fingerprint": "data", "samples": 10},
            evaluation_settings={"metric": "top1_accuracy", "precision": "float64"},
        )
        assert baseline_identity == relocated_identity
        assert "source_commit" not in json.dumps(baseline_identity)

        record = ann_baseline.publish(
            cache_root=cache,
            baseline_identity=baseline_identity,
            source_log=source_log,
            metrics=parse_metrics(source_log.read_text()),
            elapsed_seconds=12.5,
        )
        loaded = ann_baseline.load(
            cache_root=cache,
            baseline_identity=relocated_identity,
            parse_metrics=parse_metrics,
        )
        assert loaded is not None and loaded[0] == record
        phase = ann_baseline.materialize_phase(
            record=loaded[0], source_log=loaded[1], output=root / "result",
        )
        assert phase["elapsed_seconds"] == 0.0
        assert phase["metrics"] == {"correct": 9, "total": 10}
        assert (root / "result" / phase["log_file"]).read_bytes() == source_log.read_bytes()

        changed = ann_baseline.build_identity(
            model_key="model",
            model_family="vit",
            checkpoint_identity="b" * 64,
            evaluation_dataset={"fingerprint": "data", "samples": 10},
            evaluation_settings={"metric": "top1_accuracy", "precision": "float64"},
        )
        assert ann_baseline.load(
            cache_root=cache, baseline_identity=changed, parse_metrics=parse_metrics,
        ) is None

        cached_log = loaded[1]
        cached_log.write_text('{"correct": 8, "total": 10}\n')
        reject(lambda: ann_baseline.load(
            cache_root=cache,
            baseline_identity=baseline_identity,
            parse_metrics=parse_metrics,
        ), ValueError)
    print("ANN baseline cache checks passed")


if __name__ == "__main__":
    main()
