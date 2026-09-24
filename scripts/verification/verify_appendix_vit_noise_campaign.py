#!/usr/bin/env python3
"""Verify the Appendix ViT-B raw-timestamp campaign contract."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_appendix_vit_noise_campaign as campaign
from scripts.experiments.run_vit_local_range_noise_condition import (
    APPENDIX_RAW_TIMESTAMP_TAG,
    ED_INTERNAL_NOISE_CONTRACT,
    RAW_TIMESTAMP_CONTRACT,
)
from scripts.runtime import identity


def main() -> None:
    assert len(campaign.noise_cells()) == 21
    run_ids = {
        campaign.run_id(fraction, ratio, seed)
        for fraction, ratio in campaign.noise_cells()
        for seed in range(3)
    }
    assert len(run_ids) == 63

    with tempfile.TemporaryDirectory(prefix="appendix-vit-noise-") as directory:
        artifacts = Path(directory)
        old_artifacts = campaign.ARTIFACTS
        campaign.ARTIFACTS = artifacts
        try:
            baseline = campaign.baseline_root()
            baseline.mkdir(parents=True)
            calibration = baseline / "calibration.json"
            calibration.write_text("{}")
            result = {
                "state": "complete",
                "calibration_sha256": identity.sha256_file(calibration),
            }
            (baseline / "result.json").write_text(json.dumps(result))
            (baseline / "manifest.json").write_text(json.dumps({
                "source_commit": "1" * 40,
                "tag": campaign.BASELINE_TAG,
                "checkpoint_sha256": "2" * 64,
            }))
            args = Namespace(
                expected_commit="1" * 40,
                source_root=ROOT,
                local_gpus=tuple(range(8)),
                poseidon_gpus=(1, 2, 3),
            )
            manifest = campaign.build_manifest(args)
        finally:
            campaign.ARTIFACTS = old_artifacts

    assert manifest["noise_tag"] == APPENDIX_RAW_TIMESTAMP_TAG
    assert manifest["raw_timestamp_contract"] == RAW_TIMESTAMP_CONTRACT
    assert manifest["exponential_difference_internal_noise"] == ED_INTERNAL_NOISE_CONTRACT
    assert len(manifest["assignments"]) == 63
    assert {row["host_label"] for row in manifest["assignments"]} == {"local", "poseidon"}
    assert {
        (row["host_label"], row["physical_gpu"])
        for row in manifest["assignments"]
    } == {
        *(("local", gpu) for gpu in range(8)),
        *(("poseidon", gpu) for gpu in (1, 2, 3)),
    }
    assert len({row["run_id"] for row in manifest["assignments"]}) == 63

    command_args = Namespace(
        python_bin="python",
        source_root=ROOT,
        expected_commit="1" * 40,
        host_label="local",
    )
    command = campaign.task_command(command_args, manifest, manifest["assignments"][0], 0)
    assert "--allow-all-local-gpus" in command
    assert "--time-noise-exponential-difference-internal" not in command
    assert "--no-time-noise-exponential-difference-internal" not in command
    assert command[command.index("--campaign-tag") + 1] == APPENDIX_RAW_TIMESTAMP_TAG
    print("Appendix ViT-B raw-timestamp campaign verification passed")


if __name__ == "__main__":
    main()
