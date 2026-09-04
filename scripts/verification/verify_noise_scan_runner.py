"""Dataset-free verification for the ViT-B noise-scan runner contract."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "scripts" / "experiments" / "noise_scan_vit.sh"
CAMPAIGN = (
    REPOSITORY_ROOT / "scripts" / "experiments" / "run_noise_campaign_vit.sh"
)
PYTHON = Path(sys.executable)


def _run_dry(log_dir: Path, gpus: str) -> subprocess.CompletedProcess[str]:
    """Run only validation and manifest construction under an isolated tag."""
    environment = os.environ.copy()
    environment.update(
        {
            "GPUS": gpus,
            "PYTHON_BIN": str(PYTHON),
            "NOISE_SCAN_DRY_RUN": "1",
            "SCAN_LOGDIR": str(log_dir),
            "SCAN_TAG": "verification",
            "REPLICA_SEEDS": "0 1",
            "TIME_NOISE_STD_FRACS": "1e-10",
            "MISMATCH_THETA_STDS": "1e-5",
        }
    )
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def verify_noise_scan_runner() -> None:
    """Verify the GPU allowlist and seeded two-axis manifest without launching jobs."""
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        accepted = _run_dry(root / "accepted", "4 5 6 7")
        assert accepted.returncode == 0, accepted.stderr
        assert "Dry run validated GPUs: 4 5 6 7" in accepted.stdout

        manifest = root / "accepted" / "expected_runs.tsv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, dialect="excel-tab"))
        assert len(rows) == 5
        assert [row["axis"] for row in rows].count("baseline") == 1
        for axis in ("gaussian", "mismatch"):
            axis_rows = [row for row in rows if row["axis"] == axis]
            assert {row["seed"] for row in axis_rows} == {"0", "1"}
            assert len({row["log_file"] for row in axis_rows}) == 2

        for gpus, message in (
            ("0 4", "only physical GPUs 4, 5, 6, and 7"),
            ("4 4", "must not contain duplicate device 4"),
        ):
            rejected = _run_dry(root / f"rejected-{gpus.replace(' ', '-')}", gpus)
            assert rejected.returncode == 2
            assert message in rejected.stderr

    environment = os.environ.copy()
    environment.update(
        {
            "NOISE_CAMPAIGN_DRY_RUN": "1",
            "GPU_POLL_SECONDS": "60",
            "GPU_IDLE_SAMPLES": "2",
        }
    )
    campaign = subprocess.run(
        ["bash", str(CAMPAIGN)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    for marker in (
        "Allowed GPUs: 4 5 6 7",
        "Idle rule: 2 consecutive samples, 60s apart",
        "Stages: smoke quick(70) full(19) theta(111) validate publish build",
    ):
        assert marker in campaign.stdout


if __name__ == "__main__":
    verify_noise_scan_runner()
    print("Noise-scan runner verification passed.")
