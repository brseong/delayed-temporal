"""Small parsers shared by Slurm-facing experiment controllers."""
from __future__ import annotations

import re


def parse_queue(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        job, state, name, resources = line.split("|", 3)
        matches = re.findall(r"gpu(?::[^:,]+)?:([0-9]+)", resources)
        rows.append(
            {
                "job_id": job.strip(),
                "state": state.strip(),
                "name": name.strip(),
                "gpus": sum(map(int, matches)),
            }
        )
    return rows
