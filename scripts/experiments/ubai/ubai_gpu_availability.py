#!/usr/bin/env python3
"""Report currently free UBAI GPUs by theta-selection hardware family."""

from __future__ import annotations

import re
import subprocess


PARTITION_FAMILY = {
    "gpu1": "rtx3090",
    "gpu2": "a10",
    "gpu6": "a10",
    "gpu3": "rtx6000ada",
    "gpu4": "rtxa6000",
    "gpu5": "rtxa6000",
}


def field(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]*)", line)
    return match.group(1) if match else ""


def gpu_count(tres: str) -> int:
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
    return int(match.group(1)) if match else 0


def main() -> None:
    output = subprocess.run(
        ("scontrol", "show", "nodes", "-o"),
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    free = {family: 0 for family in set(PARTITION_FAMILY.values())}
    for line in output.splitlines():
        state = field(line, "State")
        if any(token in state for token in ("DOWN", "DRAIN", "FAIL", "RESV")):
            continue
        partition = field(line, "Partitions").split(",", 1)[0]
        family = PARTITION_FAMILY.get(partition)
        if family is None:
            continue
        configured = gpu_count(field(line, "CfgTRES"))
        allocated = gpu_count(field(line, "AllocTRES"))
        free[family] += max(0, configured - allocated)
    for family in sorted(free):
        print(f"{family}={free[family]}")


if __name__ == "__main__":
    main()
