#!/usr/bin/env python3
"""Create a stable file-level identity manifest for a file or directory artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime.identity import artifact_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identity = artifact_identity(args.path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(identity["aggregate_sha256"])


if __name__ == "__main__":
    main()
