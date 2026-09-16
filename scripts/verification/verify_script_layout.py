#!/usr/bin/env python3
"""Verify the dependency boundary around the maintained evaluator entry points."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATORS = tuple(
    ROOT / "scripts" / "evaluation" / f"error_analysis_{family}.py"
    for family in ("vit", "bert", "roberta", "gpt2")
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


# @lat: [[code-layout#Script Architecture#Dependency Boundary Verification]]
def verify_layout() -> None:
    for path in EVALUATORS:
        if not path.is_file():
            raise AssertionError(f"Missing evaluator entry point: {path.relative_to(ROOT)}")
        forbidden = sorted(
            module for module in imported_modules(path)
            if module == "scripts.experiments" or module.startswith("scripts.experiments.")
        )
        if forbidden:
            raise AssertionError(f"Evaluator imports campaign code: {path.name}: {forbidden}")

    runtime = ROOT / "scripts" / "runtime"
    for path in sorted(runtime.glob("*.py")):
        forbidden = sorted(
            module for module in imported_modules(path)
            if module.startswith(("scripts.experiments", "scripts.evaluation", "scripts.analysis"))
        )
        if forbidden:
            raise AssertionError(f"Runtime helper depends on a higher layer: {path.name}: {forbidden}")

    deprecated_import = "scripts.experiments.run_calibrated_three_sweeps"
    allowed = {
        ROOT / "scripts" / "verification" / "verify_calibrated_three_sweep_runner.py",
    }
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        if path in allowed or path.name == "run_calibrated_three_sweeps.py":
            continue
        if deprecated_import in imported_modules(path):
            raise AssertionError(
                f"Generic helper imported from a campaign controller: {path.relative_to(ROOT)}"
            )

    for relative in (
        "scripts/README.md",
        "scripts/evaluation/README.md",
        "scripts/experiments/README.md",
    ):
        if not (ROOT / relative).is_file():
            raise AssertionError(f"Missing script map: {relative}")


if __name__ == "__main__":
    verify_layout()
    print("Script layout verification passed.")

