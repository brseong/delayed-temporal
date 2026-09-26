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
GENERIC_OWNERS = {
    "absolute_path": "scripts/runtime/files.py",
    "atomic_json": "scripts/runtime/files.py",
    "atomic_text": "scripts/runtime/files.py",
    "immutable": "scripts/runtime/files.py",
    "immutable_json": "scripts/runtime/files.py",
    "json_bytes": "scripts/runtime/files.py",
    "new_json": "scripts/runtime/files.py",
    "safe_output": "scripts/runtime/files.py",
    "artifact_records": "scripts/runtime/identity.py",
    "artifact_identity": "scripts/runtime/identity.py",
    "checked_hash": "scripts/runtime/identity.py",
    "json_sha256": "scripts/runtime/identity.py",
    "package_source_files": "scripts/runtime/identity.py",
    "package_source_identity": "scripts/runtime/identity.py",
    "sha256_file": "scripts/runtime/identity.py",
    "sha256_bytes": "scripts/runtime/identity.py",
    "verify_clean_checkout": "scripts/runtime/identity.py",
    "verify_file_identities": "scripts/runtime/identity.py",
    "verify_package_identities": "scripts/runtime/identity.py",
    "gpu_activity": "scripts/runtime/local_gpu.py",
    "gpu_available": "scripts/runtime/local_gpu.py",
    "gpu_occupancy": "scripts/runtime/local_gpu.py",
    "parse_gpu_activity": "scripts/runtime/local_gpu.py",
    "parse_gpu_occupancy": "scripts/runtime/local_gpu.py",
    "parse_queue": "scripts/runtime/slurm.py",
    "require_single_gpu": "scripts/runtime/local_gpu.py",
    "worker_environment": "scripts/runtime/environment.py",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


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

    forbidden_providers = {
        "scripts.experiments.run_calibrated_three_sweeps",
        "scripts.experiments.ubai.prepare_calibrated_three_sweeps_ubai",
        "scripts.experiments.ubai.run_calibrated_three_sweep_pair",
    }
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        providers = imported_modules(path).intersection(forbidden_providers)
        if path.is_relative_to(ROOT / "scripts/experiments") and providers:
            raise AssertionError(
                f"Campaign module imported as a generic helper: {path.relative_to(ROOT)}: "
                f"{sorted(providers)}"
            )
        relative = path.relative_to(ROOT).as_posix()
        for name in defined_functions(path).intersection(GENERIC_OWNERS):
            if relative != GENERIC_OWNERS[name]:
                raise AssertionError(
                    f"Duplicate owner for {name}: {relative}; expected {GENERIC_OWNERS[name]}"
                )

    setup_imports = imported_modules(ROOT / "scripts/setup/hash_artifact.py")
    if "scripts.runtime.identity" not in setup_imports:
        raise AssertionError("The artifact hashing command must use the canonical runtime identity")

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
