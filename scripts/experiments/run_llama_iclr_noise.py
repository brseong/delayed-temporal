#!/usr/bin/env python3
"""Run the Llama timing-noise grid with fixed calibration on free local GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.screening_median_text_sweep import TEXT_INITIAL_ALPHAS, TEXT_SEEDS
from scripts.experiments.screening_median_model_sweep import HARDWARE_SUMMARY_SHA256


EVALUATOR = ROOT / "scripts/evaluation/error_analysis_llama.py"
T_CRITICAL_DF2_95 = 4.302652729911275


def free_gpus(requested: list[int]) -> None:
    if len(requested) != len(set(requested)) or not requested:
        raise ValueError("GPU list must contain distinct devices")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    used = {}
    for line in result.stdout.splitlines():
        index, memory = (int(part.strip()) for part in line.split(","))
        used[index] = memory
    for gpu in requested:
        if gpu not in used or used[gpu] > 1024:
            raise RuntimeError(f"GPU {gpu} is unavailable or already occupied")


def run_cell(
    *, model_id: Path, calibration_path: Path, output_dir: Path,
    clean: dict, python_bin: str, gpus: tuple[int, ...], alpha: str, seed: int,
) -> Path:
    slug = alpha.replace(".", "p")
    result_path = output_dir / f"alpha_{slug}_seed_{seed}.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("model_id") != str(model_id)
            or any(
                existing.get(key) != clean.get(key)
                for key in (
                    "calibration_sha256", "implementation_sha256",
                    "dataset_fingerprint", "examples", "batch_size", "max_length", "device",
                    "calibration_dataset_fingerprint", "calibration_examples",
                )
            )
            or existing.get("alpha") != alpha
            or existing.get("seed") != seed
            or existing.get("noise_enabled") is not True
            or existing.get("hardware_summary_sha256") != HARDWARE_SUMMARY_SHA256
        ):
            raise ValueError(f"existing Llama condition has a different identity: {result_path}")
        return result_path
    log_path = output_dir / f"alpha_{slug}_seed_{seed}.log"
    command = [
        python_bin, "-u", str(EVALUATOR),
        "--model-id", str(model_id),
        "--calibration-path", str(calibration_path),
        "--output-dir", str(output_dir),
        "--mode", "noise", "--alpha", alpha, "--seed", str(seed),
        "--devices", *(f"cuda:{index}" for index in range(len(gpus))),
        "--batch-size", "8", "--max-length", "128",
        "--max-calibration-examples", "5000", "--max-eval-examples", "2891",
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in gpus),
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"alpha={alpha} seed={seed} GPUs={gpus} failed; see {log_path}")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    if args.gpus_per_replica <= 0 or len(args.gpus) < args.gpus_per_replica:
        parser.error("gpus-per-replica must fit the selected GPU list")
    free_gpus(args.gpus)
    model_id = args.model_id.resolve(strict=True)
    calibration_path = args.calibration_path.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    clean_path = output_dir / "spiking_clean.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if (
        clean.get("model_id") != str(model_id)
        or clean.get("calibration_examples") != 5_000
        or clean.get("examples") != 2_891
        or clean.get("batch_size") != 8
        or clean.get("max_length") != 128
        or clean.get("noise_enabled") is not False
        or clean.get("device") != ",".join(
            f"cuda:{index}" for index in range(args.gpus_per_replica)
        )
    ):
        raise ValueError("Llama clean reference is not the complete paper population")
    cells = [
        (alpha, seed) for alpha in TEXT_INITIAL_ALPHAS for seed in TEXT_SEEDS
    ]
    groups = [
        tuple(args.gpus[index:index + args.gpus_per_replica])
        for index in range(0, len(args.gpus), args.gpus_per_replica)
        if len(args.gpus[index:index + args.gpus_per_replica]) == args.gpus_per_replica
    ]
    per_group = {
        group: cells[index::len(groups)]
        for index, group in enumerate(groups)
    }

    def run_group(gpus: tuple[int, ...], assigned: list[tuple[str, int]]) -> list[Path]:
        paths = []
        for alpha, seed in assigned:
            print(json.dumps({"event": "start", "gpus": gpus, "alpha": alpha, "seed": seed}), flush=True)
            paths.append(run_cell(
                model_id=model_id, calibration_path=calibration_path,
                output_dir=output_dir, python_bin=args.python_bin,
                clean=clean,
                gpus=gpus, alpha=alpha, seed=seed,
            ))
            print(json.dumps({"event": "complete", "gpus": gpus, "alpha": alpha, "seed": seed}), flush=True)
        return paths

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(run_group, group, assigned) for group, assigned in per_group.items()]
        for future in as_completed(futures):
            paths.extend(future.result())
    if len(paths) != len(cells):
        raise RuntimeError("Llama timing-noise grid is incomplete")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    means = []
    for alpha in TEXT_INITIAL_ALPHAS:
        values = [
            row["relative_inverse_perplexity_percent"]
            for row in rows if row["alpha"] == alpha
        ]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"alpha={alpha} has an incomplete or nonfinite replica set")
        half_width = T_CRITICAL_DF2_95 * statistics.stdev(values) / math.sqrt(3)
        means.append({
            "alpha": alpha, "mean_relative_inverse_perplexity_percent": statistics.mean(values),
            "confidence_interval_95_percent": [statistics.mean(values) - half_width, statistics.mean(values) + half_width],
            "seeds": list(TEXT_SEEDS), "values": values,
        })
    summary = {
        "model_id": str(model_id),
        "calibration_sha256": clean["calibration_sha256"],
        "evaluation_dataset_fingerprint": clean["dataset_fingerprint"],
        "examples": 2_891, "dtype": "float64",
        "clean_perplexity": clean["perplexity"],
        "conditions": means,
    }
    result = output_dir / "figure3a_llama.json"
    result.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "grid_complete", "summary": str(result)}), flush=True)


if __name__ == "__main__":
    main()
