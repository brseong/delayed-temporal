#!/usr/bin/env python3
"""Run the bounded two-dataset Llama appendix timing-noise diagnostic."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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

from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    HARDWARE_SUMMARY_SHA256,
    canonical_alpha,
)
from scripts.runtime.local_gpu import gpu_activity, gpu_available


EVALUATOR = ROOT / "scripts/evaluation/error_analysis_llama.py"
DATASETS = ("wikitext2", "imdb")
ALPHAS = ("0.00001", "0.0001", "0.001")
CALIBRATION_EXAMPLES = 32
EVALUATION_EXAMPLES = 128
BATCH_SIZE = 8
MAX_LENGTH = 128
T_CRITICAL_DF2_95 = 4.302652729911275


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def expected_identity(
    *, model_id: Path, calibration_path: Path, dataset: str,
    implementation_digest: str, device_count: int,
) -> dict:
    return {
        "model_id": str(model_id),
        "dtype": "float64",
        "evaluation_dataset": dataset,
        "device": ",".join(f"cuda:{index}" for index in range(device_count)),
        "calibration_sha256": sha256(calibration_path),
        "implementation_sha256": implementation_digest,
        "calibration_examples": CALIBRATION_EXAMPLES,
        "examples": EVALUATION_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
    }


def require_identity(row: dict, expected: dict, path: Path) -> None:
    differences = [key for key, value in expected.items() if row.get(key) != value]
    if differences:
        raise ValueError(f"result identity mismatch ({', '.join(differences)}): {path}")


def clean_pair(directory: Path, expected: dict) -> tuple[dict, dict]:
    hf_path = directory / "hf_clean.json"
    converted_path = directory / "spiking_clean.json"
    hf = json.loads(hf_path.read_text(encoding="utf-8"))
    converted = json.loads(converted_path.read_text(encoding="utf-8"))
    for path, row, backend in ((hf_path, hf, "hf"), (converted_path, converted, "spiking")):
        require_identity(row, {**expected, "backend": backend, "noise_enabled": False}, path)
        if not math.isfinite(row["perplexity"]) or row["perplexity"] <= 0:
            raise ValueError(f"nonfinite clean perplexity: {path}")
    for key in ("dataset_fingerprint", "dataset_path", "calibration_dataset_fingerprint"):
        if not converted.get(key) or hf.get(key) != converted[key]:
            raise ValueError(f"clean population mismatch ({key}): {directory}")
    return hf, converted


def condition_path(directory: Path, alpha: str, seed: int) -> Path:
    return directory / f"alpha_{alpha.replace('.', 'p')}_seed_{seed}.json"


def require_noise(path: Path, clean: dict, alpha: str, seed: int) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    fields = (
        "model_id", "dtype", "evaluation_dataset", "device", "calibration_sha256",
        "implementation_sha256", "calibration_dataset_fingerprint",
        "calibration_examples", "dataset_fingerprint", "dataset_path",
        "examples", "batch_size", "max_length",
    )
    require_identity(
        row,
        {**{key: clean[key] for key in fields}, "backend": "spiking",
         "noise_enabled": True, "alpha": alpha, "seed": seed,
         "hardware_summary_sha256": HARDWARE_SUMMARY_SHA256,
         "deadline_margin_std_ratio": DEADLINE_MARGIN_SIGMA_RATIO},
        path,
    )
    if not math.isfinite(row["perplexity"]) or row["perplexity"] <= 0:
        raise ValueError(f"nonfinite noisy perplexity: {path}")
    expected_change = 100.0 * (clean["perplexity"] / row["perplexity"] - 1.0)
    if not math.isclose(row["relative_inverse_perplexity_percent"], expected_change, abs_tol=1e-9):
        raise ValueError(f"relative inverse perplexity differs from clean reference: {path}")
    return row


def run_cell(
    *, python_bin: str, model_id: Path, calibration_path: Path,
    directory: Path, dataset: str, clean: dict, alpha: str, seed: int,
    gpus: tuple[int, ...],
) -> Path:
    result_path = condition_path(directory, alpha, seed)
    if result_path.is_file():
        require_noise(result_path, clean, alpha, seed)
        return result_path
    command = [
        python_bin, "-u", str(EVALUATOR),
        "--model-id", str(model_id),
        "--calibration-path", str(calibration_path),
        "--output-dir", str(directory),
        "--mode", "noise", "--evaluation-dataset", dataset,
        "--alpha", alpha, "--seed", str(seed),
        "--max-calibration-examples", str(CALIBRATION_EXAMPLES),
        "--max-eval-examples", str(EVALUATION_EXAMPLES),
        "--batch-size", str(BATCH_SIZE), "--max-length", str(MAX_LENGTH),
        "--devices", *(f"cuda:{index}" for index in range(len(gpus))),
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in gpus),
        TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    log_path = result_path.with_suffix(".log")
    print(json.dumps({"event": "start", "dataset": dataset, "alpha": alpha,
                      "seed": seed, "gpus": gpus}), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Llama condition failed; see {log_path}")
    require_noise(result_path, clean, alpha, seed)
    print(json.dumps({"event": "complete", "dataset": dataset, "alpha": alpha,
                      "seed": seed, "gpus": gpus}), flush=True)
    return result_path


def summarize(
    *, output_root: Path, datasets: tuple[str, ...],
    cleans: dict[str, tuple[dict, dict]], seeds: tuple[int, ...],
) -> dict:
    if len(seeds) not in (1, 3):
        raise ValueError("summary requires one seed or the complete three-seed set")
    rows = []
    for dataset in datasets:
        hf, converted = cleans[dataset]
        conditions = []
        for alpha in ALPHAS:
            replicas = [require_noise(condition_path(output_root / dataset, alpha, seed),
                                      converted, alpha, seed) for seed in seeds]
            ppls = [row["perplexity"] for row in replicas]
            changes = [row["relative_inverse_perplexity_percent"] for row in replicas]
            condition = {
                "alpha": alpha, "seeds": list(seeds), "perplexities": ppls,
                "mean_perplexity": statistics.mean(ppls),
                "relative_inverse_perplexity_percent": changes,
                "mean_relative_inverse_perplexity_percent": statistics.mean(changes),
            }
            if len(seeds) == 3:
                ppl_half_width = T_CRITICAL_DF2_95 * statistics.stdev(ppls) / math.sqrt(3)
                condition["perplexity_ci_95"] = [
                    condition["mean_perplexity"] - ppl_half_width,
                    condition["mean_perplexity"] + ppl_half_width,
                ]
                center = condition["mean_relative_inverse_perplexity_percent"]
                half_width = T_CRITICAL_DF2_95 * statistics.stdev(changes) / math.sqrt(3)
                condition["relative_inverse_perplexity_ci_95_percent"] = [
                    center - half_width, center + half_width,
                ]
            conditions.append(condition)
        rows.append({
            "dataset": dataset, "dataset_fingerprint": converted["dataset_fingerprint"],
            "hf_clean_perplexity": hf["perplexity"],
            "converted_clean_perplexity": converted["perplexity"],
            "conditions": conditions,
        })
    return {
        "model_id": cleans[datasets[0]][1]["model_id"],
        "calibration_sha256": cleans[datasets[0]][1]["calibration_sha256"],
        "implementation_sha256": cleans[datasets[0]][1]["implementation_sha256"],
        "hardware_summary_sha256": HARDWARE_SUMMARY_SHA256,
        "calibration_examples": CALIBRATION_EXAMPLES,
        "evaluation_examples_per_dataset": EVALUATION_EXAMPLES,
        "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
        "dtype": "float64", "seeds": list(seeds), "datasets": rows,
    }


def render_table(summary: dict) -> str:
    if tuple(row["dataset"] for row in summary["datasets"]) != DATASETS:
        raise ValueError("appendix table requires both datasets")
    rows = summary["datasets"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Llama 2 7B diagnostic perplexity on the first 128 test texts",
        r"per dataset after calibration on 32 WikiText-2 training texts.",
        r"The three noisy rows scale both measured timing noise fractions",
        r"by the indicated multiplier; smaller perplexity is better.",
        (r"The noisy cells use seed 0 only; no confidence interval is implied.}"
         if len(summary["seeds"]) == 1 else
         r"The noisy cells show mean perplexity and 95\% Student-$t$"
         r" confidence intervals over seeds $\{0,1,2\}$.}"),
        r"\label{tab:llama_noise_diagnostic}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Condition & WikiText-2 & IMDb " + "\\\\",
        r"\midrule",
    ]
    for heading, key in (("Source clean", "hf_clean_perplexity"),
                         ("Converted clean", "converted_clean_perplexity")):
        lines.append(heading + " & " + " & ".join(f"{row[key]:.3f}" for row in rows) + " \\\\")
    lines.append(r"\midrule")
    for index, (alpha, power) in enumerate(zip(ALPHAS, (5, 4, 3), strict=True)):
        values = []
        for row in rows:
            cell = row["conditions"][index]
            if cell["alpha"] != alpha:
                raise ValueError("appendix table noise conditions are out of order")
            mean = cell["mean_perplexity"]
            if len(summary["seeds"]) == 3:
                lower, upper = cell["perplexity_ci_95"]
                values.append(rf"\({mean:.3f}\pm{(upper - lower) / 2:.3f}\)")
            else:
                values.append(f"{mean:.3f}")
        lines.append(rf"$10^{{-{power}}}$ & " + " & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run", "summarize"), required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[])
    parser.add_argument("--gpus-per-replica", type=int, default=2)
    parser.add_argument("--alphas", nargs="+", default=list(ALPHAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    if args.gpus_per_replica < 1 or len(args.seeds) not in (1, 3):
        parser.error("one or three seeds and a positive GPU count are required")
    seeds = tuple(args.seeds)
    if seeds not in ((0,), (0, 1, 2)):
        parser.error("supported seeds are 0 or 0 1 2")
    alphas = tuple(canonical_alpha(alpha) for alpha in args.alphas)
    if len(alphas) != len(set(alphas)) or not set(alphas).issubset(ALPHAS):
        parser.error("alphas must be distinct members of 0.00001 0.0001 0.001")
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU list must contain distinct indices")
    model_id = args.model_id.resolve(strict=True)
    calibration_path = args.calibration_path.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    from scripts.evaluation.error_analysis_llama import implementation_sha256
    implementation_digest = implementation_sha256()
    cleans = {}
    for dataset in DATASETS:
        expected = expected_identity(
            model_id=model_id, calibration_path=calibration_path,
            dataset=dataset, implementation_digest=implementation_digest,
            device_count=args.gpus_per_replica,
        )
        cleans[dataset] = clean_pair(output_root / dataset, expected)
    cells = [(dataset, alpha, seed) for dataset in DATASETS for alpha in alphas for seed in seeds]
    missing = [cell for cell in cells if not condition_path(output_root / cell[0], cell[1], cell[2]).is_file()]
    for dataset, alpha, seed in cells:
        path = condition_path(output_root / dataset, alpha, seed)
        if path.is_file():
            require_noise(path, cleans[dataset][1], alpha, seed)
    print(json.dumps({"event": "plan", "total": len(cells), "completed": len(cells) - len(missing),
                      "missing": missing}), flush=True)
    if args.mode == "plan":
        return
    if args.mode == "run" and missing:
        if not args.gpus or len(args.gpus) % args.gpus_per_replica:
            parser.error("run mode requires complete GPU replica groups")
        activity = gpu_activity(gpu_ids=tuple(args.gpus))
        occupied = [gpu for gpu in args.gpus if not gpu_available(activity[gpu])]
        if occupied:
            raise RuntimeError(f"requested GPUs are occupied: {occupied}")
        groups = [tuple(args.gpus[index:index + args.gpus_per_replica])
                  for index in range(0, len(args.gpus), args.gpus_per_replica)]

        def run_group(group: tuple[int, ...], assigned: list[tuple[str, str, int]]) -> None:
            for dataset, alpha, seed in assigned:
                run_cell(
                    python_bin=args.python_bin, model_id=model_id,
                    calibration_path=calibration_path,
                    directory=output_root / dataset, dataset=dataset,
                    clean=cleans[dataset][1], alpha=alpha, seed=seed, gpus=group,
                )

        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = [executor.submit(run_group, group, missing[index::len(groups)])
                       for index, group in enumerate(groups)]
            for future in as_completed(futures):
                future.result()
    full_cells = [(dataset, alpha, seed) for dataset in DATASETS for alpha in ALPHAS for seed in seeds]
    pending = [cell for cell in full_cells if not condition_path(output_root / cell[0], cell[1], cell[2]).is_file()]
    if pending:
        if args.mode == "summarize":
            raise ValueError(f"appendix summary is incomplete: {pending}")
        print(json.dumps({"event": "summary_pending", "missing": pending}), flush=True)
        return
    result = summarize(output_root=output_root, datasets=DATASETS, cleans=cleans, seeds=seeds)
    atomic_text(output_root / "appendix_llama_noise_diagnostic.json",
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_text(output_root / "appendix_llama_noise_table.tex", render_table(result))
    print(json.dumps({"event": "summary_complete", "output_root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
