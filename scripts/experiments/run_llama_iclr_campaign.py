#!/usr/bin/env python3
"""Wait for fixed Llama calibration, then run clean and timing noise evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/evaluation/error_analysis_llama.py"
NOISE_RUNNER = ROOT / "scripts/experiments/run_llama_iclr_noise.py"


def available_gpus(allowed: tuple[int, ...]) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    used = {
        int(index.strip()): int(memory.strip())
        for line in result.stdout.splitlines()
        for index, memory in [line.split(",")]
    }
    return [gpu for gpu in allowed if gpu in used and used[gpu] <= 1024]


def calibration_session_running(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False, capture_output=True,
    )
    return result.returncode == 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], *, log_path: Path, environment: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed; see {log_path}")


def merge_clipping(rows: list[dict]) -> list[dict]:
    groups = [
        {(item["module_name"], item["tensor_name"]): item
         for item in row["calibration_clipping"]}
        for row in rows
    ]
    keys = set(groups[0])
    if any(set(group) != keys for group in groups):
        raise ValueError("clean shards have different calibration sites")
    merged = []
    for key in sorted(keys):
        count = sum(group[key]["num_values"] for group in groups)
        underflows = sum(group[key]["underflows"] for group in groups)
        overflows = sum(group[key]["overflows"] for group in groups)
        merged.append({
            "module_name": key[0], "tensor_name": key[1],
            "num_values": count, "underflows": underflows, "overflows": overflows,
            "underflow_rate": underflows / count if count else 0.0,
            "overflow_rate": overflows / count if count else 0.0,
        })
    return merged


def merge_clean_shards(
    output_dir: Path, *,
    shard_count: int,
    gpu_groups: list[list[int | str]],
    expected: dict,
) -> tuple[dict, dict]:
    if len(gpu_groups) != shard_count:
        raise ValueError("clean shard count differs from GPU groups")
    merged_backends = []
    for backend, filename in (("hf", "hf_clean.json"), ("spiking", "spiking_clean.json")):
        rows = [
            json.loads((output_dir / "shards" / f"shard_{index}" / filename).read_text(encoding="utf-8"))
            for index in range(shard_count)
        ]
        stable_keys = (
            "model_id", "dtype", "evaluation_dataset", "calibration_sha256",
            "implementation_sha256", "calibration_dataset_fingerprint",
            "calibration_examples", "dataset_fingerprint",
            "batch_size", "max_length", "backend", "noise_enabled", "loss_aggregation",
        )
        first = rows[0]
        for index, row in enumerate(rows):
            if any(row.get(key) != first.get(key) for key in stable_keys):
                raise ValueError("clean shards have different result identities")
            if any(row.get(key) != value for key, value in expected.items() if key != "examples"):
                raise ValueError("clean shard differs from requested evaluation")
            total_batches = (expected["examples"] + expected["batch_size"] - 1) // expected["batch_size"]
            start = min(expected["examples"], total_batches * index // shard_count * expected["batch_size"])
            stop = min(expected["examples"], total_batches * (index + 1) // shard_count * expected["batch_size"])
            if (
                row.get("backend") != backend or row.get("noise_enabled") is not False
                or row.get("shard_index") != index or row.get("shard_count") != shard_count
                or row.get("shard_start") != start or row.get("shard_stop") != stop
                or row.get("examples") != stop - start
                or row.get("example_count") != stop - start
            ):
                raise ValueError("clean shard population is incomplete or out of order")
        if first["loss_aggregation"] != "mean_of_batch_losses":
            raise ValueError("clean shard loss aggregation differs")
        batch_count = sum(row["batch_count"] for row in rows)
        token_count = sum(row["valid_token_count"] for row in rows)
        if batch_count <= 0 or token_count <= 0:
            raise ValueError("clean shards contain no valid loss targets")
        batch_loss = sum(row["loss"] * row["batch_count"] for row in rows) / batch_count
        token_loss = sum(
            row["token_weighted_loss"] * row["valid_token_count"] for row in rows
        ) / token_count
        merged = dict(first)
        for key in ("shard_index", "shard_count", "shard_start", "shard_stop", "microbatch_size"):
            merged.pop(key)
        merged.update({
            "examples": expected["examples"],
            "example_count": expected["examples"],
            "batch_count": batch_count,
            "valid_token_count": token_count,
            "loss": batch_loss,
            "perplexity": math.exp(batch_loss),
            "token_weighted_loss": token_loss,
            "token_weighted_perplexity": math.exp(token_loss),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "evaluation_shards": shard_count,
            "physical_gpu_groups": gpu_groups,
            "physical_microbatch_sizes": [row.get("microbatch_size", row["batch_size"]) for row in rows],
        })
        if len({row["device"] for row in rows}) > 1:
            merged["device"] = "distributed"
        if backend == "spiking":
            merged["calibration_clipping"] = merge_clipping(rows)
        merged_backends.append(merged)
    if merged_backends[0]["valid_token_count"] != merged_backends[1]["valid_token_count"]:
        raise ValueError("clean backends have different target populations")
    return merged_backends[0], merged_backends[1]


def run_remote_shards(
    *, host: str, workroot: str, python_bin: str, model_path: str,
    extra_site_packages: str, calibration_path: Path,
    output_dir: Path, model_id: Path, first_shard: int, stop_shard: int, shard_count: int,
    environment: dict[str, str],
) -> None:
    remote_calibration = f"{workroot}/calibration5000.json"
    remote_output = f"{workroot}/full_eval"
    remote_log = output_dir / f"remote_clean_{host}.log"
    run_logged(
        ["rsync", "-a", str(calibration_path), f"{host}:{remote_calibration}"],
        log_path=remote_log, environment=environment,
    )
    run_logged(
        ["ssh", host, shlex.join(["mkdir", "-p", remote_output])],
        log_path=remote_log, environment=environment,
    )
    exports = {
        "LLAMA_WORKROOT": workroot,
        "LLAMA_MODEL_PATH": model_path,
        "LLAMA_PYTHON_BIN": python_bin,
        "LLAMA_EXTRA_SITE_PACKAGES": extra_site_packages,
        "LLAMA_CALIBRATION_PATH": remote_calibration,
        "LLAMA_LOGICAL_MODEL_ID": str(model_id),
        "LLAMA_OUTPUT_ROOT": remote_output,
        "LLAMA_SHARD_COUNT": str(shard_count),
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    sbatch = [
        "sbatch", "--wait", f"--array={first_shard}-{stop_shard - 1}",
        f"--output={remote_output}/slurm-%A_%a.out",
        f"--error={remote_output}/slurm-%A_%a.err",
        f"--export={export_arg}",
        f"{workroot}/repo/scripts/experiments/ubai/run_llama_clean_shard.sbatch",
    ]
    try:
        run_logged(
            ["ssh", host, shlex.join(sbatch)],
            log_path=remote_log, environment=environment,
        )
    finally:
        (output_dir / "remote_logs").mkdir(parents=True, exist_ok=True)
        run_logged(
            ["rsync", "-a", f"{host}:{remote_output}/", str(output_dir / "remote_logs" / host) + "/"],
            log_path=remote_log, environment=environment,
        )
    for index in range(first_shard, stop_shard):
        target = output_dir / "shards" / f"shard_{index}"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("hf_clean.json", "spiking_clean.json"):
            run_logged(
                ["rsync", "-a", f"{host}:{remote_output}/shards/shard_{index}/{filename}", str(target / filename)],
                log_path=remote_log, environment=environment,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-gpus", nargs="+", type=int, required=True)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--calibration-session", default="dt_llama_calibration")
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--eval-shards", type=int, default=1)
    parser.add_argument(
        "--remote-target", nargs=6, action="append", default=[],
        metavar=("HOST", "WORKROOT", "PYTHON_BIN", "MODEL_PATH", "EXTRA_SITE", "SHARDS"),
    )
    args = parser.parse_args()
    if len(args.allowed_gpus) != len(set(args.allowed_gpus)):
        parser.error("allowed GPUs must be distinct")
    if args.gpus_per_replica <= 0:
        parser.error("gpus-per-replica must be positive")
    if args.eval_shards <= 0 or args.eval_shards > 362:
        parser.error("eval-shards must be between 1 and 362")
    remote_targets = []
    for host, workroot, python_bin, model_path, extra_site, count_text in args.remote_target:
        try:
            count = int(count_text)
        except ValueError:
            parser.error("remote target shard count must be an integer")
        if count <= 0 or not all((host, workroot, python_bin, model_path, extra_site)):
            parser.error("remote target requires five paths and a positive shard count")
        remote_targets.append((host, workroot, python_bin, model_path, extra_site, count))
    remote_shards = sum(target[5] for target in remote_targets)
    if remote_shards and not args.clean_only:
        parser.error("remote shards are supported only in clean-only campaigns")

    model_id = args.model_id.resolve(strict=True)
    calibration_path = args.calibration_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_gpus = tuple(args.allowed_gpus)
    local_devices = [f"cuda:{index}" for index in range(args.gpus_per_replica)]

    while not calibration_path.is_file():
        if not calibration_session_running(args.calibration_session):
            raise RuntimeError("calibration did not produce a table")
        time.sleep(30)
    while calibration_session_running(args.calibration_session):
        time.sleep(10)
    calibration_digest = file_sha256(calibration_path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluation.error_analysis_llama import atomic_json, implementation_sha256

    implementation_digest = implementation_sha256()

    clean_path = output_dir / "spiking_clean.json"
    hf_path = output_dir / "hf_clean.json"
    completed_clean = False
    if clean_path.is_file() and hf_path.is_file():
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        completed_clean = (
            clean.get("model_id") == str(model_id)
            and clean.get("calibration_sha256") == calibration_digest
            and clean.get("implementation_sha256") == implementation_digest
            and clean.get("calibration_examples") == 5_000
            and clean.get("examples") == 2_891
            and clean.get("batch_size") == 8
            and clean.get("max_length") == 128
            and clean.get("noise_enabled") is False
            and clean.get("device") == ("distributed" if remote_shards else ",".join(local_devices))
            and (not remote_shards or clean.get("evaluation_shards") == args.eval_shards + remote_shards)
        )
        if not completed_clean:
            raise ValueError("existing clean result has a different identity")

    environment = dict(
        os.environ,
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    if not completed_clean:
        free = available_gpus(allowed_gpus)
        local_shards = min(args.eval_shards, len(free) // args.gpus_per_replica)
        if local_shards == 0:
            raise RuntimeError("no free GPU is available for the Llama clean evaluation")
        if remote_shards and local_shards != args.eval_shards:
            raise RuntimeError("hybrid evaluation requires all requested local GPUs")
        shard_count = local_shards + remote_shards
        if shard_count > 362:
            raise RuntimeError("clean evaluation has more shards than batches")
        needed = args.gpus_per_replica * local_shards
        groups = [
            free[index:index + args.gpus_per_replica]
            for index in range(0, needed, args.gpus_per_replica)
        ]
        print(json.dumps({"event": "clean_start", "gpus": groups, "remote_shards": remote_shards}), flush=True)

        def run_shard(index: int, selected: list[int]) -> None:
            directory = (
                output_dir if shard_count == 1
                else output_dir / "shards" / f"shard_{index}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                args.python_bin, "-u", str(EVALUATOR),
                "--model-id", str(model_id),
                "--calibration-path", str(calibration_path),
                "--output-dir", str(directory),
                "--mode", "clean", "--batch-size", "8", "--max-length", "128",
                "--max-calibration-examples", "5000", "--max-eval-examples", "2891",
                "--devices", *local_devices,
            ]
            if shard_count > 1:
                command.extend([
                    "--eval-shard-index", str(index),
                    "--eval-shard-count", str(shard_count),
                ])
            run_logged(
                command,
                log_path=directory / "clean.log",
                environment=dict(
                    environment,
                    CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in selected),
                ),
            )

        with ThreadPoolExecutor(max_workers=local_shards + len(remote_targets)) as executor:
            futures = [
                executor.submit(run_shard, index, selected)
                for index, selected in enumerate(groups)
            ]
            next_shard = local_shards
            for host, workroot, python_bin, model_path, extra_site, count in remote_targets:
                futures.append(executor.submit(
                    run_remote_shards,
                    host=host, workroot=workroot,
                    python_bin=python_bin, model_path=model_path,
                    extra_site_packages=extra_site,
                    calibration_path=calibration_path, output_dir=output_dir,
                    model_id=model_id, first_shard=next_shard,
                    stop_shard=next_shard + count,
                    shard_count=shard_count, environment=environment,
                ))
                next_shard += count
            for future in futures:
                future.result()
        if shard_count > 1:
            hf, converted = merge_clean_shards(
                output_dir,
                shard_count=shard_count,
                gpu_groups=(
                    [[f"baekryun:{gpu}" for gpu in group] for group in groups]
                    + [[f"{host}:gpu6:cuda:{device}" for device in range(4)]
                       for host, _, _, _, _, count in remote_targets for _ in range(count)]
                    if remote_shards else groups
                ),
                expected={
                    "model_id": str(model_id),
                    "calibration_sha256": calibration_digest,
                    "implementation_sha256": implementation_digest,
                    "calibration_examples": 5_000,
                    "evaluation_dataset": "wikitext2",
                    "examples": 2_891,
                    "batch_size": 8,
                    "max_length": 128,
                    **({} if remote_shards else {"device": ",".join(local_devices)}),
                },
            )
            atomic_json(hf_path, hf)
            atomic_json(clean_path, converted)
        print(json.dumps({"event": "clean_complete", "gpus": groups}), flush=True)

    if args.clean_only:
        return

    free = available_gpus(allowed_gpus)
    usable_count = len(free) // args.gpus_per_replica * args.gpus_per_replica
    free = free[:usable_count]
    if not free:
        raise RuntimeError("no free GPU is available for the Llama noise grid")
    print(json.dumps({"event": "noise_start", "gpus": free}), flush=True)
    run_logged(
        [
            args.python_bin, "-u", str(NOISE_RUNNER),
            "--model-id", str(model_id),
            "--calibration-path", str(calibration_path),
            "--output-dir", str(output_dir),
            "--gpus", *(str(gpu) for gpu in free),
            "--gpus-per-replica", str(args.gpus_per_replica),
            "--python-bin", args.python_bin,
        ],
        log_path=output_dir / "noise_grid.log",
        environment=environment,
    )
    print(json.dumps({"event": "campaign_complete", "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
