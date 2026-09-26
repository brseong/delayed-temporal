#!/usr/bin/env python3
"""Collect Llama ranges and evaluate the ICLR clean and timing-noise conditions."""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import AttentionInterface, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    canonical_alpha,
    load_screening_median,
    scaled_fractions,
)
from scripts.experiments.screening_median_text_sweep import TEXT_SEEDS
from utils.transforms.calibration import (
    CalibrationMode,
    create_calibration_collector,
    create_calibration_runtime,
    finalize_calibration_collection,
    get_calibration_clipping_report,
    load_calibration_table,
    save_calibration_table,
    start_histogram_calibration_pass,
    validate_calibration_table_specs,
)
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms.types import Potential
from utils.transformers.calibration import bind_model_calibration, clear_model_calibration
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transformers.models.spiking_llama.calibration import (
    llama_calibration_metadata,
    llama_calibration_specs,
)
from utils.transformers.models.spiking_llama.configuration_llama import LlamaConfig
from utils.transformers.models.spiking_llama.modeling_spiking_llama import LlamaForCausalLM


DATASET_ROOT = ROOT / "artifacts/assets/conversion-comparison-text-v1/wikitext2"
EVALUATION_PATH = DATASET_ROOT / "test_nonempty_2891"
CALIBRATION_PATH = DATASET_ROOT / "train_nonempty_seed0_5000"
HARDWARE_PATH = ROOT / "artifacts/brainscales2-primitives/20260924T_best_median_screen_summary.json"
EVALUATION_FINGERPRINT = "38d46c7ecf7254ca"
CALIBRATION_FINGERPRINT = "f1506153809011c4"
IMDB_TEST_FINGERPRINT = "0c4517be449a88ae"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def serializable_noise_stats() -> dict:
    return {
        site: {
            name: None if isinstance(value, float) and not math.isfinite(value) else value
            for name, value in counts.items()
        }
        for site, counts in get_gaussian_noise_stats().items()
    }


def place_decoder_on_devices(model: torch.nn.Module, devices: tuple[torch.device, ...]) -> None:
    """Place decoder blocks in order and move their tensor inputs at each boundary."""
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("decoder devices must be nonempty and distinct")
    decoder = model.model
    primary = devices[0]
    decoder.embed_tokens.to(primary)
    decoder.rotary_emb.to(primary)
    decoder.norm.to(primary)
    model.lm_head.to(primary)

    def transfer(value: object, device: torch.device) -> object:
        if isinstance(value, Potential):
            return Potential(value.value.to(device), value.domain)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(transfer(item, device) for item in value)
        if isinstance(value, list):
            return [transfer(item, device) for item in value]
        if isinstance(value, dict):
            return {key: transfer(item, device) for key, item in value.items()}
        return value

    def route(module: torch.nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        device = next(module.parameters()).device
        return transfer(args, device), transfer(kwargs, device)

    for index, layer in enumerate(decoder.layers):
        device = devices[index * len(devices) // len(decoder.layers)]
        layer.to(device)
        layer.register_forward_pre_hook(route, with_kwargs=True)
    decoder.norm.register_forward_pre_hook(route, with_kwargs=True)
    model.lm_head.register_forward_pre_hook(route, with_kwargs=True)
    if model.config.tie_word_embeddings:
        model.tie_weights()
        if model.lm_head.weight.device != decoder.embed_tokens.weight.device:
            raise RuntimeError("tied Llama weights must share the primary device")


def prepare_batches(
    tokenizer: object,
    *,
    dataset: Dataset,
    source: str,
    expected_size: int,
    expected_fingerprint: str,
    max_examples: int,
    batch_size: int,
    max_length: int,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[DataLoader, dict]:
    if len(dataset) != expected_size or dataset._fingerprint != expected_fingerprint or "text" not in dataset.column_names:
        raise ValueError("text population differs from the selected dataset asset")
    if not 0 < max_examples <= expected_size:
        raise ValueError("max_examples differs from the selected text population")
    if max_examples != len(dataset):
        dataset = dataset.select(range(max_examples))
    population_fingerprint = dataset._fingerprint
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    if not 0 <= shard_index < shard_count <= total_batches:
        raise ValueError("evaluation shard index is outside its shard count")
    shard_start = min(len(dataset), total_batches * shard_index // shard_count * batch_size)
    shard_stop = min(len(dataset), total_batches * (shard_index + 1) // shard_count * batch_size)
    if shard_count > 1:
        dataset = dataset.select(range(shard_start, shard_stop))

    def tokenize(examples: dict) -> dict:
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokens["labels"] = [
            [token if mask else -100 for token, mask in zip(ids, masks)]
            for ids, masks in zip(tokens["input_ids"], tokens["attention_mask"])
        ]
        return tokens

    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names,
        load_from_cache_file=False, keep_in_memory=True,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    metadata = {
        "dataset_path": source,
        "dataset_fingerprint": population_fingerprint,
        "examples": len(dataset),
        "batch_size": batch_size,
        "max_length": max_length,
    }
    if shard_count > 1:
        metadata.update({
            "shard_index": shard_index,
            "shard_count": shard_count,
            "shard_start": shard_start,
            "shard_stop": shard_stop,
        })
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False), metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    paths = (
        "scripts/evaluation/error_analysis_llama.py",
        "scripts/experiments/screening_median_model_sweep.py",
        "scripts/experiments/screening_median_text_sweep.py",
        "utils/transforms/calibration.py",
        "utils/transforms/functions.py",
        "utils/transforms/noise.py",
        "utils/transforms/potential_to_spike.py",
        "utils/transforms/primitive.py",
        "utils/transforms/spike_to_potential.py",
        "utils/transforms/types.py",
        "utils/transformers/calibration.py",
        "utils/transformers/integrations/spiking_sdpa_attention.py",
        "utils/transformers/models/spiking_ops.py",
        "utils/transformers/models/spiking_llama/calibration.py",
        "utils/transformers/models/spiking_llama/configuration_llama.py",
        "utils/transformers/models/spiking_llama/modeling_spiking_llama.py",
        "utils/transformers/tokenizer_identity.py",
    )
    digest = hashlib.sha256()
    for relative_path in paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(ROOT / relative_path)))
    return digest.hexdigest()


@torch.inference_mode()
def collect_calibration(model: LlamaForCausalLM, loader: DataLoader, metadata: object, device: torch.device, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    specs = llama_calibration_specs(model)
    collector = create_calibration_collector(metadata, specs, bin_count=2048)
    set_gaussian_time_noise(enabled=False)
    bind_model_calibration(model, collector)
    try:
        for pass_index in range(2):
            start = time.monotonic()
            for batch_index, batch in enumerate(loader, start=1):
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=False,
                )
                if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                    print(json.dumps({
                        "condition": "calibration", "pass": pass_index + 1,
                        "batches": batch_index, "total_batches": len(loader),
                        "elapsed_seconds": round(time.monotonic() - start, 2),
                    }), flush=True)
            if pass_index == 0:
                start_histogram_calibration_pass(collector)
        table = finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model, expected_state=collector)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_table(table, path)
    print(json.dumps({"condition": "calibration_complete", "sites": len(table.layers), "calibration_path": str(path)}), flush=True)


def bind_frozen_calibration(model: LlamaForCausalLM, metadata: object, path: Path):
    table = load_calibration_table(path)
    validate_calibration_table_specs(table, llama_calibration_specs(model))
    state = create_calibration_runtime(CalibrationMode.VALIDATE, table, expected_metadata=metadata)
    bind_model_calibration(model, state)
    return state


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, *,
    condition: str, microbatch_size: int,
) -> dict:
    total_batch_loss = 0.0
    token_nll_sum = 0.0
    token_count = 0
    batch_count = 0
    example_count = 0
    start = time.monotonic()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        valid_tokens = int((labels[:, 1:] != -100).sum().item())
        if not valid_tokens:
            raise ValueError("a text batch has no next-token targets")
        batch_nll = 0.0
        for offset in range(0, input_ids.shape[0], microbatch_size):
            select = slice(offset, offset + microbatch_size)
            micro_labels = labels[select]
            micro_tokens = int((micro_labels[:, 1:] != -100).sum().item())
            if not micro_tokens:
                continue
            output = model(
                input_ids=input_ids[select], attention_mask=attention_mask[select],
                labels=micro_labels, use_cache=False,
            )
            micro_loss = float(output.loss.item())
            if not math.isfinite(micro_loss) or not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError(f"{condition}: nonfinite loss or logits in batch {batch_count}")
            batch_nll += micro_loss * micro_tokens
            del output
        total_batch_loss += batch_nll / valid_tokens
        token_nll_sum += batch_nll
        token_count += valid_tokens
        batch_count += 1
        example_count += input_ids.shape[0]
        if batch_count == 1 or batch_count % 20 == 0 or batch_count == len(loader):
            print(json.dumps({"condition": condition, "batches": batch_count, "total_batches": len(loader), "examples": example_count, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    batch_mean_loss = total_batch_loss / batch_count
    token_mean_loss = token_nll_sum / token_count
    if max(batch_mean_loss, token_mean_loss) >= math.log(sys.float_info.max):
        raise FloatingPointError(f"{condition}: perplexity overflow")
    return {
        "loss": batch_mean_loss,
        "perplexity": math.exp(batch_mean_loss),
        "loss_aggregation": "mean_of_batch_losses",
        "token_weighted_loss": token_mean_loss,
        "token_weighted_perplexity": math.exp(token_mean_loss),
        "valid_token_count": token_count,
        "batch_count": batch_count,
        "example_count": example_count,
        "elapsed_seconds": time.monotonic() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--calibration-model-id")
    parser.add_argument("--mode", choices=("collect", "clean", "noise"), required=True)
    parser.add_argument("--evaluation-dataset", choices=("wikitext2", "imdb"), default="wikitext2")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--max-eval-examples", type=int, default=2_891)
    parser.add_argument("--eval-shard-index", type=int, default=0)
    parser.add_argument("--eval-shard-count", type=int, default=1)
    parser.add_argument("--max-calibration-examples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--alpha")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length < 2:
        parser.error("batch size must be positive and max length at least 2")
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size
    if not 1 <= args.microbatch_size <= args.batch_size:
        parser.error("microbatch size must be between 1 and batch size")
    eval_batches = (args.max_eval_examples + args.batch_size - 1) // args.batch_size
    if not 0 <= args.eval_shard_index < args.eval_shard_count <= eval_batches:
        parser.error("evaluation shard index is outside its shard count")
    if args.mode != "clean" and (args.eval_shard_index != 0 or args.eval_shard_count != 1):
        parser.error("evaluation shards are supported only in clean mode")
    if args.calibration_model_id is not None and (
        args.mode == "collect" or not args.calibration_model_id.strip()
    ):
        parser.error("calibration-model-id requires a nonempty clean or noise model identity")
    if args.mode == "noise":
        if args.alpha is None or args.seed not in TEXT_SEEDS:
            parser.error("noise mode requires alpha and seed selected from 0, 1, 2")
        args.alpha = canonical_alpha(args.alpha)
    elif args.alpha is not None or args.seed is not None:
        parser.error("alpha and seed are only valid in noise mode")

    devices = tuple(torch.device(name) for name in args.devices)
    if len(devices) != len(set(devices)):
        parser.error("decoder devices must be distinct")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = devices[0]
    model_path = args.model_id.resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("model-id must be a local checkpoint directory")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration_loader, calibration_population = prepare_batches(
        tokenizer, dataset=load_from_disk(str(CALIBRATION_PATH)), source=str(CALIBRATION_PATH),
        expected_size=5_000, expected_fingerprint=CALIBRATION_FINGERPRINT,
        max_examples=args.max_calibration_examples, batch_size=args.batch_size,
        max_length=args.max_length,
    )
    set_gaussian_time_noise(enabled=False)
    AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)
    loader = None
    population = None
    source_result = None
    if args.mode != "collect":
        if args.evaluation_dataset == "wikitext2":
            evaluation_dataset = load_from_disk(str(EVALUATION_PATH))
            evaluation_source = str(EVALUATION_PATH)
            evaluation_size = 2_891
            evaluation_fingerprint = EVALUATION_FINGERPRINT
        else:
            evaluation_dataset = load_dataset("imdb", split="test")
            evaluation_source = "imdb/test"
            evaluation_size = 25_000
            evaluation_fingerprint = IMDB_TEST_FINGERPRINT
        loader, population = prepare_batches(
            tokenizer, dataset=evaluation_dataset, source=evaluation_source,
            expected_size=evaluation_size, expected_fingerprint=evaluation_fingerprint,
            max_examples=args.max_eval_examples, batch_size=args.batch_size,
            max_length=args.max_length,
            shard_index=args.eval_shard_index, shard_count=args.eval_shard_count,
        )
    if args.mode == "clean":
        source = HFLlamaForCausalLM.from_pretrained(
            str(model_path), dtype=torch.float64, attn_implementation="eager",
        ).eval()
        place_decoder_on_devices(source, devices)
        source_result = evaluate(
            source, loader, device, condition="hf_clean",
            microbatch_size=args.microbatch_size,
        )
        del source
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    config = LlamaConfig.from_pretrained(str(model_path), tau_s=1.0, use_spiking_mlp=True)
    converted = LlamaForCausalLM.from_pretrained(
        str(model_path), config=config, dtype=torch.float64, attn_implementation="eager",
    ).eval()
    converted.config._attn_implementation = "spiking_sdpa"
    place_decoder_on_devices(converted, devices)
    metadata = llama_calibration_metadata(
        converted, tokenizer, model_path=model_path,
        calibration_fingerprint=calibration_population["dataset_fingerprint"],
        max_length=args.max_length,
        calibration_samples=args.max_calibration_examples, calibration_seed=0,
    )
    if args.calibration_model_id is not None:
        metadata = replace(metadata, model_id=args.calibration_model_id)
    if args.mode == "collect":
        collect_calibration(converted, calibration_loader, metadata, device, args.calibration_path)
        return

    calibration_digest = file_sha256(args.calibration_path)
    identity = {
        "model_id": metadata.model_id, "dtype": "float64",
        "evaluation_dataset": args.evaluation_dataset,
        "device": ",".join(str(selected) for selected in devices),
        "calibration_sha256": calibration_digest,
        "implementation_sha256": implementation_sha256(),
        "calibration_dataset_fingerprint": calibration_population["dataset_fingerprint"],
        "calibration_examples": args.max_calibration_examples,
        "microbatch_size": args.microbatch_size,
        **population,
    }
    runtime = bind_frozen_calibration(converted, metadata, args.calibration_path)
    set_gaussian_time_noise(enabled=False)
    if args.mode == "clean":
        atomic_json(args.output_dir / "hf_clean.json", {**identity, "backend": "hf", "noise_enabled": False, **source_result})
        clean = evaluate(
            converted, loader, device, condition="spiking_clean",
            microbatch_size=args.microbatch_size,
        )
        atomic_json(args.output_dir / "spiking_clean.json", {
            **identity, "backend": "spiking", "noise_enabled": False,
            "attn_implementation": "spiking_sdpa",
            "calibration_clipping": [item.__dict__ for item in get_calibration_clipping_report(runtime)],
            **clean,
        })
        return

    clean = json.loads((args.output_dir / "spiking_clean.json").read_text(encoding="utf-8"))
    for key in ("model_id", "evaluation_dataset", "dataset_fingerprint", "examples", "batch_size", "max_length", "calibration_sha256", "implementation_sha256", "calibration_dataset_fingerprint", "calibration_examples"):
        if clean.get(key) != identity[key]:
            raise ValueError(f"clean reference {key} differs from noise condition")
    if clean.get("backend") != "spiking" or clean.get("noise_enabled") is not False:
        raise ValueError("noise condition requires converted clean reference")
    pair = load_screening_median(HARDWARE_PATH)
    linear, logarithmic = scaled_fractions(pair, args.alpha)
    condition = f"alpha_{args.alpha.replace('.', 'p')}_seed_{args.seed}"
    set_gaussian_time_noise(
        enabled=True,
        linear_time_std_fraction=linear,
        log_time_std_fraction=logarithmic,
        deadline_margin_std_ratio=DEADLINE_MARGIN_SIGMA_RATIO,
        seed=args.seed,
        device=devices,
    )
    noisy = evaluate(
        converted, loader, device, condition=condition,
        microbatch_size=args.microbatch_size,
    )
    result = {
        **identity, "backend": "spiking", "noise_enabled": True,
        "attn_implementation": "spiking_sdpa", "alpha": args.alpha, "seed": args.seed,
        "linear_time_std_fraction": linear, "log_time_std_fraction": logarithmic,
        "deadline_margin_std_ratio": DEADLINE_MARGIN_SIGMA_RATIO,
        "hardware_summary_sha256": pair.summary_sha256,
        "relative_inverse_perplexity_percent": 100.0 * (clean["perplexity"] / noisy["perplexity"] - 1.0),
        "gaussian_noise_stats": serializable_noise_stats(),
        "calibration_clipping": [item.__dict__ for item in get_calibration_clipping_report(runtime)],
        **noisy,
    }
    atomic_json(args.output_dir / f"{condition}.json", result)
    set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
