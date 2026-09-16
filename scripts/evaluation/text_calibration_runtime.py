"""Shared deterministic calibration and local progress for text evaluators."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.transforms.calibration import (
    CalibrationMode,
    create_calibration_collector,
    create_calibration_runtime,
    get_calibration_clipping_report,
    load_calibration_table,
    save_calibration_table,
    validate_calibration_table_specs,
)
from utils.transformers.calibration import (
    bind_model_calibration,
    clear_model_calibration,
    select_calibration_subset,
)


CALIBRATION_ARGUMENT_DEFAULTS = {
    "dtype": "float32",
    "tensorboard": True,
    "calibration_mode": "none",
    "calibration_path": "",
    "calibration_samples": 5000,
    "calibration_seed": 0,
    "calibration_bins": 2048,
    "calibration_lower_quantile": 0.0,
    "calibration_upper_quantile": 1.0,
    "calibration_margin_fraction": 0.05,
}


def add_text_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same explicit collection and frozen table controls in both families."""
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-mode", choices=("none", "collect", "validate", "inference"), default="none")
    parser.add_argument("--calibration-path", default="")
    parser.add_argument("--calibration-samples", type=int, default=5000)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--calibration-bins", type=int, default=2048)
    parser.add_argument("--calibration-lower-quantile", type=float, default=0.0)
    parser.add_argument("--calibration-upper-quantile", type=float, default=1.0)
    parser.add_argument("--calibration-margin-fraction", type=float, default=0.05)


def text_calibration_argument_values(args: argparse.Namespace) -> dict[str, Any]:
    """Copy parser controls without changing units or compatibility defaults."""
    return {name: getattr(args, name) for name in CALIBRATION_ARGUMENT_DEFAULTS}


def validate_text_calibration_arguments(args: Any) -> CalibrationMode | None:
    """Reject invalid collection before loading any dataset or checkpoint."""
    if args.dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    for name in ("batch_size", "max_length"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(args.max_eval_batches) is not int or args.max_eval_batches < 0:
        raise ValueError("max_eval_batches must be a non-negative integer")
    if args.calibration_mode == "none":
        return None
    mode = CalibrationMode(args.calibration_mode)
    if args.model_backend != "spiking":
        raise ValueError("calibration requires model_backend=spiking")
    if not isinstance(args.calibration_path, str) or not args.calibration_path.strip():
        raise ValueError("calibration_path is required")
    for name, minimum in (("calibration_samples", 1), ("calibration_seed", 0), ("calibration_bins", 2)):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    for name in ("calibration_lower_quantile", "calibration_upper_quantile", "calibration_margin_fraction"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if not 0.0 <= args.calibration_lower_quantile <= args.calibration_upper_quantile <= 1.0:
        raise ValueError("calibration quantiles must be ordered within [0, 1]")
    if args.calibration_margin_fraction < 0:
        raise ValueError("calibration_margin_fraction must be non-negative")
    if mode is CalibrationMode.COLLECT:
        if args.gaussian_time_noise or args.collect_quantiles:
            raise ValueError("calibration collection requires timing noise and quantile diagnostics off")
        if Path(args.calibration_path).exists():
            raise FileExistsError("preserve the existing calibration file; choose a new path")
    elif not Path(args.calibration_path).is_file():
        raise FileNotFoundError(args.calibration_path)
    return mode


def load_text_calibration_subset(args: Any, load_split: Callable[[str], Any]) -> Any:
    """Select exactly the requested training texts selected using the seed, never validation or test texts."""
    training = load_split("train")
    return select_calibration_subset(
        training, sample_count=args.calibration_samples, seed=args.calibration_seed,
    )


def make_text_dataloader(
    dataset: Any,
    tokenizer: Any,
    *,
    text_column: str,
    max_length: int,
    batch_size: int,
    include_labels: bool,
) -> DataLoader:
    """Keep sample order and all model input columns returned by the tokenizer."""
    if text_column not in dataset.column_names:
        raise ValueError(f"missing text column: {text_column}")
    if include_labels and "label" not in dataset.column_names:
        raise ValueError("evaluation requires a label column")

    def tokenize_batch(examples):
        result = tokenizer(
            examples[text_column], padding="max_length",
            truncation=True, max_length=max_length,
        )
        if include_labels:
            result["labels"] = examples["label"]
        return result

    processed = dataset.map(
        tokenize_batch, batched=True, remove_columns=dataset.column_names,
    )
    columns = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in processed.column_names]
    if "input_ids" not in columns or "attention_mask" not in columns:
        raise ValueError("tokenizer must supply input_ids and attention_mask")
    if include_labels:
        columns.append("labels")
    processed.set_format(type="torch", columns=columns)
    return DataLoader(processed, batch_size=batch_size, shuffle=False)


def model_state_sha256(model: nn.Module) -> str:
    """Identify actual loaded checkpoint tensors instead of trusting a mutable path."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_text_calibration(
    model: nn.Module,
    args: Any,
    *,
    model_family: str,
    calibration_dataset: Any,
    text_column: str,
    tokenizer: Any,
    config: Any,
    device: torch.device,
    attention_implementation: str,
    checkpoint_sha256: str,
) -> Any:
    """Collect only training data, or bind an exactly matching immutable table."""
    from utils.transformers.models.text_calibration import (
        build_text_calibration_metadata,
        collect_text_calibration_table,
        text_calibration_specs,
    )

    if isinstance(model, nn.DataParallel):
        raise RuntimeError("calibration does not support DataParallel")
    model.eval()
    config_dict = config.to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype"):
        config_dict.pop(key, None)
    config_sha256 = hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()
    dataset_id = args.dataset_name if args.dataset_config_name is None else f"{args.dataset_name}:{args.dataset_config_name}"
    metadata = build_text_calibration_metadata(
        model_family=model_family, model_id=args.model_id,
        dataset_id=dataset_id, calibration_split="train",
        calibration_dataset_fingerprint=calibration_dataset._fingerprint,
        calibration_samples=args.calibration_samples, calibration_seed=args.calibration_seed,
        tokenizer=tokenizer, config=config, max_length=args.max_length,
        attention_implementation=attention_implementation, dtype=args.dtype,
        text_column=text_column,
        evaluation_options={
            "task": args.task, "batch_size": args.batch_size,
            "checkpoint_sha256": checkpoint_sha256, "config_sha256": config_sha256,
            "calibration_bins": args.calibration_bins,
            "calibration_lower_quantile": args.calibration_lower_quantile,
            "calibration_upper_quantile": args.calibration_upper_quantile,
            "calibration_margin_fraction": args.calibration_margin_fraction,
        },
    )
    specs = text_calibration_specs(
        model, lower_quantile=args.calibration_lower_quantile,
        upper_quantile=args.calibration_upper_quantile,
        margin_fraction=args.calibration_margin_fraction,
    )
    mode = CalibrationMode(args.calibration_mode)
    if mode is CalibrationMode.COLLECT:
        dataloader = make_text_dataloader(
            calibration_dataset, tokenizer, text_column=text_column,
            max_length=args.max_length, batch_size=args.batch_size, include_labels=False,
        )
        collector = create_calibration_collector(metadata, specs, bin_count=args.calibration_bins)
        table = collect_text_calibration_table(
            model, dataloader, collector, device=device,
            expected_samples=args.calibration_samples,
            dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        )
        # Do not overwrite a table that another process created while collecting.
        if Path(args.calibration_path).exists():
            raise FileExistsError(args.calibration_path)
        save_calibration_table(table, args.calibration_path)
        print(f"Saved calibration artifact with {len(table.layers)} layer ranges to {args.calibration_path}", flush=True)
        return None
    table = load_calibration_table(args.calibration_path)
    validate_calibration_table_specs(table, specs)
    state = create_calibration_runtime(mode, table, expected_metadata=metadata)
    bind_model_calibration(model, state)
    print(f"Loaded calibration artifact with {len(table.layers)} layer ranges from {args.calibration_path}", flush=True)
    return state


def finish_text_calibration(model: nn.Module, state: Any) -> None:
    """Report per-site raw counts without changing or widening a frozen table."""
    if state is None:
        return
    for item in get_calibration_clipping_report(state):
        site = f"{item.module_name}/{item.tensor_name}"
        print(
            f"Calibration[{site}] values={item.num_values}, "
            f"underflows={item.underflows} (rate={item.underflow_rate:.6g}), "
            f"overflows={item.overflows} (rate={item.overflow_rate:.6g})",
            flush=True,
        )
    clear_model_calibration(model, expected_state=state)


@dataclass
class ClassificationProgress:
    """Accumulate task accuracy locally and emit one flushed record per batch."""

    expected_total: int
    batches_total: int
    correct: int = 0
    total: int = 0
    batches: int = 0
    started: float = field(default_factory=time.monotonic)
    digest: Any = field(default_factory=hashlib.sha256)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        predicted = predictions.detach().cpu().to(torch.int64).reshape(-1)
        reference = labels.detach().cpu().to(torch.int64).reshape(-1)
        if predicted.shape != reference.shape or not predicted.numel():
            raise ValueError("prediction and label counts must match and be nonzero")
        batch_correct = int((predicted == reference).sum().item())
        self.correct += batch_correct
        self.total += predicted.numel()
        self.batches += 1
        if self.total > self.expected_total or self.batches > self.batches_total:
            raise ValueError("evaluation exceeded the declared sample or batch count")
        self.digest.update(predicted.numpy().astype("<i8", copy=False).tobytes())
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.expected_total - self.total) / self.total
        print(
            f"Evaluation progress: batch={self.batches}/{self.batches_total} "
            f"correct={self.correct} total={self.total}/{self.expected_total} "
            f"accuracy={self.correct / self.total:.8f} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )
        return batch_correct / predicted.numel()

    def final_accuracy(self) -> float:
        if self.total <= 0 or self.total != self.expected_total or self.batches != self.batches_total:
            raise ValueError("incomplete evaluation cannot produce final accuracy")
        print(f"Correct/total: {self.correct}/{self.total}", flush=True)
        print(f"Prediction SHA256: {self.digest.hexdigest()}", flush=True)
        return self.correct / self.total
