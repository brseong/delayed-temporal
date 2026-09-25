#!/usr/bin/env python3
"""Evaluate official CCT-7 with the maintained converted Transformer blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from transformers import AttentionInterface

from utils.transforms.calibration import (
    CalibrationMetadata,
    CalibrationMode,
    create_calibration_collector,
    create_calibration_runtime,
    finalize_calibration_collection,
    load_calibration_table,
    save_calibration_table,
    start_histogram_calibration_pass,
)
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transformers.calibration import (
    OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
    VIT_CALIBRATION_POLICY_VERSION,
    bind_model_calibration,
    clear_model_calibration,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    spiking_sdpa_attention_forward,
)
from utils.transformers.models.spiking_cct import (
    CCT7_CIFAR10_MEAN,
    CCT7_CIFAR10_STD,
    CCT7ForImageClassification,
    load_official_cct7_checkpoint,
)
from utils.transformers.models.spiking_ops import (
    SpikingConv2d,
    SpikingLayerNorm,
    SpikingLinear,
)
from utils.transformers.models.spiking_vit.calibration import vit_calibration_specs


AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_finite(value: object) -> object:
    """Replace empty-counter infinities with JSON null without changing counts."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/nas/cct_7_3x1_32_cifar10_300epochs.pth"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/nas/SNN-Verification_Optimization"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--noise-fractions",
        type=float,
        nargs="*",
        default=(1.0e-3, 3.0e-3, 0.009160358089066789),
    )
    parser.add_argument("--log-noise-fraction", type=float, default=0.0)
    parser.add_argument(
        "--log-noise-fractions",
        type=float,
        nargs="*",
        default=(),
        help="Additional phi_NL-only timing-noise fractions.",
    )
    parser.add_argument(
        "--joint-noise-fractions",
        type=float,
        nargs="*",
        default=(),
        help="Additional equal phi_NP and phi_NL timing-noise fractions.",
    )
    parser.add_argument("--deadline-margin", type=float, default=4.0)
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help=(
            "Evaluate only requested noisy conditions with an existing calibration; "
            "skip repeated ANN and converted-clean baselines."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def calibration_metadata(checkpoint: Path, calibration_samples: int) -> CalibrationMetadata:
    return CalibrationMetadata(
        model_family="vit",
        model_id=f"shi-labs/cct_7_3x1_32@sha256:{sha256_file(checkpoint)}",
        dataset_id="cifar10",
        dataset_split="train",
        preprocessing=json.dumps(
            {
                "mean": CCT7_CIFAR10_MEAN,
                "std": CCT7_CIFAR10_STD,
                "subset": "seed-0 permutation prefix",
                "samples": calibration_samples,
            },
            sort_keys=True,
        ),
        dtype="float64",
        tau_s=1.0,
        tau_m=1.0,
        clip_margin=1.0e-5,
        max_sequence_length=256,
        input_shape=(3, 32, 32),
        model_options=tuple(sorted({
            "attention_implementation": "spiking_sdpa",
            "converted_learned_modules": "tokenizer_conv,encoder,attention_pool,classifier",
            "dense_learned_modules": "",
            "hidden_act": "gelu",
            "layer_norm_clip_margin": 1.0e-5,
            "layer_norm_eps": 1.0e-5,
            "operator_backed_output_head_version": OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
            "spiking_ln_expdiff": True,
            "spiking_ln_log": True,
            "spiking_ln_mul": True,
            "spiking_mlp_exact_gelu": False,
            "use_spiking_layernorm": True,
            "use_spiking_mlp": True,
            "vit_calibration_policy_version": VIT_CALIBRATION_POLICY_VERSION,
        }.items())),
    )


def verify_converted_learned_modules(model: nn.Module) -> list[str]:
    """Reject an enabled converted model that retains a dense learned module."""

    converted: list[str] = []
    violations: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, (SpikingLinear, SpikingConv2d, SpikingLayerNorm)):
            converted.append(name)
            continue
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            violations.append(f"{name}:{type(module).__name__}")
    if violations:
        raise RuntimeError(
            "converted CCT retains dense learned modules: " + ", ".join(violations)
        )
    required = {"tokenizer_conv", "attention_pool", "classifier", "final_norm"}
    missing = sorted(required.difference(converted))
    if missing:
        raise RuntimeError(
            "converted CCT is missing operator-backed modules: " + ", ".join(missing)
        )
    return converted


def collect_calibration(
    model: CCT7ForImageClassification,
    loader: DataLoader,
    metadata: CalibrationMetadata,
    *,
    device: torch.device,
) -> object:
    specs = vit_calibration_specs(
        model,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_fraction=0.05,
    )
    collector = create_calibration_collector(metadata, specs, bin_count=2048)
    bind_model_calibration(model, collector)
    try:
        for pass_index in range(2):
            observed = 0
            with torch.no_grad():
                for images, _ in loader:
                    observed += int(images.shape[0])
                    model(images.to(device=device, dtype=torch.float64))
            if observed != len(loader.dataset):
                raise RuntimeError("calibration pass did not consume its fixed population")
            print(f"calibration pass {pass_index + 1}/2: {observed} samples", flush=True)
            if pass_index == 0:
                start_histogram_calibration_pass(collector)
        return finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model, expected_state=collector)


def evaluate(
    model: CCT7ForImageClassification,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[int, int, str]:
    correct = 0
    total = 0
    prediction_digest = hashlib.sha256()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device=device, dtype=torch.float64))
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("CCT evaluation produced non-finite logits")
            prediction = logits.argmax(dim=-1).to(device="cpu", dtype=torch.int64)
            correct += int((prediction == labels).sum().item())
            total += int(labels.numel())
            prediction_digest.update(prediction.contiguous().numpy().tobytes())
    return correct, total, prediction_digest.hexdigest()


# @lat: [[evaluation#Compact Transformer Diagnostic]]
def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.calibration_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch size must be positive")
    if not all(math.isfinite(value) and value >= 0.0 for value in args.noise_fractions):
        raise ValueError("noise fractions must be finite and non-negative")
    if not math.isfinite(args.log_noise_fraction) or args.log_noise_fraction < 0.0:
        raise ValueError("log noise fraction must be finite and non-negative")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (*args.log_noise_fractions, *args.joint_noise_fractions)
    ):
        raise ValueError("additional noise fractions must be finite and non-negative")
    if not math.isfinite(args.deadline_margin) or args.deadline_margin < 0.0:
        raise ValueError("deadline margin must be finite and non-negative")
    checkpoint = args.checkpoint.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    if args.noise_only and args.calibration_path is None:
        raise ValueError("noise-only evaluation requires --calibration-path")
    if args.noise_only and not (
        args.noise_fractions
        or args.log_noise_fractions
        or args.joint_noise_fractions
    ):
        raise ValueError("noise-only evaluation requires at least one noise condition")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    transform = Compose([
        ToTensor(),
        Normalize(CCT7_CIFAR10_MEAN, CCT7_CIFAR10_STD),
    ])
    train = CIFAR10(dataset_root, train=True, download=False, transform=transform)
    test = CIFAR10(dataset_root, train=False, download=False, transform=transform)
    generator = torch.Generator().manual_seed(0)
    calibration_indices = torch.randperm(len(train), generator=generator)[: args.calibration_samples]
    calibration_loader = DataLoader(
        Subset(train, calibration_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    evaluation_loader = DataLoader(
        Subset(test, range(args.samples)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    started = time.monotonic()
    ann_result = None
    if not args.noise_only:
        reference = CCT7ForImageClassification(converted=False)
        load_official_cct7_checkpoint(reference, checkpoint)
        reference.to(device=device, dtype=torch.float64).eval()
        set_gaussian_time_noise(enabled=False, device=device)
        ann_correct, ann_total, ann_prediction_sha256 = evaluate(
            reference,
            evaluation_loader,
            device=device,
        )
        ann_result = {
            "correct": ann_correct,
            "samples": ann_total,
            "accuracy": ann_correct / ann_total,
            "prediction_sha256": ann_prediction_sha256,
        }
        print(
            f"ANN reference: {ann_correct}/{ann_total} "
            f"({ann_correct / ann_total:.6f})",
            flush=True,
        )
        del reference
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = CCT7ForImageClassification(converted=True)
    load_official_cct7_checkpoint(model, checkpoint)
    converted_modules = verify_converted_learned_modules(model)
    print(
        f"converted learned-module audit: {len(converted_modules)} modules, 0 dense",
        flush=True,
    )
    model.to(device=device, dtype=torch.float64).eval()
    metadata = calibration_metadata(checkpoint, args.calibration_samples)
    set_gaussian_time_noise(enabled=False, device=device)
    calibration_path = (
        args.calibration_path.resolve(strict=True)
        if args.calibration_path is not None
        else output_dir / "calibration.json"
    )
    if calibration_path.is_file():
        table = load_calibration_table(calibration_path)
        create_calibration_runtime(
            CalibrationMode.VALIDATE,
            table,
            expected_metadata=metadata,
        )
        print(f"reusing calibration: {calibration_path}", flush=True)
    else:
        table = collect_calibration(model, calibration_loader, metadata, device=device)
        save_calibration_table(table, calibration_path)

    noisy_conditions = []
    for fraction in args.noise_fractions:
        if args.log_noise_fraction > 0.0:
            name = f"np_{fraction:.17g}_nl_{args.log_noise_fraction:.17g}"
        else:
            name = f"np_{fraction:.17g}"
        noisy_conditions.append((name, fraction, args.log_noise_fraction))
    noisy_conditions.extend(
        (f"nl_{fraction:.17g}", 0.0, fraction)
        for fraction in args.log_noise_fractions
    )
    noisy_conditions.extend(
        (f"np_nl_equal_{fraction:.17g}", fraction, fraction)
        for fraction in args.joint_noise_fractions
    )
    condition_pairs = [
        (linear_fraction, log_fraction)
        for _, linear_fraction, log_fraction in noisy_conditions
    ]
    if len(condition_pairs) != len(set(condition_pairs)):
        raise ValueError("noise sweep contains duplicate NP/NL fraction pairs")
    conditions = (
        noisy_conditions
        if args.noise_only
        else [("converted_clean", 0.0, 0.0), *noisy_conditions]
    )
    rows = []
    for name, linear_fraction, log_fraction in conditions:
        set_gaussian_time_noise(
            enabled=linear_fraction > 0.0 or log_fraction > 0.0,
            time_std_fraction=0.0,
            linear_time_std_fraction=linear_fraction,
            log_time_std_fraction=log_fraction,
            time_mean=0.0,
            deadline_margin_std_ratio=args.deadline_margin,
            seed=args.seed,
            device=device,
        )
        runtime = create_calibration_runtime(
            CalibrationMode.VALIDATE,
            table,
            expected_metadata=metadata,
        )
        bind_model_calibration(model, runtime)
        try:
            correct, total, prediction_sha256 = evaluate(
                model,
                evaluation_loader,
                device=device,
            )
        finally:
            clear_model_calibration(model, expected_state=runtime)
        accuracy = correct / total
        stats = json_finite(get_gaussian_noise_stats())
        row = {
            "condition": name,
            "linear_time_noise_std_fraction": linear_fraction,
            "log_time_noise_std_fraction": log_fraction,
            "deadline_margin_std_ratio": args.deadline_margin,
            "seed": args.seed,
            "correct": correct,
            "samples": total,
            "accuracy": accuracy,
            "prediction_sha256": prediction_sha256,
            "gaussian_counts": stats,
        }
        rows.append(row)
        print(f"{name}: {correct}/{total} ({accuracy:.6f})", flush=True)

    clean_row = next(
        (row for row in rows if row["condition"] == "converted_clean"),
        None,
    )
    result = {
        "schema_version": 3,
        "architecture": "cct_7_3x1_32",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_samples": args.samples,
        "calibration_samples": args.calibration_samples,
        "precision": "float64",
        "converted_scope": "complete learned model path",
        "converted_learned_modules": converted_modules,
        "dense_learned_modules": [],
        "ann_reference": ann_result,
        "conditions": rows,
        "conversion_change_percentage_points": (
            None
            if ann_result is None or clean_row is None
            else 100.0 * (clean_row["accuracy"] - ann_result["accuracy"])
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
