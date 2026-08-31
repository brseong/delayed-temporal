from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, cast
import math
import argparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.transformers.optional_tensorboard import create_summary_writer
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from utils.transforms.calibration import (
    CalibrationMode,
    create_calibration_collector,
    create_calibration_runtime,
    get_calibration_clipping_report,
    load_calibration_table,
    save_calibration_table,
    validate_calibration_table_specs,
)
from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
    GPT2Attention,
    GPT2LMHeadModel,
    SpikingConv1D,
)
from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
from utils.transformers.models.spiking_gpt2.calibration import (
    build_gpt2_calibration_metadata,
    collect_gpt2_calibration_table,
    gpt2_calibration_specs,
)
from utils.transformers.calibration import (
    bind_model_calibration,
    clear_model_calibration,
    select_calibration_subset,
)
from utils.transformers.models.spiking_ops import SpikingLayerNorm, SpikingLinear
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transforms.types import Potential
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms import types
from tqdm import tqdm

_TB_LOG_BATCHES = 10
_QUANTILE_DIR = _REPO_ROOT / "artifacts" / "quantiles"

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

DATASET_PRESETS = {
    "wikitext2": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "dataset_split": "test",
        "calibration_split": "train",
        "text_column": "text",
        "model_id": "neulab/gpt2-finetuned-wikitext103",
    },
    "wikitext103": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-raw-v1",
        "dataset_split": "test",
        "calibration_split": "train",
        "text_column": "text",
        "model_id": "neulab/gpt2-finetuned-wikitext103",
    },
}

@dataclass
class Arguments:
    """Command-line configuration consumed by the GPT-2 evaluator.

    Direct Gaussian spike-time error is the sole dynamic event-noise interface.
    Evaluation converts its dimensionless standard-deviation fraction with the
    base identity window ``2 * theta`` and uses the absolute mean and seed for one
    evaluation-wide seeded noise state.
    """

    # Dataset, backend, and model-conversion controls remain independent from the
    # stochastic timing experiment selected below.
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    task: Literal["wikitext2", "wikitext103"]
    model_id: str
    dataset_name: str | None
    dataset_config_name: str | None
    dataset_split: str
    cache_dir: str
    max_length: int
    batch_size: int
    device: Literal["cuda", "cpu"]
    max_eval_batches: int
    spiking_layernorm: bool
    spiking_attention: bool
    spiking_ln_mul: bool
    spiking_ln_log: bool
    spiking_ln_expdiff: bool
    spiking_mlp: bool
    activation: str
    theta: float
    attention_theta: float
    tau_s: float

    # Layer-wise calibration uses a deterministic training subset for collection and
    # immutable artifact ranges for validation or inference.
    calibration_mode: Literal["none", "collect", "validate", "inference"]
    calibration_path: str
    calibration_samples: int
    calibration_seed: int
    calibration_bins: int
    calibration_lower_quantile: float
    calibration_upper_quantile: float
    calibration_margin_fraction: float

    # These four fields match ViT, BERT, and RoBERTa exactly; distribution choice
    # and a separate evaluation-mode switch are intentionally absent.
    gaussian_time_noise: bool
    time_noise_std_frac: float
    time_noise_mean: float
    time_noise_seed: int

    # Quantile collection is calibration instrumentation, not dynamic noise state.
    collect_quantiles: bool
    report_clamp_stats: bool

def parse_arguments() -> Arguments:
    """Parse GPT-2 evaluation and direct Gaussian timing options.

    This function resolves WikiText presets and preserves the relative timing scale
    exactly as entered. Absolute-sigma conversion and generator installation remain
    responsibilities of :func:`evaluate_gpt2_model`.

    Returns:
        A dataset-resolved :class:`Arguments` instance.
    """
    # Keep language-model, dataset, and spiking-ablation controls independent from
    # the shared Gaussian replica parameters.
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on WikiText-2/103.")
    parser.add_argument("--experiment_name", type=str, default="gpt2_eval",
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF GPT-2, spiking: spiking_gpt2 class).")
    parser.add_argument("--task", type=str, choices=["wikitext2", "wikitext103"], default="wikitext2",
                        help="Preset task to evaluate. Sets dataset, split, and default model.")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Optional Hugging Face model ID. If omitted, task preset default is used.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional dataset name override. If omitted, task preset is used.")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Optional dataset config override. If omitted, task preset is used.")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="Optional dataset split override. If omitted, task preset is used.")
    parser.add_argument("--cache-dir", type=str, default="/data/nas/datasets/",
                        help="Hugging Face dataset cache directory.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum token length for tokenizer padding/truncation.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="If > 0, stop after this many evaluation batches for smoke testing.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--spiking-layernorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SpikingLayerNorm when --model_backend spiking is selected.")
    parser.add_argument("--spiking-attention", action=argparse.BooleanOptionalAction, default=True,
                        help="Use spiking SDPA attention when --model_backend spiking is selected.")
    parser.add_argument("--spiking-ln-mul", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 1: use ψ_M for variance.")
    parser.add_argument("--spiking-ln-log", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 2: use φ_NL for spike encoding.")
    parser.add_argument("--spiking-ln-expdiff", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 3: use ψ_ED for normalisation.")
    parser.add_argument("--spiking-mlp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SpikingConv1D in MLP layers when --model_backend spiking is selected.")
    parser.add_argument("--activation", type=str, default="gelu_new",
                        help="Activation function for the spiking backend (default: gelu_new).")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound theta used by spiking backend modules.")
    parser.add_argument(
        "--attention-theta",
        type=float,
        default=None,
        help="Attention-only theta; defaults to the global --theta value.",
    )
    parser.add_argument("--tau-s", type=float, default=1.0,
                        help="Spike-time constant tau_s used by SpikingLayerNorm.")

    # Collection and frozen execution are separate lifecycle phases. The artifact
    # path remains explicit so an experiment cannot silently reuse a stale default.
    parser.add_argument(
        "--calibration-mode",
        choices=("none", "collect", "validate", "inference"),
        default="none",
        help="Create or consume a layer-wise calibration artifact.",
    )
    parser.add_argument(
        "--calibration-path",
        type=str,
        default="",
        help="Calibration JSON path; required unless calibration mode is none.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1024,
        help="Fixed number of training texts replayed in both collection passes.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help="Seed defining the deterministic training-subset permutation.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=2048,
        help="Number of fixed histogram bins per calibrated residual site.",
    )
    parser.add_argument(
        "--calibration-lower-quantile",
        type=float,
        default=0.0,
        help="Lower histogram endpoint; defaults to the observed minimum.",
    )
    parser.add_argument(
        "--calibration-upper-quantile",
        type=float,
        default=1.0,
        help="Upper histogram endpoint; defaults to the observed maximum.",
    )
    parser.add_argument(
        "--calibration-margin-fraction",
        type=float,
        default=0.05,
        help="Per-side range expansion after endpoint selection.",
    )

    # These options match the other model evaluators so one experiment convention
    # can configure every supported architecture.
    parser.add_argument(
        "--gaussian-time-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply direct Gaussian error to every event-aware spike time.",
    )
    parser.add_argument(
        "--time-noise-std-frac",
        type=float,
        default=0.0,
        help="Gaussian time std as a fraction of the identity window 2*theta.",
    )
    parser.add_argument(
        "--time-noise-mean",
        type=float,
        default=0.0,
        help="Gaussian timing mean in absolute time units (default: 0.0).",
    )
    parser.add_argument(
        "--time-noise-seed",
        type=int,
        default=0,
        help="Seed for the evaluation-wide timing-noise generator.",
    )
    parser.add_argument("--collect-quantiles", action="store_true",
                        help="Collect and print 99.9%% quantiles of absolute activations.")
    parser.add_argument("--report-clamp-stats", action="store_true",
                        help="Aggregate and print per-site fixed-domain clamp counts.")

    # Resolve dataset defaults first and copy all Gaussian values without changing
    # units; evaluation will perform the single 2*theta conversion.
    args = parser.parse_args()
    preset = DATASET_PRESETS[args.task]
    model_id = cast(str, args.model_id or preset["model_id"])
    dataset_name = cast(str | None, args.dataset_name or preset["dataset_name"])
    dataset_config_name = cast(str | None, args.dataset_config_name if args.dataset_config_name is not None else preset["dataset_config_name"])
    dataset_split = cast(str, args.dataset_split or preset["dataset_split"])
    attention_theta = args.theta if args.attention_theta is None else args.attention_theta
    if not math.isfinite(attention_theta) or attention_theta <= 0.0:
        parser.error("--attention-theta must be finite and positive")

    return Arguments(
        experiment_name=args.experiment_name,
        model_backend=args.model_backend,
        task=args.task,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_split=dataset_split,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        max_eval_batches=args.max_eval_batches,
        spiking_layernorm=args.spiking_layernorm,
        spiking_attention=args.spiking_attention,
        spiking_ln_mul=args.spiking_ln_mul,
        spiking_ln_log=args.spiking_ln_log,
        spiking_ln_expdiff=args.spiking_ln_expdiff,
        spiking_mlp=args.spiking_mlp,
        activation=args.activation,
        theta=args.theta,
        attention_theta=attention_theta,
        tau_s=args.tau_s,
        calibration_mode=args.calibration_mode,
        calibration_path=args.calibration_path,
        calibration_samples=args.calibration_samples,
        calibration_seed=args.calibration_seed,
        calibration_bins=args.calibration_bins,
        calibration_lower_quantile=args.calibration_lower_quantile,
        calibration_upper_quantile=args.calibration_upper_quantile,
        calibration_margin_fraction=args.calibration_margin_fraction,
        gaussian_time_noise=args.gaussian_time_noise,
        time_noise_std_frac=args.time_noise_std_frac,
        time_noise_mean=args.time_noise_mean,
        time_noise_seed=args.time_noise_seed,
        collect_quantiles=args.collect_quantiles,
        report_clamp_stats=args.report_clamp_stats,
    )


def validate_gpt2_calibration_arguments(
    args: Arguments,
) -> CalibrationMode | None:
    """Validate GPT-2 calibration controls before loading external resources.

    ``none`` preserves analytic fixed ranges. Collection is restricted to a clean
    deterministic spiking model, while validation and inference may combine the
    already frozen table with independently configured Gaussian timing noise.

    Args:
        args: Parsed GPT-2 evaluator configuration.

    Returns:
        Internal calibration mode, or ``None`` when calibration is disabled.

    Raises:
        TypeError: If calibration fields have incompatible scalar types.
        ValueError: If paths, counts, quantiles, margins, or backend/noise combinations
            are invalid for the selected lifecycle phase.
    """
    # Translate the user-facing disabled state separately because CalibrationMode
    # contains only active phases shared by collectors and frozen runtimes.
    if not isinstance(args.calibration_mode, str):
        raise TypeError("calibration_mode must be a string")
    if args.calibration_mode == "none":
        return None
    try:
        mode = CalibrationMode(args.calibration_mode)
    except ValueError as error:
        raise ValueError("unsupported calibration_mode") from error
    if args.model_backend != "spiking":
        raise ValueError("layer-wise calibration requires model_backend=spiking")
    if not isinstance(args.calibration_path, str):
        raise TypeError("calibration_path must be a string")
    if not args.calibration_path.strip():
        raise ValueError("calibration_path is required for active calibration")

    # Counts determine the exact training population and histogram representation.
    # Reject Boolean aliases before any dataset or checkpoint download can begin.
    for name, value in (
        ("calibration_samples", args.calibration_samples),
        ("calibration_seed", args.calibration_seed),
        ("calibration_bins", args.calibration_bins),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if args.calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if args.calibration_seed < 0:
        raise ValueError("calibration_seed must be non-negative")
    if args.calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")

    # GPT-2 residual sites are signed-symmetric, so both probability cutoffs must be
    # ordered and the per-side expansion must remain finite and non-negative.
    for name, value in (
        ("calibration_lower_quantile", args.calibration_lower_quantile),
        ("calibration_upper_quantile", args.calibration_upper_quantile),
        ("calibration_margin_fraction", args.calibration_margin_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= args.calibration_lower_quantile <= 1.0:
        raise ValueError("calibration_lower_quantile must lie in [0, 1]")
    if not 0.0 <= args.calibration_upper_quantile <= 1.0:
        raise ValueError("calibration_upper_quantile must lie in [0, 1]")
    if args.calibration_lower_quantile > args.calibration_upper_quantile:
        raise ValueError("calibration quantiles must be ordered")
    if args.calibration_margin_fraction < 0.0:
        raise ValueError("calibration_margin_fraction must be non-negative")

    # Collection measures only the clean deterministic residual distribution. Frozen
    # phases may add timing noise after exact artifact metadata compatibility succeeds.
    if mode is CalibrationMode.COLLECT and args.gaussian_time_noise:
        raise ValueError("calibration collection requires Gaussian timing noise off")
    return mode

def infer_text_column(column_names: list[str], preferred: str | None = None) -> str:
    if preferred is not None and preferred in column_names:
        return preferred

    for candidate in ("text", "content", "sentence"):
        if candidate in column_names:
            return candidate

    raise ValueError(f"No supported text column found in dataset columns: {column_names}")

def evaluate_gpt2_model(args: Arguments) -> None:
    """Evaluate one GPT-2 backend with optional direct Gaussian event timing.

    The evaluator converts ``time_noise_std_frac`` to absolute time using the base
    identity-code window ``2 * theta`` and installs one seeded process-wide noise
    state. Causal-language-model loss and perplexity aggregation remain unchanged;
    Gaussian event and saturation diagnostics are emitted after the task loop.

    Args:
        args: Parsed GPT-2 dataset, conversion, and timing-noise settings.

    Raises:
        RuntimeError: If a Gaussian-enabled model is wrapped in ``DataParallel``.
        ValueError: If the shared Gaussian configuration rejects its parameters.
    """
    # Resolve model and dataset identity once so timing logs and task results refer
    # to the same effective evaluation configuration.
    model_backend = args.model_backend
    model_id = cast(str, args.model_id)
    dataset_name = cast(str | None, args.dataset_name)
    dataset_config_name = cast(str | None, args.dataset_config_name)
    dataset_split = cast(str, args.dataset_split)
    max_length = args.max_length
    batch_size = args.batch_size
    max_eval_batches = args.max_eval_batches
    device_str = args.device

    torch_device = torch.device(device_str)
    calibration_mode = validate_gpt2_calibration_arguments(args)

    # Convert the common user-facing fraction exactly once. tau_s does not rescale
    # this value; every encoder receives the same absolute sigma based on 2*theta.
    identity_time_window = 2.0 * float(args.theta)
    time_noise_std = float(args.time_noise_std_frac) * identity_time_window
    gaussian_enabled = bool(
        model_backend == "spiking" and args.gaussian_time_noise
    )

    # Install one evaluation-wide seeded generator and clear previous counters.
    # Explicitly disable it for the dense HF backend in reused Python processes.
    set_gaussian_time_noise(
        enabled=gaussian_enabled,
        time_std=time_noise_std,
        time_mean=args.time_noise_mean,
        seed=args.time_noise_seed,
        device=torch_device,
    )

    # Log both relative and absolute timing scales so theta sweeps do not obscure
    # the physical perturbation applied by the shared encoder boundary.
    cfg = {
        **vars(args),
        "gaussian_time_noise_effective": gaussian_enabled,
        "identity_time_window": identity_time_window,
        "time_noise_std": time_noise_std,
    }
    effective_attn_impl = "eager"
    if model_backend == "spiking" and torch_device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl
    wandb.init(entity="CIDA", project="gpt2-evaluation", config=cfg, name=args.experiment_name)
    print(f"Using device: {torch_device}")
    print(f"Model backend: {model_backend}")
    print(
        "Gaussian time noise — "
        f"enabled: {gaussian_enabled}, "
        f"std_frac: {args.time_noise_std_frac}, "
        f"identity_window: {identity_time_window}, "
        f"std_abs: {time_noise_std}, "
        f"mean_abs: {args.time_noise_mean}, "
        f"seed: {args.time_noise_seed}"
    )
    if model_backend == "spiking":
        print(
            "Spiking config - "
            f"ln:{args.spiking_layernorm}, attn:{args.spiking_attention}, "
            f"mul:{args.spiking_ln_mul}, log:{args.spiking_ln_log}, "
            f"expdiff:{args.spiking_ln_expdiff}, mlp:{args.spiking_mlp}, "
            f"act:{args.activation}, theta:{args.theta}, "
            f"attention_theta:{args.attention_theta}, tau_s:{args.tau_s}"
        )

    # Evaluation and calibration use disjoint splits. Collection never loads test
    # examples, while frozen phases reconstruct the exact training-subset identity
    # before evaluating the untouched requested split.
    assert dataset_name is not None
    calibration_split = cast(
        str,
        DATASET_PRESETS.get(args.task, {}).get("calibration_split", "train"),
    )
    preferred_text_column = DATASET_PRESETS.get(args.task, {}).get("text_column")

    def load_requested_split(split: str) -> Any:
        """Load one evaluator split under the resolved dataset configuration."""
        # A missing dataset configuration is a supported Hugging Face dataset form;
        # avoid passing an explicit None as a positional builder configuration.
        if dataset_config_name is None:
            return load_dataset(
                dataset_name,
                split=split,
                cache_dir=args.cache_dir,
            )

        # Calibration and evaluation must share the same named configuration so the
        # artifact identity differs only by its explicitly recorded split.
        return load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )

    dataset = None
    if calibration_mode is not CalibrationMode.COLLECT:
        print(
            f"Loading evaluation dataset: {dataset_name}/{dataset_config_name} "
            f"({dataset_split})..."
        )
        dataset = load_requested_split(dataset_split)

    calibration_dataset = None
    if calibration_mode is not None:
        print(
            f"Loading calibration dataset: {dataset_name}/{dataset_config_name} "
            f"({calibration_split})..."
        )
        training_dataset = load_requested_split(calibration_split)

        # Empty WikiText rows carry no language-model tokens beyond padding and would
        # make subset identity depend on non-examples. Filter before permutation so
        # the selected fingerprint describes the actual calibration population.
        calibration_text_column = infer_text_column(
            training_dataset.column_names,
            preferred=preferred_text_column,
        )
        training_dataset = training_dataset.filter(
            lambda example: len(example[calibration_text_column].strip()) > 0
        )
        calibration_dataset = select_calibration_subset(
            training_dataset,
            sample_count=args.calibration_samples,
            seed=args.calibration_seed,
        )

    # Evaluation preserves its existing empty-line removal independently. Frozen
    # setup does not select or mutate validation examples to match calibration.
    text_column = None
    if dataset is not None:
        text_column = infer_text_column(
            dataset.column_names,
            preferred=preferred_text_column,
        )
        dataset = dataset.filter(
            lambda example: len(example[text_column].strip()) > 0
        )
    elif calibration_dataset is not None:
        text_column = infer_text_column(
            calibration_dataset.column_names,
            preferred=preferred_text_column,
        )
    if text_column is None:
        raise RuntimeError("no GPT-2 dataset was loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels = []
        for i in range(len(tokenized["input_ids"])):
            label = [
                -100 if mask == 0 else token
                for mask, token in zip(tokenized["attention_mask"][i], tokenized["input_ids"][i])
            ]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    # Tokenize both phases with one tokenizer instance and fixed padded capacity.
    # Labels remain available only because the normal evaluator computes task loss;
    # the collection helper deliberately ignores them.
    dataloader = None
    if dataset is not None:
        processed_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
        )
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        dataloader = DataLoader(
            cast(Any, processed_dataset),
            batch_size=batch_size,
            shuffle=False,
        )

    calibration_dataloader = None
    if calibration_dataset is not None:
        processed_calibration_dataset = calibration_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=calibration_dataset.column_names,
        )
        processed_calibration_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        calibration_dataloader = DataLoader(
            cast(Any, processed_calibration_dataset),
            batch_size=batch_size,
            shuffle=False,
        )

    print(f"Loading model: {model_id}...")
    model: nn.Module
    config: Any
    if model_backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        config = GPT2Config.from_pretrained(model_id)
        config.use_spiking_layernorm = args.spiking_layernorm
        config.spiking_ln_mul = args.spiking_ln_mul
        config.spiking_ln_log = args.spiking_ln_log
        config.spiking_ln_expdiff = args.spiking_ln_expdiff
        config.use_spiking_mlp = args.spiking_mlp
        config.activation_function = args.activation
        config.theta = args.theta
        config.attention_theta = args.attention_theta
        config.tau_s = args.tau_s
        model = GPT2LMHeadModel.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl)

    # GPT-2 currently constructs no DataParallel wrapper itself. Retain an explicit
    # rejection so future or externally inserted wrapping cannot share one global RNG.
    if gaussian_enabled and isinstance(model, nn.DataParallel):
        raise RuntimeError(
            "Gaussian spike-time noise does not support DataParallel; "
            "run one evaluation process per GPU"
        )
    if calibration_mode is not None and isinstance(model, nn.DataParallel):
        raise RuntimeError(
            "layer-wise calibration does not support DataParallel; "
            "run one evaluation process per GPU"
        )

    if torch_device.type == "cuda":
        model = nn.Module.cuda(model)
    else:
        model = nn.Module.cpu(model)
    model.eval()

    # Construct the clean artifact identity before collection or frozen binding. The
    # filtered-and-selected training fingerprint is required even in frozen phases so
    # a changed WikiText revision or tokenization setup fails at load time.
    calibration_metadata = None
    if calibration_mode is not None:
        if calibration_dataset is None:
            raise RuntimeError("active calibration requires a selected training subset")
        dataset_identity = (
            dataset_name
            if dataset_config_name is None
            else f"{dataset_name}:{dataset_config_name}"
        )
        calibration_metadata = build_gpt2_calibration_metadata(
            model_id=model_id,
            dataset_id=dataset_identity,
            calibration_split=calibration_split,
            calibration_dataset_fingerprint=calibration_dataset._fingerprint,
            calibration_samples=args.calibration_samples,
            calibration_seed=args.calibration_seed,
            tokenizer=tokenizer,
            config=config,
            max_length=max_length,
            attention_implementation=effective_attn_impl,
        )

    # Collection writes the immutable artifact and exits without constructing loss,
    # perplexity, validation clipping, TensorBoard, or quantile diagnostics.
    if calibration_mode is CalibrationMode.COLLECT:
        if calibration_dataloader is None or calibration_metadata is None:
            raise RuntimeError("calibration collection setup is incomplete")
        specs = gpt2_calibration_specs(
            model,
            lower_quantile=args.calibration_lower_quantile,
            upper_quantile=args.calibration_upper_quantile,
            margin_fraction=args.calibration_margin_fraction,
        )
        collector = create_calibration_collector(
            calibration_metadata,
            specs,
            bin_count=args.calibration_bins,
        )
        table = collect_gpt2_calibration_table(
            model,
            calibration_dataloader,
            collector,
            device=torch_device,
            expected_samples=args.calibration_samples,
        )
        save_calibration_table(table, args.calibration_path)
        print(
            f"Saved calibration artifact with {len(table.layers)} layer ranges "
            f"to {args.calibration_path}"
        )
        wandb.log({"Calibration/layer_ranges": len(table.layers)})
        wandb.finish()
        return

    # Validation and inference install only a table whose full clean identity matches
    # the reconstructed model, subset, tokenizer, numerical, and ablation metadata.
    calibration_state = None
    if calibration_mode in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
        if calibration_metadata is None:
            raise RuntimeError("frozen calibration setup is incomplete")
        table = load_calibration_table(args.calibration_path)
        expected_specs = gpt2_calibration_specs(
            model,
            lower_quantile=args.calibration_lower_quantile,
            upper_quantile=args.calibration_upper_quantile,
            margin_fraction=args.calibration_margin_fraction,
        )
        validate_calibration_table_specs(table, expected_specs)
        calibration_state = create_calibration_runtime(
            calibration_mode,
            table,
            expected_metadata=calibration_metadata,
        )
        bind_model_calibration(model, calibration_state)

    if dataloader is None:
        raise RuntimeError("GPT-2 evaluation requires an evaluation DataLoader")

    tb_writer = create_summary_writer(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []
    clamp_totals: dict[tuple[str, str], dict[str, int]] = {}

    def make_ln_hook(tag):
        def hook_fn(_module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value if isinstance(out, Potential) else out
                tb_writer.add_histogram(f"{tag}/input", inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(), log_step[0])
        return hook_fn

    def make_clamp_hook(name: str):
        previous_names: list[str | None] = []

        def pre_hook(_module, _inp):
            previous_names.append(types.get_current_module_name())
            types.set_current_module_name(name)

        def post_hook(_module, _inp, _out):
            previous = previous_names.pop() if previous_names else None
            types.set_current_module_name(previous)

        return pre_hook, post_hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            hooks.append(module.register_forward_hook(make_ln_hook(name)))
        
        if (
            model_backend == "spiking"
            and args.report_clamp_stats
            and isinstance(
                module,
                (GPT2Attention, SpikingLayerNorm, SpikingLinear, SpikingConv1D),
            )
        ):
            pre_h, post_h = make_clamp_hook(name)
            hooks.append(module.register_forward_pre_hook(pre_h))
            hooks.append(module.register_forward_hook(post_h))

    quantiles = []
    def make_quantile_hook():
        def hook_fn(module, inp, out):
            val = out.value if isinstance(out, Potential) else out
            if isinstance(val, torch.Tensor):
                val_flat = val.detach().abs().float().view(-1)
                if val_flat.numel() > 16000000:
                    step = val_flat.numel() // 16000000 + 1
                    val_flat = val_flat[::step]
                q = torch.quantile(val_flat, 0.999).item()
                quantiles.append(q)
        return hook_fn

    if args.collect_quantiles:
        from transformers.pytorch_utils import Conv1D
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, SpikingLayerNorm, SpikingLinear, SpikingConv1D, Conv1D)):
                hooks.append(module.register_forward_hook(make_quantile_hook()))

    if model_backend == "spiking" and args.report_clamp_stats:
        types.clear_clamp_stats()
        types.set_current_module_name(None)
        types.set_clamp_log_enabled(True)

    print("Starting evaluation...")

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        labels = batch["labels"].to(torch_device)

        if model_backend == "spiking" and args.report_clamp_stats:
            types.clear_clamp_stats()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Log clamp stats
        clamp_stats = (
            types.get_clamp_stats()
            if model_backend == "spiking" and args.report_clamp_stats
            else {}
        )
        for (module_name, clamp_name), stats in clamp_stats.items():
            aggregate = clamp_totals.setdefault(
                (module_name, clamp_name),
                {"underflow": 0, "overflow": 0, "total": 0},
            )
            for field in ("underflow", "overflow", "total"):
                aggregate[field] += stats[field]
            total = stats["total"]
            if total > 0:
                underflow_ratio = stats["underflow"] / total
                overflow_ratio = stats["overflow"] / total
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/underflow", underflow_ratio, log_step[0])
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/overflow", overflow_ratio, log_step[0])
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/total_clamped", underflow_ratio + overflow_ratio, log_step[0])

        loss = outputs.loss

        if not torch.isnan(loss):
            total_loss += loss.item()
            total_steps += 1
            wandb.log({"Batch Loss": loss.item(), "Batch Perplexity": math.exp(min(loss.item(), 20.0))})

        log_step[0] += 1
        if max_eval_batches > 0 and log_step[0] >= max_eval_batches:
            break

    for h in hooks:
        h.remove()
    tb_writer.close()
    types.set_current_module_name(None)
    if model_backend == "spiking" and args.report_clamp_stats:
        types.set_clamp_log_enabled(False)

    if args.collect_quantiles and quantiles:
        max_q = max(quantiles)
        print(f"RESULT_QUANTILE: {max_q}")
        _QUANTILE_DIR.mkdir(parents=True, exist_ok=True)
        with (_QUANTILE_DIR / f"quantile_gpt2_{args.task}.txt").open("w") as f:
            f.write(str(max_q))

    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < float("inf") else float("inf")

    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    wandb.log({"Final Average Loss": avg_loss, "Final Perplexity": perplexity})

    for (module_name, clamp_name), stats in sorted(clamp_totals.items()):
        total = stats["total"]
        underflow_rate = stats["underflow"] / total if total else 0.0
        overflow_rate = stats["overflow"] / total if total else 0.0
        site = f"{module_name}/{clamp_name}"
        print(
            f"Clamp[{site}] values={total}, underflows={stats['underflow']} "
            f"(rate={underflow_rate:.6g}), overflows={stats['overflow']} "
            f"(rate={overflow_rate:.6g})"
        )

    # Convert only the maintained raw counters into report rates, preserving event
    # and output denominators as separate physical populations.
    if gaussian_enabled:
        for site, counts in sorted(get_gaussian_noise_stats().items()):
            events = counts["events"]
            outputs = counts["outputs"]
            miss_rate = counts["misses"] / events if events else 0.0
            underflow_rate = (
                counts["output_underflows"] / outputs if outputs else 0.0
            )
            overflow_rate = (
                counts["output_overflows"] / outputs if outputs else 0.0
            )
            print(
                f"Gaussian[{site}] events={events}, misses={counts['misses']} "
                f"(rate={miss_rate:.6g}), outputs={outputs}, "
                f"underflows={counts['output_underflows']} "
                f"(rate={underflow_rate:.6g}), "
                f"overflows={counts['output_overflows']} "
                f"(rate={overflow_rate:.6g})"
            )
            wandb.log({
                f"Gaussian/{site}/events": events,
                f"Gaussian/{site}/misses": counts["misses"],
                f"Gaussian/{site}/miss_rate": miss_rate,
                f"Gaussian/{site}/outputs": outputs,
                f"Gaussian/{site}/output_underflows": counts["output_underflows"],
                f"Gaussian/{site}/output_underflow_rate": underflow_rate,
                f"Gaussian/{site}/output_overflows": counts["output_overflows"],
                f"Gaussian/{site}/output_overflow_rate": overflow_rate,
            })

    # Frozen calibration counts raw element excursions before every declared clamp.
    # Report underflow and overflow separately, then remove only the model binding;
    # the completed runtime state remains available to this reporting code unchanged.
    if calibration_state is not None:
        for item in get_calibration_clipping_report(calibration_state):
            site = f"{item.module_name}/{item.tensor_name}"
            print(
                f"Calibration[{site}] values={item.num_values}, "
                f"underflows={item.underflows} "
                f"(rate={item.underflow_rate:.6g}), "
                f"overflows={item.overflows} "
                f"(rate={item.overflow_rate:.6g})"
            )
            wandb.log({
                f"Calibration/{site}/values": item.num_values,
                f"Calibration/{site}/underflows": item.underflows,
                f"Calibration/{site}/underflow_rate": item.underflow_rate,
                f"Calibration/{site}/overflows": item.overflows,
                f"Calibration/{site}/overflow_rate": item.overflow_rate,
            })
        clear_model_calibration(model, expected_state=calibration_state)

    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_gpt2_model(args)
