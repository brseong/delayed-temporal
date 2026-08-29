from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch, wandb, argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForImageClassification
from transformers.models.vit import ViTImageProcessor
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification, SpikingLayerNorm
from utils.transforms.types import Potential
from utils.transforms.calibration import (
    CalibrationMode,
    create_calibration_collector,
    create_calibration_runtime,
    get_calibration_clipping_report,
    load_calibration_table,
    save_calibration_table,
)
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    install_device_mismatch,
    set_gaussian_time_noise,
)
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.calibration import bind_model_calibration, clear_model_calibration
from utils.transformers.models.spiking_vit.calibration import (
    build_vit_calibration_metadata,
    collect_vit_calibration_table,
    image_processor_pixel_bounds,
    select_calibration_subset,
    vit_calibration_specs,
)
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
import evaluate
from tqdm import tqdm

_TB_LOG_BATCHES = 10  # 처음 N 배치에서만 히스토그램 로그
_QUANTILE_DIR = _REPO_ROOT / "artifacts" / "quantiles"

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@dataclass
class Arguments:
    """Command-line configuration consumed by the ViT evaluator.

    Dynamic event noise is represented only by direct Gaussian spike-time error.
    ``time_noise_std_frac`` is dimensionless and is converted later by the
    evaluation function to an absolute standard deviation using the identity-code
    window ``2 * theta``. Static threshold mismatch and learned-parameter
    perturbations remain independent experiment axes.
    """

    # Evaluation, backend, and model-conversion controls are independent of the
    # selected non-ideality experiments.
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    model_id: str
    dataset_id: str
    batch_size: int
    device: Literal["cuda", "cpu"]
    precision: Literal["float32", "float64", "bfloat16", "float16"]
    max_eval_batches: int
    spiking_layernorm: bool
    spiking_attention: bool
    spiking_ln_mul: bool
    spiking_ln_log: bool
    spiking_ln_expdiff: bool
    spiking_mlp: bool
    spiking_mlp_exact_gelu: bool
    spiking_mlp_exact_gelu_layers: tuple[int, ...]
    activation: Literal["relu", "gelu"]
    theta: float

    # Layer-wise calibration is an explicit artifact lifecycle. Collection uses a
    # deterministic subset of the training split; frozen phases only load and apply
    # the resulting table while recording clipping statistics.
    calibration_mode: Literal["none", "collect", "validate", "inference"]
    calibration_path: str
    calibration_samples: int
    calibration_seed: int
    calibration_bins: int
    calibration_lower_quantile: float
    calibration_upper_quantile: float
    calibration_margin_fraction: float

    # Direct Gaussian timing noise uses one relative input scale, one absolute mean,
    # and one replica seed shared by every event-aware encoder.
    gaussian_time_noise: bool
    time_noise_std_frac: float
    time_noise_mean: float
    time_noise_seed: int

    # Static device and parameter non-idealities remain separate from event timing
    # so their effects can be swept and attributed independently.
    mismatch_enabled: bool
    mismatch_theta_std: float
    weight_noise_std: float
    bias_noise_std: float

    # Diagnostic and smoke-evaluation controls do not alter operator definitions.
    collect_quantiles: bool
    quick_test: bool

def parse_arguments() -> Arguments:
    """Parse the ViT evaluator command line into its typed configuration.

    The maintained dynamic-noise interface exposes one direct Gaussian timing
    model. Its standard deviation is entered as a fraction of the identity-code
    window and converted to absolute time inside evaluation, while the mean and
    seed are already absolute replica parameters. Static mismatch and parameter
    perturbation options remain independent.

    Returns:
        A fully populated :class:`Arguments` instance.
    """
    # General evaluation and spiking-ablation options remain unchanged so this
    # migration affects only the dynamic event-noise interface.
    parser = argparse.ArgumentParser(description="Evaluate ViT model with Spiking SDPA attention.")
    parser.add_argument("--experiment_name", type=str,
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF ViT, spiking: spiking_vit class).")
    parser.add_argument("--model_id", type=str, default="/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k",
                        help="Pretrained ViT model ID from Hugging Face.")
    parser.add_argument("--dataset_id", type=str, default="cifar10",
                        help="Dataset ID from Hugging Face datasets library.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="If > 0, stop after this many evaluation batches for smoke testing.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--precision", type=str, choices=["float32", "float64", "bfloat16", "float16"], default="float32",
                        help="PyTorch precision (dtype) to use (default: float32).")
    parser.add_argument("--spiking-layernorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SpikingLayerNorm instead of standard nn.LayerNorm.")
    parser.add_argument("--spiking-attention", action=argparse.BooleanOptionalAction, default=True,
                        help="Use spiking SDPA attention instead of standard eager attention.")
    parser.add_argument("--spiking-ln-mul", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 1: use ψ_M for variance (vs direct x²).")
    parser.add_argument("--spiking-ln-log", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 2: use φ_NL for spike encoding (vs standard log).")
    parser.add_argument("--spiking-ln-expdiff", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 3: use ψ_ED for normalisation (vs direct exp).")
    parser.add_argument("--spiking-mlp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use φ_NL clip activation in MLP (vs GELU). Implements ψ_L via PWM.")
    parser.add_argument("--spiking-mlp-exact-gelu", action=argparse.BooleanOptionalAction, default=False,
                        help="Replace every temporal GELU with the same tanh formula evaluated densely.")
    parser.add_argument(
        "--spiking-mlp-exact-gelu-layers",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Zero-based ViT encoder layers whose temporal GELU is replaced by "
            "the same tanh formula evaluated densely."
        ),
    )
    parser.add_argument("--activation", type=str, choices=["relu", "gelu"], default="gelu",
                        help="Activation function to use when --no-spiking-mlp is set (default: gelu).")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound θ for SpikingLayerNorm clamping (default: 100.0).")

    # Layer-wise calibration is intentionally separate from the old diagnostic
    # quantile hook. Collection writes one reusable artifact from a deterministic
    # training subset, while validate/inference only consume that frozen artifact.
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
        help="Fixed number of training samples selected for both collection passes.",
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
        help="Number of fixed histogram bins per calibrated layer output.",
    )
    parser.add_argument(
        "--calibration-lower-quantile",
        type=float,
        default=0.001,
        help="Lower signed residual quantile retained during calibration.",
    )
    parser.add_argument(
        "--calibration-upper-quantile",
        type=float,
        default=0.999,
        help="Upper signed residual quantile retained during calibration.",
    )
    parser.add_argument(
        "--calibration-margin-fraction",
        type=float,
        default=0.05,
        help="Per-side residual range expansion after quantile selection.",
    )

    # Direct Gaussian spike-time noise uses the common four-option CLI shared by
    # every model family.
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
        help="Seed for the evaluator replica's dedicated timing-noise generator.",
    )

    # Static threshold mismatch and learned-parameter perturbations deliberately
    # remain separate controls rather than being folded into event timing noise.
    parser.add_argument("--mismatch-enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="[C] Static per-neuron threshold mismatch (frozen).")
    parser.add_argument("--mismatch-theta-std", type=float, default=0.0,
                        help="[C] σ_θ: per-neuron θ offset std, relative to θ.")
    parser.add_argument("--weight-noise-std", type=float, default=0.0,
                        help="Standard deviation of Gaussian noise to add to weights (default: 0.0).")
    parser.add_argument("--bias-noise-std", type=float, default=0.0,
                        help="Standard deviation of Gaussian noise to add to biases (default: 0.0).")
    parser.add_argument("--collect-quantiles", action="store_true",
                        help="Collect and print 99.9%% quantiles of absolute activations.")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a quick test with a small subset of the dataset and fewer batches.")

    # Parse once, then copy every field explicitly into the dataclass so omissions
    # or stale option names fail visibly during this staged interface migration.
    args = parser.parse_args()
    return Arguments(
        experiment_name=args.experiment_name,
        model_backend=args.model_backend,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        max_eval_batches=args.max_eval_batches,
        spiking_layernorm=args.spiking_layernorm,
        spiking_attention=args.spiking_attention,
        spiking_ln_mul=args.spiking_ln_mul,
        spiking_ln_log=args.spiking_ln_log,
        spiking_ln_expdiff=args.spiking_ln_expdiff,
        spiking_mlp=args.spiking_mlp,
        spiking_mlp_exact_gelu=args.spiking_mlp_exact_gelu,
        spiking_mlp_exact_gelu_layers=tuple(args.spiking_mlp_exact_gelu_layers),
        activation=args.activation,
        theta=args.theta,
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
        mismatch_enabled=args.mismatch_enabled,
        mismatch_theta_std=args.mismatch_theta_std,
        weight_noise_std=args.weight_noise_std,
        bias_noise_std=args.bias_noise_std,
        collect_quantiles=args.collect_quantiles,
        quick_test=args.quick_test,
    )


def validate_vit_calibration_arguments(
    args: Arguments,
) -> CalibrationMode | None:
    """Validate the ViT calibration artifact lifecycle before external setup.

    ``none`` preserves analytic fixed ranges without binding calibration state.
    Collection must use a clean deterministic spiking model, while validation and
    inference may reuse the frozen clean table under separately configured robustness
    noise. All statistical controls remain explicit and are persisted in the table.

    Args:
        args: Parsed ViT evaluator configuration.

    Returns:
        The internal calibration mode, or ``None`` when calibration is disabled.

    Raises:
        TypeError: If calibration fields have invalid scalar types.
        ValueError: If paths, counts, quantiles, margins, or backend combinations are
            invalid for the selected lifecycle phase.
    """
    # Convert the user-facing disabled value separately because CalibrationMode has
    # only the three active phases shared by collectors and frozen runtimes.
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

    # Counts and seed define the exact deterministic training subset and histogram
    # layout. Reject Boolean aliases and invalid ranges before loading any dataset.
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

    # Signed residual calibration needs ordered probability cutoffs and a
    # non-negative per-side margin. The collector performs the same validation, but
    # checking here avoids expensive model and data initialization on bad CLI input.
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

    # Collection measures only the clean deterministic model. Frozen phases are
    # allowed to add these independent robustness axes after metadata compatibility
    # has been established against the clean table.
    if mode is CalibrationMode.COLLECT and (
        args.gaussian_time_noise
        or args.mismatch_enabled
        or args.mismatch_theta_std != 0.0
        or args.weight_noise_std != 0.0
        or args.bias_noise_std != 0.0
    ):
        raise ValueError(
            "calibration collection requires timing noise, mismatch, and parameter "
            "perturbations to be disabled"
        )
    return mode

DATASET_CONFIGS = {
    "cifar10": {
        "split": "test",
        "calibration_split": "train",
        "image_key": "img",
        "label_key": "label",
    },
    "imagenet-1k": {
        "split": "validation",
        "calibration_split": "train",
        "image_key": "image",
        "label_key": "label",
    },
}

def configure_vit_exact_gelu_layers(
    model: nn.Module,
    layer_indices: tuple[int, ...],
) -> None:
    """Select ViT blocks that evaluate the maintained GELU formula densely.

    The ablation preserves both MLP affine layers and the tanh-approximation
    formula. It changes only whether the nonlinear formula is assembled from
    temporal operators, allowing an accuracy difference to be attributed to the
    selected block's temporal GELU implementation rather than to a different
    mathematical activation.

    Args:
        model: A local spiking ViT image-classification model.
        layer_indices: Unique zero-based encoder-block indices to bypass.

    Raises:
        ValueError: If an index is duplicated or outside the encoder depth.
        RuntimeError: If the supplied model does not expose the expected local
            spiking ViT encoder/intermediate topology.
    """
    # An empty selection is the normal production path. Returning before topology
    # inspection keeps the option harmless for dense backends and ordinary runs.
    if not layer_indices:
        return

    # Repeating an index usually indicates a malformed sweep condition. Reject it
    # instead of silently collapsing the experiment identity to a set.
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(
            "spiking_mlp_exact_gelu_layers must contain unique layer indices"
        )

    # Resolve the local adapter's explicit block list once. A clear topology error
    # is preferable to partially mutating a model from an incompatible backend.
    try:
        encoder_layers = model.vit.encoder.layer
    except AttributeError as exc:
        raise RuntimeError(
            "per-layer exact-GELU ablation requires the local spiking ViT topology"
        ) from exc

    # Validate the complete selection before changing any module so an invalid
    # later index cannot leave the model in a partially configured state.
    depth = len(encoder_layers)
    invalid = tuple(index for index in layer_indices if index < 0 or index >= depth)
    if invalid:
        raise ValueError(
            f"exact-GELU layer indices {invalid} are outside [0, {depth})"
        )

    # Resolve and validate every target before mutation. This preserves the same
    # all-or-nothing behavior when a custom model exposes only part of the adapter.
    intermediates = tuple(encoder_layers[index].intermediate for index in layer_indices)
    for index, intermediate in zip(layer_indices, intermediates, strict=True):
        if not hasattr(intermediate, "_spiking_mlp_exact_gelu"):
            raise RuntimeError(
                f"ViT encoder layer {index} has no selectable temporal GELU"
            )

    # Toggle only the nonlinear branch inside each selected intermediate module.
    # All unselected blocks retain the temporal GELU and share the same noise run.
    for intermediate in intermediates:
        intermediate._spiking_mlp_exact_gelu = True

    print(
        "Dense-formula GELU ablation layers: "
        + ", ".join(str(index) for index in layer_indices)
    )

def apply_parameter_noise(model: nn.Module, weight_std: float, bias_std: float):
    if weight_std <= 0 and bias_std <= 0:
        return

    print(f"Applying parameter noise: weight_std={weight_std}, bias_std={bias_std}")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and weight_std > 0:
                noise = torch.randn_like(param) * weight_std
                param.mul_(1 + noise)
            elif 'bias' in name and bias_std > 0:
                noise = torch.randn_like(param) * bias_std * param.abs().max() 
                param.add_(noise)

def evaluate_vit_model(args: Arguments) -> None:
    """Evaluate one ViT backend under the requested non-idealities.

    The evaluator converts the dimensionless timing-noise fraction to one absolute
    Gaussian standard deviation using the base identity-code window ``2 * theta``.
    That absolute value and one seeded generator are installed once for the whole
    replica. Because the configuration and generator are process-wide mutable
    state, Gaussian execution explicitly rejects multi-GPU ``DataParallel``.

    Args:
        args: Parsed ViT evaluation, conversion, and non-ideality settings.

    Raises:
        RuntimeError: If Gaussian timing noise would execute through
            ``DataParallel`` across multiple CUDA devices.
        ValueError: If Gaussian parameters fail shared noise validation.
    """
    # ---------------------------------------------------------
    # 0. 시드 설정
    # ---------------------------------------------------------
    torch.manual_seed(42)
    calibration_mode = validate_vit_calibration_arguments(args)
    
    # Precision mapping
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.precision]
    
    # ---------------------------------------------------------
    # 1. 설정 (Configuration)
    # ---------------------------------------------------------
    model_backend = args.model_backend
    model_id = args.model_id
    dataset_id = args.dataset_id
    batch_size = args.batch_size
    device_str = args.device

    ds_config = DATASET_CONFIGS.get(
        dataset_id,
        {
            "split": "test",
            "calibration_split": "train",
            "image_key": "image",
            "label_key": "label",
        },
    )
    split = ds_config["split"]
    calibration_split = ds_config["calibration_split"]
    image_key = ds_config["image_key"]
    label_key = ds_config["label_key"]

    # GPU 사용 가능 여부 확인
    device = torch.device(device_str)

    # Convert the user-facing fraction exactly once from the default identity-code
    # duration. Every decorated encoder then receives this same absolute sigma_t.
    identity_time_window = 2.0 * float(args.theta)
    time_noise_std = float(args.time_noise_std_frac) * identity_time_window
    gaussian_enabled = bool(
        model_backend == "spiking" and args.gaussian_time_noise
    )

    # Per-layer selection has meaning only inside the temporal MLP path. Reject
    # combinations that would otherwise record a requested but inactive ablation.
    if args.spiking_mlp_exact_gelu_layers and model_backend != "spiking":
        raise ValueError(
            "per-layer exact-GELU ablation requires --model_backend spiking"
        )
    if args.spiking_mlp_exact_gelu_layers and not args.spiking_mlp:
        raise ValueError(
            "per-layer exact-GELU ablation requires --spiking-mlp"
        )
    if args.spiking_mlp_exact_gelu_layers and args.spiking_mlp_exact_gelu:
        raise ValueError(
            "choose either all-layer or per-layer exact-GELU ablation, not both"
        )

    # A process-wide generator cannot represent independent per-device replica
    # streams under DataParallel, so reject that topology before external setup.
    use_data_parallel = device.type == "cuda" and torch.cuda.device_count() > 1
    if gaussian_enabled and use_data_parallel:
        raise RuntimeError(
            "Gaussian spike-time noise does not support DataParallel; "
            "run one evaluation process per GPU"
        )
    if calibration_mode is not None and use_data_parallel:
        raise RuntimeError(
            "layer-wise calibration does not support DataParallel; "
            "run one evaluation process per GPU"
        )

    # Installing a configuration starts one seeded measurement replica and clears
    # prior Gaussian counters. HF evaluation installs the disabled state explicitly.
    set_gaussian_time_noise(
        enabled=gaussian_enabled,
        time_std=time_noise_std,
        time_mean=args.time_noise_mean,
        seed=args.time_noise_seed,
        device=device,
    )

    # Log both the dimensionless input and the derived absolute quantity so runs at
    # different theta values remain interpretable without reconstructing the CLI.
    cfg = {
        **vars(args),
        "gaussian_time_noise_effective": gaussian_enabled,
        "identity_time_window": identity_time_window,
        "time_noise_std": time_noise_std,
    }
    effective_attn_impl = "eager"
    if model_backend == "spiking" and device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl

    wandb.init(entity="CIDA", project=f"vit-evaluation-{args.dataset_id}", config=cfg, name=args.experiment_name)
    print(f"Using device: {device}")
    print(f"Model backend: {model_backend}")
    print(f"Model: {model_id}, Dataset: {dataset_id} ({split})")
    print(f"Precision: {args.precision}")
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
        print(f"Spiking LayerNorm: {args.spiking_layernorm}, Spiking Attention: {args.spiking_attention}")
        if args.spiking_layernorm:
            print(f"  LN stages — mul: {args.spiking_ln_mul}, log: {args.spiking_ln_log}, expdiff: {args.spiking_ln_expdiff}")
        print(f"Spiking MLP: {args.spiking_mlp}")
        print(
            "Per-layer dense-formula GELU: "
            f"{args.spiking_mlp_exact_gelu_layers or 'none'}"
        )

    # ---------------------------------------------------------
    # 2. 데이터셋 및 전처리 도구 로드
    # ---------------------------------------------------------
    # Evaluation and calibration use disjoint dataset splits. Collection needs only
    # the training subset, while validate/inference additionally load the untouched
    # evaluation split used for task accuracy and clipping reports.
    dataset = None
    if calibration_mode is not CalibrationMode.COLLECT:
        print(f"Loading evaluation dataset: {dataset_id} ({split})...")
        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir="/data/nas/datasets/",
        )
        if args.quick_test:
            dataset = dataset.select(range(min(5000, len(dataset))))

    calibration_dataset = None
    if calibration_mode is not None:
        print(
            f"Loading calibration dataset: {dataset_id} "
            f"({calibration_split})..."
        )
        training_dataset = load_dataset(
            dataset_id,
            split=calibration_split,
            cache_dir="/data/nas/datasets/",
        )
        calibration_dataset = select_calibration_subset(
            training_dataset,
            sample_count=args.calibration_samples,
            seed=args.calibration_seed,
        )

    # 모델에 맞는 Feature Extractor(Image Processor) 로드
    if model_id == "mpiorczynski/relu-vit-base-patch16-224":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    else:
        processor = ViTImageProcessor.from_pretrained(model_id)

    # Collection writes an artifact and exits without touching validation labels or
    # metrics. Frozen and calibration-free evaluation retain the existing accuracy.
    metric_int = None
    metric_tot = None
    if calibration_mode is not CalibrationMode.COLLECT:
        metric_int = evaluate.load("accuracy")
        metric_tot = evaluate.load("accuracy")

    # ---------------------------------------------------------
    # 3. 데이터 전처리 함수 정의
    # ---------------------------------------------------------
    def transform(examples):
        # 이미지 데이터를 RGB로 변환 (흑백 이미지가 섞여 있을 경우 대비)
        images = [x.convert("RGB") for x in examples[image_key]]

        # ViT 입력 형태에 맞게 리사이즈 및 정규화
        inputs = processor(images, return_tensors="pt")

        # 'pixel_values'는 모델의 입력, 'labels'는 정답
        inputs["labels"] = examples[label_key]
        return inputs

    # Evaluation order does not affect accuracy, but sequential sampling makes smoke
    # runs and clipping reports reproducible. The calibration loader must reuse the
    # exact selected dataset object with shuffle disabled for both collection passes.
    dataloader = None
    if dataset is not None:
        processed_dataset = dataset.with_transform(transform)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    calibration_dataloader = None
    if calibration_dataset is not None:
        processed_calibration_dataset = calibration_dataset.with_transform(transform)
        calibration_dataloader = DataLoader(
            processed_calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    # ---------------------------------------------------------
    # 4. 모델 로드
    # ---------------------------------------------------------
    print(f"Loading model: {model_id}...")
    
    if model_backend == "hf":
        config = ViTConfig.from_pretrained(model_id, hidden_act=args.activation)
        model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=dtype, config=config)
    else:
        config = ViTConfig.from_pretrained(
            model_id,
            use_spiking_layernorm=args.spiking_layernorm,
            spiking_ln_mul=args.spiking_ln_mul,
            spiking_ln_log=args.spiking_ln_log,
            spiking_ln_expdiff=args.spiking_ln_expdiff,
            use_spiking_mlp=args.spiking_mlp,
            spiking_mlp_exact_gelu=args.spiking_mlp_exact_gelu,
            hidden_act=args.activation,
            theta=args.theta,
        )
        pixel_domain = image_processor_pixel_bounds(
            processor,
            num_channels=int(config.num_channels),
        )
        config.pixel_value_min = pixel_domain.min
        config.pixel_value_max = pixel_domain.max
        model = ViTForImageClassification.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl, torch_dtype=dtype)

        configure_vit_exact_gelu_layers(
            model,
            args.spiking_mlp_exact_gelu_layers,
        )
    
    # Build the clean artifact identity before applying any robustness perturbation.
    # The selected dataset fingerprint includes the training data revision and exact
    # seeded subset indices, while model options describe the converted architecture.
    calibration_metadata = None
    if calibration_mode is not None:
        if calibration_dataset is None:
            raise RuntimeError("active calibration requires a selected training subset")
        calibration_metadata = build_vit_calibration_metadata(
            model_id=model_id,
            dataset_id=dataset_id,
            calibration_split=calibration_split,
            calibration_dataset_fingerprint=calibration_dataset._fingerprint,
            calibration_samples=args.calibration_samples,
            calibration_seed=args.calibration_seed,
            processor=processor,
            config=config,
            dtype=args.precision,
            attention_implementation=effective_attn_impl,
        )

    # Collection must observe the clean converted checkpoint. Frozen validation and
    # inference deliberately apply independent parameter noise only after the clean
    # metadata identity has been constructed.
    if calibration_mode is not CalibrationMode.COLLECT:
        apply_parameter_noise(model, args.weight_noise_std, args.bias_noise_std)

    model.to(device)
    model.eval()

    # Static device mismatch (frozen per-neuron threshold offsets) via forward pre-hooks.
    # Installed after .to(device) so offsets are sampled on the model's device.
    if (
        calibration_mode is not CalibrationMode.COLLECT
        and model_backend == "spiking"
        and args.mismatch_enabled
        and args.mismatch_theta_std > 0
    ):
        handles = install_device_mismatch(model, theta_std=args.mismatch_theta_std, enabled=True)
        print(f"Installed static device mismatch on {len(handles)} spiking modules (σ_θ={args.mismatch_theta_std}).")

    # Collection executes the deterministic training subset twice and terminates
    # after atomically writing the artifact. No validation example or task metric is
    # touched in this phase.
    if calibration_mode is CalibrationMode.COLLECT:
        if calibration_dataloader is None or calibration_metadata is None:
            raise RuntimeError("calibration collection setup is incomplete")
        specs = vit_calibration_specs(
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
        table = collect_vit_calibration_table(
            model,
            calibration_dataloader,
            collector,
            device=device,
            dtype=dtype,
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

    # Validation and inference reject any table collected under different data,
    # preprocessing, numerical, capacity, or model-path metadata before binding the
    # immutable ranges to their named ViT blocks.
    calibration_state = None
    if calibration_mode in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
        if calibration_metadata is None:
            raise RuntimeError("frozen calibration setup is incomplete")
        table = load_calibration_table(args.calibration_path)
        calibration_state = create_calibration_runtime(
            calibration_mode,
            table,
            expected_metadata=calibration_metadata,
        )
        bind_model_calibration(model, calibration_state)

    # GPU 병렬화 (DataParallel) 설정
    if use_data_parallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
        
    model.eval() # 평가 모드로 전환

    # ---------------------------------------------------------
    # 5. TensorBoard 히스토그램 훅 등록
    # ---------------------------------------------------------
    tb_writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []

    def make_ln_hook(tag, theta):
        def hook_fn(module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value    if isinstance(out,    Potential) else out
                
                # Analysis of centered input (x_err)
                x = inp_val.detach().float()
                x_mean = x.mean(dim=-1, keepdim=True)
                x_err = x - x_mean
                max_abs_err = x_err.abs().max().item()
                std_err = x_err.std().item()
                
                if max_abs_err > theta:
                    print(f"[CLAMPING ALERT] {tag}: max_abs_err={max_abs_err:.2f} > theta={theta:.2f}, std={std_err:.2f}")
                
                tb_writer.add_histogram(f"{tag}/input",  inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(),  log_step[0])
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            hooks.append(module.register_forward_hook(make_ln_hook(name, args.theta)))

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
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv2d, SpikingLayerNorm)):
                hooks.append(module.register_forward_hook(make_quantile_hook()))

    # ---------------------------------------------------------
    # 6. 평가 루프 (Evaluation Loop)
    # ---------------------------------------------------------
    if dataloader is None or metric_int is None or metric_tot is None:
        raise RuntimeError("evaluation dataset or accuracy metric setup is incomplete")
    print("Starting evaluation...")

    for batch in tqdm(dataloader):
        # 데이터를 디바이스(GPU/CPU)로 이동
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        labels = batch["labels"].to(device)
        
        if log_step[0] == 0:
            print(f"[DEBUG] Ground Truth Labels for Batch 0: {labels.tolist()}")

        # 예측 (Gradients 계산 불필요)
        with torch.no_grad():
            outputs = model(pixel_values)

        log_step[0] += 1

        # Logits에서 가장 높은 확률을 가진 클래스 인덱스 추출
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 배치 단위로 메트릭에 추가
        metric_tot.add_batch(predictions=predictions, references=labels)
        wandb.log({"Intermediate accuracy": metric_int.compute(predictions=predictions, references=labels)["accuracy"]})

        if args.max_eval_batches > 0 and log_step[0] >= args.max_eval_batches:
            break

    for h in hooks:
        h.remove()
    tb_writer.close()

    if args.collect_quantiles and quantiles:
        max_q = max(quantiles)
        print(f"RESULT_QUANTILE: {max_q}")
        _QUANTILE_DIR.mkdir(parents=True, exist_ok=True)
        with (_QUANTILE_DIR / f"quantile_vit_{args.model_id.replace('/', '_')}.txt").open("w") as f:
            f.write(str(max_q))

    # ---------------------------------------------------------
    # 6. 최종 결과 계산 및 출력
    # ---------------------------------------------------------
    final_score = metric_tot.compute()
    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Accuracy: {final_score['accuracy']:.4f}")
    wandb.log({"Final Accuracy": final_score["accuracy"]})

    # Report event delivery and physical rail saturation with their own denominators.
    # The stored statistics remain limited to the five maintained raw count fields.
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

    # Layer-wise clipping uses the number of tensor elements at each residual as its
    # denominator. Report both counts and rates without changing the frozen table,
    # then remove only model bindings while preserving the completed runtime snapshot.
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
    evaluate_vit_model(args)
