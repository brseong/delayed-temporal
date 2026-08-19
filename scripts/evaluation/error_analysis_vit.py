from dataclasses import dataclass
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
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    install_device_mismatch,
    set_gaussian_time_noise,
)
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
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
    activation: Literal["relu", "gelu"]
    theta: float

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
                        help="If --spiking-mlp is set, bypass approx GELU and use exact PyTorch GELU instead.")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu"], default="gelu",
                        help="Activation function to use when --no-spiking-mlp is set (default: gelu).")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound θ for SpikingLayerNorm clamping (default: 100.0).")

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
        activation=args.activation,
        theta=args.theta,
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

DATASET_CONFIGS = {
    "cifar10": {"split": "test", "image_key": "img", "label_key": "label"},
    "imagenet-1k": {"split": "validation", "image_key": "image", "label_key": "label"},
}

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

    ds_config = DATASET_CONFIGS.get(dataset_id, {"split": "test", "image_key": "image", "label_key": "label"})
    split = ds_config["split"]
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

    # A process-wide generator cannot represent independent per-device replica
    # streams under DataParallel, so reject that topology before external setup.
    use_data_parallel = device.type == "cuda" and torch.cuda.device_count() > 1
    if gaussian_enabled and use_data_parallel:
        raise RuntimeError(
            "Gaussian spike-time noise does not support DataParallel; "
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

    # ---------------------------------------------------------
    # 2. 데이터셋 및 전처리 도구 로드
    # ---------------------------------------------------------
    # 데이터셋 로드
    print(f"Loading dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split=split, cache_dir="/data/nas/datasets/")
    if args.quick_test:
        dataset = dataset.select(range(5000))  # Quick test with only 5000 samples

    # 모델에 맞는 Feature Extractor(Image Processor) 로드
    if model_id == "mpiorczynski/relu-vit-base-patch16-224":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    else:
        processor = ViTImageProcessor.from_pretrained(model_id)

    # 평가 지표(Metric) 로드 - 정확도(Accuracy)
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

    # 데이터셋에 전처리 적용 (On-the-fly 방식)
    # with_format("torch")를 사용하여 출력을 PyTorch 텐서로 설정
    processed_dataset = dataset.with_transform(transform)

    # DataLoader 생성
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True)

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
        model = ViTForImageClassification.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl, torch_dtype=dtype)
    
    apply_parameter_noise(model, args.weight_noise_std, args.bias_noise_std)

    model.to(device)

    # Static device mismatch (frozen per-neuron threshold offsets) via forward pre-hooks.
    # Installed after .to(device) so offsets are sampled on the model's device.
    if model_backend == "spiking" and args.mismatch_enabled and args.mismatch_theta_std > 0:
        handles = install_device_mismatch(model, theta_std=args.mismatch_theta_std, enabled=True)
        print(f"Installed static device mismatch on {len(handles)} spiking modules (σ_θ={args.mismatch_theta_std}).")

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

    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_vit_model(args)
