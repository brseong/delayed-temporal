from dataclasses import dataclass
from typing import Literal

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
from utils.transforms.noise import set_spike_time_noise, install_device_mismatch
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
import evaluate
from tqdm import tqdm

_TB_LOG_BATCHES = 10  # 처음 N 배치에서만 히스토그램 로그

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@dataclass
class Arguments:
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
    noise_std: float
    # --- neuromorphic noise model (see utils/transforms/noise.py) ---
    jitter_enabled: bool
    jitter_mode: Literal["potential", "time"]
    hazard_enabled: bool
    hazard_delta_u: float
    hazard_insert_rate: float
    mismatch_enabled: bool
    mismatch_theta_std: float
    # --- static synaptic parameter noise (applied to loaded weights) ---
    weight_noise_std: float
    bias_noise_std: float
    collect_quantiles: bool
    quick_test: bool

def parse_arguments():
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
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Jitter σ as a fraction of the (potential or time) domain range (default: 0.0).")
    # --- neuromorphic noise model: three independently-toggleable components ---
    parser.add_argument("--jitter-enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="[A] Operating-point-dependent temporal jitter of magnitude --noise-std.")
    parser.add_argument("--jitter-mode", type=str, choices=["potential", "time"], default="potential",
                        help="[A] 'potential' (V-referred, σ_t=σ_V/|dV/dt|) or 'time' (legacy, uniform on t).")
    parser.add_argument("--hazard-enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="[B] Escape-noise drop/insertion hazard near threshold.")
    parser.add_argument("--hazard-delta-u", type=float, default=0.0,
                        help="[B] Δu (soft-threshold width) as a fraction of the potential-domain range.")
    parser.add_argument("--hazard-insert-rate", type=float, default=0.0,
                        help="[B] ρ₀, baseline spurious-spike (insertion) probability.")
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
        noise_std=args.noise_std,
        jitter_enabled=args.jitter_enabled,
        jitter_mode=args.jitter_mode,
        hazard_enabled=args.hazard_enabled,
        hazard_delta_u=args.hazard_delta_u,
        hazard_insert_rate=args.hazard_insert_rate,
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

def evaluate_vit_model(args:Arguments):
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

    cfg = vars(args)
    effective_attn_impl = "eager"
    if model_backend == "spiking" and device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl

    wandb.init(entity="CIDA", project=f"vit-evaluation-{args.dataset_id}", config=cfg, name=args.experiment_name)
    print(f"Using device: {device}")
    print(f"Model backend: {model_backend}")
    print(f"Model: {model_id}, Dataset: {dataset_id} ({split})")
    print(f"Precision: {args.precision}")
    
    if model_backend == "spiking":
        print(f"Spiking LayerNorm: {args.spiking_layernorm}, Spiking Attention: {args.spiking_attention}")
        if args.spiking_layernorm:
            print(f"  LN stages — mul: {args.spiking_ln_mul}, log: {args.spiking_ln_log}, expdiff: {args.spiking_ln_expdiff}")
        print(f"Spiking MLP: {args.spiking_mlp}")

        # Encoder-level noise (jitter + escape hazard). Device mismatch is installed on the
        # model after it is built (see below). All components are off unless explicitly enabled.
        if args.jitter_enabled or args.hazard_enabled:
            print(f"Noise — jitter: {args.jitter_enabled} (σ={args.noise_std}, mode={args.jitter_mode}), "
                  f"hazard: {args.hazard_enabled} (Δu={args.hazard_delta_u}, ρ₀={args.hazard_insert_rate})")
        set_spike_time_noise(
            std=args.noise_std,
            eval_mode=True,
            mode=args.jitter_mode,
            potential_scale=args.theta,   # σ_V = std·θ for potential-mode jitter
            jitter_enabled=args.jitter_enabled,
            hazard_enabled=args.hazard_enabled,
            hazard_delta_u=args.hazard_delta_u,
            hazard_insert_rate=args.hazard_insert_rate,
        )

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
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        
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
        with open(f"quantiles/quantile_vit_{args.model_id.replace('/', '_')}.txt", "w") as f:
            f.write(str(max_q))

    # ---------------------------------------------------------
    # 6. 최종 결과 계산 및 출력
    # ---------------------------------------------------------
    final_score = metric_tot.compute()
    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Accuracy: {final_score['accuracy']:.4f}")
    wandb.log({"Final Accuracy": final_score["accuracy"]})
    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_vit_model(args)
