from dataclasses import dataclass
from typing import Literal

import torch, wandb, argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from transformers import AttentionInterface
from transformers.models.vit import ViTImageProcessor
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification, SpikingLayerNorm
from utils.transforms.types import Potential
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
    model_id: str
    dataset_id: Literal["cifar10"]
    batch_size: int
    device: Literal["cuda", "cpu"]
    model_hex: str
    spiking_layernorm: bool
    spiking_attention: bool
    spiking_ln_mul: bool
    spiking_ln_log: bool
    spiking_ln_expdiff: bool
    spiking_mlp: bool
    theta: float

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate ViT model with Spiking SDPA attention.")
    parser.add_argument("--experiment_name", type=str,
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_id", type=str, default="MF21377197/vit-small-patch16-224-finetuned-Cifar10",
                        help="Pretrained ViT model ID from Hugging Face.")
    parser.add_argument("--dataset_id", type=str, default="cifar10",
                        help="Dataset ID from Hugging Face datasets library.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--model_hex", type=str, default="70c547fe125b11f1a3cc0242ac11000e",
                        help="Hex identifier for the cached Abstract L2Net model.")
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
    parser.add_argument("--theta", type=float, default=10.0,
                        help="Domain bound θ for SpikingLayerNorm clamping (default: 10.0).")

    args = parser.parse_args()
    return Arguments(
        experiment_name=args.experiment_name,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        device=args.device,
        model_hex=args.model_hex,
        spiking_layernorm=args.spiking_layernorm,
        spiking_attention=args.spiking_attention,
        spiking_ln_mul=args.spiking_ln_mul,
        spiking_ln_log=args.spiking_ln_log,
        spiking_ln_expdiff=args.spiking_ln_expdiff,
        spiking_mlp=args.spiking_mlp,
        theta=args.theta,
    )

def evaluate_vit_model(args:Arguments):
    # ---------------------------------------------------------
    # 1. 설정 (Configuration)
    # ---------------------------------------------------------
    # 예시: CIFAR-10에 파인튜닝된 ViT 모델 사용 (가장 일반적인 예시)
    # model_id = "nateraw/vit-base-patch16-224-cifar10"
    model_id = args.model_id
    dataset_id = args.dataset_id
    batch_size = args.batch_size
    device_str = args.device
    model_hex = args.model_hex
    
    # GPU 사용 가능 여부 확인
    device = torch.device(device_str)
    
    cfg = vars(args)
    cfg["attn_impl"] = "spiking_sdpa" if args.spiking_attention else "eager"
    wandb.init(project="vit-evaluation", config=cfg, name=args.experiment_name)
    print(f"Using device: {device}")
    print(f"Spiking LayerNorm: {args.spiking_layernorm}, Spiking Attention: {args.spiking_attention}")
    if args.spiking_layernorm:
        print(f"  LN stages — mul: {args.spiking_ln_mul}, log: {args.spiking_ln_log}, expdiff: {args.spiking_ln_expdiff}")
    print(f"Spiking MLP: {args.spiking_mlp}")

    # ---------------------------------------------------------
    # 2. 데이터셋 및 전처리 도구 로드
    # ---------------------------------------------------------
    # 데이터셋 로드 (평가용이므로 'test' split 사용)
    print(f"Loading dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split="test")
    
    # 모델에 맞는 Feature Extractor(Image Processor) 로드
    processor = ViTImageProcessor.from_pretrained(model_id)

    # 평가 지표(Metric) 로드 - 정확도(Accuracy)
    metric_int = evaluate.load("accuracy")
    metric_tot = evaluate.load("accuracy")

    # ---------------------------------------------------------
    # 3. 데이터 전처리 함수 정의
    # ---------------------------------------------------------
    def transform(examples):
        # 이미지 데이터를 RGB로 변환 (흑백 이미지가 섞여 있을 경우 대비)
        images = [x.convert("RGB") for x in examples["img"]]
        
        # ViT 입력 형태에 맞게 리사이즈 및 정규화
        inputs = processor(images, return_tensors="pt")
        
        # 'pixel_values'는 모델의 입력, 'labels'는 정답
        inputs["labels"] = examples["label"]
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
    attn_impl = "spiking_sdpa" if args.spiking_attention else "eager"
    config = ViTConfig.from_pretrained(model_id)
    config.use_spiking_layernorm = args.spiking_layernorm
    config.spiking_ln_mul = args.spiking_ln_mul
    config.spiking_ln_log = args.spiking_ln_log
    config.spiking_ln_expdiff = args.spiking_ln_expdiff
    config.use_spiking_mlp = args.spiking_mlp
    config.theta = args.theta
    model = ViTForImageClassification.from_pretrained(model_id, config=config, attn_implementation=attn_impl)
    # model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # 모델 병렬화
    model.to(device)
    model.eval() # 평가 모드로 전환

    # ---------------------------------------------------------
    # 5. TensorBoard 히스토그램 훅 등록
    # ---------------------------------------------------------
    tb_writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []

    def make_ln_hook(tag):
        def hook_fn(module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value    if isinstance(out,    Potential) else out
                tb_writer.add_histogram(f"{tag}/input",  inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(),  log_step[0])
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            hooks.append(module.register_forward_hook(make_ln_hook(name)))

    # ---------------------------------------------------------
    # 6. 평가 루프 (Evaluation Loop)
    # ---------------------------------------------------------
    print("Starting evaluation...")

    for batch in tqdm(dataloader):
        # 데이터를 디바이스(GPU/CPU)로 이동
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # 예측 (Gradients 계산 불필요)
        with torch.no_grad():
            outputs = model(pixel_values)

        log_step[0] += 1

        # Logits에서 가장 높은 확률을 가진 클래스 인덱스 추출
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 배치 단위로 메트릭에 추가
        metric_tot.add_batch(predictions=predictions, references=labels)
        wandb.log({"Intermediate accuracy": metric_int.compute(predictions=predictions, references=labels)["accuracy"]})

    for h in hooks:
        h.remove()
    tb_writer.close()

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