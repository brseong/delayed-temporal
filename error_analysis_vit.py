from dataclasses import dataclass
from typing import Literal

import torch, wandb, argparse
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from transformers import AttentionInterface
from transformers.models.vit import ViTImageProcessor
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward, netcache
import evaluate
from tqdm import tqdm

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
    
    return Arguments(**vars(parser.parse_args()))

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
    
    netcache.hex = model_hex
    l2net, l2net_cfg = netcache[device]
    l2net_cfg["MODEL_HEX"] = model_hex
    wandb.init(project="vit-evaluation", config=l2net_cfg, name=args.experiment_name)
    print(f"Using device: {device}")

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
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="spiking_sdpa")
    # model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # 모델 병렬화
    model.to(device)
    model.eval() # 평가 모드로 전환

    # ---------------------------------------------------------
    # 5. 평가 루프 (Evaluation Loop)
    # ---------------------------------------------------------
    print("Starting evaluation...")
    
    for batch in tqdm(dataloader):
        # 데이터를 디바이스(GPU/CPU)로 이동
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # 예측 (Gradients 계산 불필요)
        with torch.no_grad():
            outputs = model(pixel_values)
            
        # Logits에서 가장 높은 확률을 가진 클래스 인덱스 추출
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 배치 단위로 메트릭에 추가
        metric_tot.add_batch(predictions=predictions, references=labels)
        wandb.log({"Intermediate accuracy": metric_int.compute(predictions=predictions, references=labels)["accuracy"]})

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