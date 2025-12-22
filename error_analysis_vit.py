import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AttentionInterface
from transformers.models.vit import ViTImageProcessor
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTForImageClassification
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
import evaluate
from tqdm import tqdm

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def evaluate_vit_model():
    # ---------------------------------------------------------
    # 1. 설정 (Configuration)
    # ---------------------------------------------------------
    # 예시: CIFAR-10에 파인튜닝된 ViT 모델 사용 (가장 일반적인 예시)
    # model_id = "nateraw/vit-base-patch16-224-cifar10"
    model_id = "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    dataset_id = "cifar10"
    batch_size = 1
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
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
    metric = evaluate.load("accuracy")

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
    dataloader = DataLoader(processed_dataset, batch_size=batch_size)

    # ---------------------------------------------------------
    # 4. 모델 로드
    # ---------------------------------------------------------
    print(f"Loading model: {model_id}...")
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="spiking_sdpa")
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
        metric.add_batch(predictions=predictions, references=labels)

    # ---------------------------------------------------------
    # 6. 최종 결과 계산 및 출력
    # ---------------------------------------------------------
    final_score = metric.compute()
    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Accuracy: {final_score['accuracy']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_vit_model()