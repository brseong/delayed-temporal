import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def generate_lp_dataset(num_samples: int, 
                        vector_dim: int, 
                        p: float = 2.,
                        max_val: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    거리 예측을 위한 데이터셋을 생성합니다.

    Args:
        num_samples: 생성할 샘플의 수.
        vector_dim: 각 벡터의 차원.
        p: 거리 척도 (기본값: 2, L2 거리).
        max_val: 벡터 요소의 최대값.

    Returns:
        X: 입력 데이터 (두 벡터가 수평으로 결합됨). Shape: (num_samples, 2, vector_dim)
        y: 출력 레이블 (Lp 거리). Shape: (num_samples, 1)
    """
    
    # 1. 두 개의 랜덤 벡터 세트 생성
    # np.random.uniform을 사용하여 (num_samples, vector_dim) 크기의 행렬 두 개를 생성
    X = np.random.uniform(0, 1, size=(num_samples, 2, vector_dim))

    # 2. Lp 거리 계산 (레이블 y)
    # axis=1을 기준으로 합산하여 각 샘플(행)의 Lp 거리를 계산
    y = np.linalg.norm(X[:, 0, :] - X[:, 1, :], ord=p, axis=1, keepdims=True)
    
    return X, y

def generate_1d_dot_classification_dataset(num_samples: int, num_classes: int, dim: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    거리 예측을 위한 데이터셋을 생성합니다.

    Args:
        num_samples: 생성할 샘플의 수.
    Returns:
        X: 입력 데이터 (두 벡터가 수평으로 결합됨). Shape: (num_samples, 2)
        y: 출력 레이블 (Lp 거리). Shape: (num_samples, 1)
    """
    
    # 1. 두 개의 랜덤 벡터 세트 생성
    # np.random.uniform을 사용하여 (num_samples, vector_dim) 크기의 행렬 두 개를 생성
    X = np.random.uniform(0, 1, size=(num_samples, 2*dim))
    X = X / (dim**0.5)  # Normalize to unit length

    # 2. Lp 거리 계산 (레이블 y)
    # axis=1을 기준으로 합산하여 각 샘플(행)의 Lp 거리를 계산
    x1, x2 = X[:, 0:dim], X[:, dim:2*dim] # Shape: (num_samples, dim), to keep 2D shape - inner product only in last dim
    y:np.ndarray = np.vecdot(x1, x2)
    y = (y * (num_classes-1)).round().reshape(-1, 1)
    return X, y


def generate_cosine_dataset(num_samples: int, 
                        vector_dim: int,
                        max_val: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    거리 예측을 위한 데이터셋을 생성합니다.

    Args:
        num_samples: 생성할 샘플의 수.
        vector_dim: 각 벡터의 차원.
        p: 거리 척도 (기본값: 2, L2 거리).
        max_val: 벡터 요소의 최대값.

    Returns:
        X: 입력 데이터 (두 벡터가 수평으로 결합됨). Shape: (num_samples, 2, vector_dim)
        y: 출력 레이블 (Lp 거리). Shape: (num_samples, 1)
    """
    
    # 1. 두 개의 랜덤 벡터 세트 생성
    # np.random.uniform을 사용하여 (num_samples, vector_dim) 크기의 행렬 두 개를 생성
    X = np.random.uniform(0, max_val, size=(num_samples, 2, vector_dim))

    # 2. Lp 거리 계산 (레이블 y)
    # axis=1을 기준으로 합산하여 각 샘플(행)의 Lp 거리를 계산
    x1, x2 = X[:, 0], X[:, 1]
    y = np.vecdot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1))
    return X, y

def encode_temporal(X_data:np.ndarray, time_steps:int, time_norm:bool=False):
    """
    입력 데이터를 Latency Coding(TTFS)으로 변환합니다.
    강한 입력(절댓값) -> 빠른 스파이크, 약한 입력 -> 늦은 스파이크.
    음수와 양수를 별도 채널로 분리합니다.

    Args:
        X_data (np.ndarray): 입력 데이터 (*,)
        time_steps (int): 총 시뮬레이션 시간 단계.

    Returns:
        np.ndarray: 인코딩된 스파이크 데이터 (time_steps, *X_data.shape)
    """
    max_val = 1.0
    
    if time_norm:
        # X_data -= np.min(X_data, axis=1, keepdims=True)
        X_data -= 0.0
    X_norm = ((max_val - X_data) * (time_steps-1)) / max_val
    X_pos = np.floor(X_norm).astype(np.int32)
    spikes_out = np.zeros((time_steps, *X_data.shape), dtype=np.float32)

    for indices in product(*[range(dim) for dim in X_data.shape]):
        spikes_out[X_pos[*indices], *indices] = 1.0

    return spikes_out

if __name__ == "__main__":
    # 파라미터 설정
    NUM_SAMPLES = 1000  # 총 1000 개의 샘플 생성
    VECTOR_DIM = 10      # 각 벡터는 10차원
    MAX_VAL = 50.0
    TIME_STEPS = 20     # SNN을 20 타임스텝 동안 실행

    # 1. 원본 데이터 생성
    X_data, y_data = generate_lp_dataset(NUM_SAMPLES, VECTOR_DIM, MAX_VAL)

    print(f"--- 원본 데이터 ---")
    print(f"X_data shape: {X_data.shape}, y_data shape: {y_data.shape}") # (1000, 10)
    print(f"첫 번째 샘플 (일부): {X_data[0, :5]}")

    # 2. Latency Coding 실행
    SX_data = encode_temporal(X_data, TIME_STEPS)

    print(f"\n--- Latency Coding 결과 ---")
    print(f"Spiked Data shape: {SX_data.shape}") # (1000, 20, 20)
    # (N, T, D*4) -> (1000, 20, 5*4=20)

    # ==========================================================
    # 시각화: 첫 번째 샘플 인코딩 결과 비교
    # ==========================================================
    def plot_spike_raster(spike_data, title):
        """스파이크 데이터를 래스터 플롯으로 시각화합니다."""
        # spike_data shape: (time_steps, num_features)
        time_steps, num_features = spike_data.shape
        
        # 스파이크가 발생한 시간(y)과 뉴런 인덱스(x) 찾기
        time_idx, neuron_idx = np.where(spike_data.T == 1)
        
        plt.figure(figsize=(10, 4))
        plt.scatter(neuron_idx, time_idx, marker='|', s=100, color='black')
        plt.xlabel("Neuron Index")
        plt.ylabel("Time Step")
        plt.title(title)
        plt.yticks(range(0, num_features, 2))
        plt.xticks(range(0, time_steps, 2))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    print("\n--- 첫 번째 샘플 시각화 ---")

    # 원본 데이터 (첫 5개 피처만 확인)
    original_sample = X_data[0]
    print(f"Original Data[0] (first 5): {original_sample[:5]}")
    # 정규화된 값 (참고용)
    norm_sample = original_sample / np.max(X_data)
    print(f"Normalized Data[0] (first 5): {norm_sample[:5]}")

    # Latency Coding 시각화
    # (T, num_features*2)
    plot_spike_raster(SX_data[0], f"Sample 0 - Latency Coding (TTFS)")