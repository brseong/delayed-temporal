import numpy as np
import torch
from itertools import product
import matplotlib.pyplot as plt

def generate_lp_dataset(num_samples: int, 
                        vector_dim: int, 
                        p: float = 2.,
                        low: float = 0.0,
                        high: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset for distance prediction using Lp squared distance.

    Args:
        num_samples: The number of samples to generate.
        vector_dim: The dimension of each vector.
        p: The distance metric (default: 2, L2 distance).
        low: The minimum value of vector elements.
        high: The maximum value of vector elements.

    Returns:
        X: Input data (two vectors concatenated horizontally). Shape: (num_samples, 2, vector_dim)
        y: Output labels (Square of Lp distance). Shape: (num_samples, 1)
    """
    
    # 1. 두 개의 랜덤 벡터 세트 생성
    # np.random.uniform을 사용하여 (num_samples, vector_dim) 크기의 행렬 두 개를 생성
    X = np.random.uniform(low, high, size=(num_samples, 2, vector_dim))

    # 2. Lp 거리 계산 (레이블 y)
    # axis=1을 기준으로 합산하여 각 샘플(행)의 Lp 거리를 계산
    y = np.linalg.norm(X[:, 0, :] - X[:, 1, :], ord=p, axis=1, keepdims=True)
    return X, y

def generate_l2_square_dataset(num_samples: int, 
                        vector_dim: int, 
                        low: float = 0.0,
                        high: float = 10.0,
                        normalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X, y = generate_lp_dataset(num_samples, vector_dim, p=2., low=low, high=high)
    if normalize:
        y = y**2 / (vector_dim * (high - low)**2)
    else:
        y = y**2
    return X, y

def unnormalize_net_output(y_pred:torch.Tensor, vector_dim:int, min_val:float, max_val:float) -> torch.Tensor:
    """
    Restore the original scale of the L2 squared distance from the normalized output of L2Net.
    
    :param y_pred: Normalized L2 squared distance predicted by L2Net
    :param vector_dim: The dimension of the vectors
    :param min_val: The minimum value of the input data
    :param max_val: The maximum value of the input data
    :return: Unnormalized L2 squared distance
    """
    return torch.mul(y_pred, vector_dim * (max_val - min_val)**2)

def encode_temporal_np(X_data:np.ndarray,
                       time_steps:int,
                       time_pad:int,
                       min_val:float=0.0,
                       max_val:float=1.0)\
        -> np.ndarray:
    """
    입력 데이터를 Latency Coding(TTFS)으로 변환합니다.
    강한 입력(절댓값) -> 빠른 스파이크, 약한 입력 -> 늦은 스파이크.
    음수와 양수를 별도 채널로 분리합니다.

    Args:
        X_data (np.ndarray): 입력 데이터 (*,)
        time_steps (int): 총 시뮬레이션 시간 단계.
        time_pad (int): 시간 패딩 크기.

    Returns:
        np.ndarray: 인코딩된 스파이크 데이터 (time_steps, *X_data.shape)
    """
    
    X_data = X_data - min_val
    X_data = X_data / (max_val - min_val)
    
    X_norm = ((max_val - X_data) * (time_steps-1)) / max_val
    X_pos = np.floor(X_norm).astype(np.int32)
    spikes_out = np.zeros((time_steps + time_pad, *X_data.shape), dtype=np.float32)

    for indices in product(*[range(dim) for dim in X_data.shape]):
        spikes_out[X_pos[*indices], *indices] = 1.0
    
    return spikes_out

def encode_temporal_th(X_data:torch.Tensor, time_steps:int, time_pad:int, min_val:float=0.0, max_val:float=1.0)\
        -> torch.Tensor:
    """
    입력 데이터를 Latency Coding(TTFS)으로 변환합니다.
    강한 입력(절댓값) -> 빠른 스파이크, 약한 입력 -> 늦은 스파이크.
    음수와 양수를 별도 채널로 분리합니다.

    Args:
        X_data (np.ndarray): 입력 데이터 (*,)
        time_steps (int): 총 시뮬레이션 시간 단계.
        time_pad (int): 시간 패딩 크기.

    Returns:
        torch.Tensor: 인코딩된 스파이크 데이터 (time_steps, *X_data.shape)
    """
    
    X_data = X_data - min_val
    X_data = X_data / (max_val - min_val)
    
    X_norm = ((max_val - X_data) * (time_steps-1)) / max_val
    X_pos = torch.floor(X_norm).long()
    
    spikes_out = torch.scatter(
        torch.zeros((time_steps + time_pad, *X_data.shape), dtype=X_data.dtype, device=X_data.device),
        dim=0,
        index=X_pos.unsqueeze(0),
        value=1.)
    
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
    SX_data = encode_temporal_np(X_data, TIME_STEPS)

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