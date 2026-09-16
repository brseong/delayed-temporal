"""Construct isolated worker environments for local and Slurm evaluation."""
from __future__ import annotations

import os
from pathlib import Path


def worker_environment(base: dict[str, str], scratch: Path, gpu: str) -> dict[str, str]:
    environment = dict(base)
    for key in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for key in ("TMPDIR", "TMP", "TEMP"):
        environment[key] = str(scratch)
    for key, name in (
        ("XDG_CACHE_HOME", "cache"), ("XDG_RUNTIME_DIR", "xdg-runtime"),
        ("HF_HOME", "huggingface"), ("HF_DATASETS_CACHE", "huggingface/datasets"),
        ("HUGGINGFACE_HUB_CACHE", "huggingface/hub"),
        ("HF_MODULES_CACHE", "huggingface/modules"), ("TORCH_HOME", "torch"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"), ("CUPY_CACHE_DIR", "cupy"),
        ("NUMBA_CACHE_DIR", "numba"), ("PIP_CACHE_DIR", "pip"),
        ("MPLCONFIGDIR", "matplotlib"), ("WANDB_DIR", "wandb"),
    ):
        environment[key] = str(scratch / name)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "4"
    environment.update(WANDB_MODE="disabled", WANDB_DISABLED="true", PYTHONDONTWRITEBYTECODE="1")
    return environment

