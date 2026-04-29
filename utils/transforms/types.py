from dataclasses import dataclass, field
from typing import Protocol, Callable, NamedTuple, TypeVar, cast
from functools import wraps
import inspect

import torch
from torch import Tensor
from torch.types import Number

@dataclass
class OpenBounds:
    """This class represents an open interval (min, max) where min and max are exclusive bounds for the input values."""
    min: Number # infimum of the domain, exclusive
    max: Number # supremum of the domain, exclusive
    
    @property
    def range(self) -> Number:
        return self.max - self.min

    def clamp(self, value: Tensor) -> Tensor:
        """Clamp the input tensor to be within the valid range defined by the bounds."""
        return value.clamp(self.min, self.max)
    

class PotentialBounds(OpenBounds): pass

class TimeBounds(OpenBounds): pass

OutBoundsT = TypeVar("OutBoundsT", bound=OpenBounds)

class NeuralTransform[InT: OpenBounds, OutT: OpenBounds](Protocol):
    def __call__(self, input_value: Tensor, domain: InT, **kwargs) -> tuple[Tensor, OutT]: ...


class Potential(NamedTuple):
    """막 전위 텐서와 그 선언 도메인의 묶음.

    ViT 내부 SNN 레이어 간에 도메인을 전파하기 위해 사용된다.
    각 레이어가 독립적으로 텐서를 측정하는 대신, 이전 레이어의
    출력 도메인을 그대로 받아 구간 산술로 출력 도메인을 계산한다.
    """
    value: Tensor
    domain: 'PotentialBounds'


def check_domain[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to check if input tensors are within their specified domains."""
    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Identify tensor-domain pairs
        tensors = {k: v for k, v in bound_args.arguments.items() if isinstance(v, torch.Tensor)}
        domains = {k: v for k, v in bound_args.arguments.items() if isinstance(v, OpenBounds)}
        
        for name, tensor in tensors.items():
            domain = None
            if f"domain_{name}" in domains:
                domain = domains[f"domain_{name}"]
            elif name == "input_value" and "domain" in domains:
                domain = domains["domain"]
            elif len(tensors) == 1 and len(domains) == 1:
                domain = list(domains.values())[0]
            
            if domain is not None:
                assert domain.min <= tensor.min() and tensor.max() <= domain.max,\
                    f"Argument '{name}' must be within the specified domain [{domain.min}, {domain.max}]. Got min {tensor.min()} and max {tensor.max()}."
        
        return func(*args, **kwargs)
    return wrapper


def _emit_spike_time_core(
    input_value: Tensor,
    domain: OpenBounds,
    *,
    noise_std: float = 0.0,
    noise_kind: str = "gaussian"
) -> Tensor:
    """Inject optional spike-time noise and project back into the declared domain.

    Args:
        input_value: Spike-time tensor to perturb.
        domain: Declared valid bounds for the spike-time tensor.
        noise_std: Relative noise scale against domain range. 0 disables noise.
        noise_kind: "gaussian" or "uniform".
        training: Noise is applied only when True.

    Returns:
        Domain-clamped spike-time tensor.
    """
    if noise_std <= 0.0:
        return domain.clamp(input_value)

    span = float(domain.range)
    if noise_kind == "gaussian":
        noise = torch.randn_like(input_value) * (noise_std * span)
    # elif noise_kind == "uniform":
    #     noise = (torch.rand_like(input_value) * 2.0 - 1.0) * (noise_std * span)
    else:
        raise ValueError(f"Unsupported noise_kind: {noise_kind}. Use 'gaussian'")

    output = input_value + noise
    return domain.clamp(output)


@check_domain
def emit_spike_time(
    input_value: Tensor,
    domain: OpenBounds,
    *,
    noise_std: float = 0.0,
    noise_kind: str = "gaussian",
) -> Tensor:
    """Checked wrapper for spike-time noise emission.

    This path keeps input-domain validation for direct calls.
    """
    return _emit_spike_time_core(
        input_value,
        domain,
        noise_std=noise_std,
        noise_kind=noise_kind,
    )


@dataclass
class NoiseConfig:
    """Global configuration for spike-time noise injection."""
    std: float = 0.0
    kind: str = "gaussian"
    eval_mode: bool = False

_GLOBAL_NOISE_CONFIG = NoiseConfig()

def set_spike_time_noise(std: float, kind: str = "gaussian", eval_mode: bool = False):
    """Register global spike-time noise parameters."""
    global _GLOBAL_NOISE_CONFIG
    _GLOBAL_NOISE_CONFIG.std = std
    _GLOBAL_NOISE_CONFIG.kind = kind
    _GLOBAL_NOISE_CONFIG.eval_mode = eval_mode

def get_spike_time_noise() -> NoiseConfig:
    """Retrieve global spike-time noise parameters."""
    return _GLOBAL_NOISE_CONFIG

def inject_spike_time_noise[**P, OutT: OpenBounds](func: Callable[P, tuple[Tensor, OutT]]) -> Callable[P, tuple[Tensor, OutT]]:
    """Decorator that injects optional noise into spike-time outputs.

    Noise is controlled either by explicit kwargs or by the global noise configuration.
    If noise_std is not provided in kwargs, the global registration is used.

    Expected kwargs:
      - noise_std: float (optional)
      - noise_kind: str (optional)
      - training: bool (default=True)
      - noise_eval: bool (optional)
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[Tensor, OutT]:
        output, out_domain = func(*args, **kwargs)
        
        global_cfg = get_spike_time_noise()
        
        # Priority: explicit kwarg > global config
        noise_std_raw = kwargs.get("noise_std", global_cfg.std)
        noise_kind_raw = kwargs.get("noise_kind", global_cfg.kind)
        training_raw = kwargs.get("training", True)
        noise_eval_raw = kwargs.get("noise_eval", global_cfg.eval_mode)

        noise_std = float(noise_std_raw) if isinstance(noise_std_raw, (int, float)) else 0.0
        noise_kind = cast(str, noise_kind_raw)
        training = bool(training_raw)
        noise_eval = bool(noise_eval_raw)

        # Call the undecorated core to avoid a second check_domain pass.
        output = _emit_spike_time_core(
            output,
            out_domain,
            noise_std=noise_std,
            noise_kind=noise_kind
        )
        return output, out_domain

    return wrapper