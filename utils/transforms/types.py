from dataclasses import dataclass, field
from typing import Protocol, Callable, NamedTuple
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