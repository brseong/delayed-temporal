from dataclasses import dataclass, field
from typing import Protocol, Callable, NamedTuple, TypeVar
from functools import wraps
import inspect
import math
from numbers import Real

import torch
from torch import Tensor
from torch.types import Number

@dataclass(frozen=True)
class ClosedBounds:
    """Immutable declaration of a closed interval with inclusive endpoints.

    Bounds are physical and mathematical contracts carried through the operator
    graph. Freezing their endpoints prevents calibration, memoization, or a later
    forward pass from silently widening an interval after it has been declared.
    Derived intervals must always be represented by newly constructed objects.
    """
    min: Number  # Inclusive lower endpoint of the domain.
    max: Number  # Inclusive upper endpoint of the domain.

    def __post_init__(self) -> None:
        """Reject malformed physical intervals at their construction boundary."""
        for name, endpoint in (("min", self.min), ("max", self.max)):
            if isinstance(endpoint, bool) or not isinstance(endpoint, Real):
                raise TypeError(f"ClosedBounds {name} endpoint must be a real scalar")
            if not math.isfinite(endpoint):
                raise ValueError("ClosedBounds endpoints must be finite")
        if self.min > self.max:
            raise ValueError("ClosedBounds endpoints must satisfy min <= max")

    @property
    def range(self) -> Number:
        return self.max - self.min

    def clamp(self, value: Tensor, name: str | None = None) -> Tensor:
        """Clamp the input tensor to the inclusive closed interval."""
        clamped = value.clamp(self.min, self.max)
        if _CLAMP_LOG_ENABLED and _CURRENT_MODULE_NAME is not None:
            _record_clamp_stats(_CURRENT_MODULE_NAME, name, value, clamped, self.min, self.max)
        return clamped


class PotentialBounds(ClosedBounds): pass

class TimeBounds(ClosedBounds): pass


class SpikeSample(NamedTuple):
    """Finite spike-time storage paired with an explicit delivery mask.

    ``time`` always remains inside ``domain`` so downstream tensor operations never
    receive infinities or sentinel values outside the declared temporal interval.
    When an event misses the fixed observation deadline, ``time`` stores
    ``domain.max`` only as a finite carrier and ``fired`` is false. Consumers must
    therefore consult ``fired`` before interpreting the stored timestamp as an
    event that physically arrived.
    """

    time: Tensor  # Delivered time, or the finite deadline carrier for a missed event.
    domain: TimeBounds  # Code interval whose maximum is the observation deadline.
    fired: Tensor  # Boolean tensor distinguishing delivered events from deadline misses.


OutBoundsT = TypeVar("OutBoundsT", bound=ClosedBounds)

_CLAMP_LOG_ENABLED = False
_CURRENT_MODULE_NAME = None
_CLAMP_STATS = {} # (module_name, clamp_name) -> {'underflow': count, 'overflow': count, 'total': count}

def set_clamp_log_enabled(enabled: bool):
    global _CLAMP_LOG_ENABLED
    _CLAMP_LOG_ENABLED = enabled

def set_current_module_name(name: str | None):
    global _CURRENT_MODULE_NAME
    _CURRENT_MODULE_NAME = name

def get_current_module_name() -> str | None:
    return _CURRENT_MODULE_NAME

def get_clamp_stats():
    return _CLAMP_STATS

def clear_clamp_stats():
    global _CLAMP_STATS
    _CLAMP_STATS = {}

def _record_clamp_stats(module_name: str, clamp_name: str | None, original: Tensor, clamped: Tensor, min_val: Number, max_val: Number):
    tag = (module_name, clamp_name or "unnamed")
    if tag not in _CLAMP_STATS:
        _CLAMP_STATS[tag] = {"underflow": 0, "overflow": 0, "total": 0}

    underflow = (original < min_val).sum().item()
    overflow = (original > max_val).sum().item()
    total = original.numel()

    _CLAMP_STATS[tag]["underflow"] += underflow
    _CLAMP_STATS[tag]["overflow"] += overflow
    _CLAMP_STATS[tag]["total"] += total

class NeuralTransform[InT: ClosedBounds, OutT: ClosedBounds](Protocol):
    def __call__(self, input_value: Tensor, domain: InT, **kwargs) -> tuple[Tensor, OutT]: ...


class Potential(NamedTuple):
    """막 전위 텐서와 그 선언 도메인의 묶음.

    ViT 내부 SNN 레이어 간에 도메인을 전파하기 위해 사용된다.
    각 레이어가 독립적으로 텐서를 측정하는 대신, 이전 레이어의
    출력 도메인을 그대로 받아 구간 산술로 출력 도메인을 계산한다.

    노이즈 모델은 utils/transforms/noise.py 에 모두 모여 있다.
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
        domains = {k: v for k, v in bound_args.arguments.items() if isinstance(v, ClosedBounds)}

        for name, tensor in tensors.items():
            domain = None
            if f"domain_{name}" in domains:
                domain = domains[f"domain_{name}"]
            elif name == "input_value" and "domain" in domains:
                domain = domains["domain"]
            elif len(tensors) == 1 and len(domains) == 1:
                domain = list(domains.values())[0]

            if domain is not None:
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                if not bool(
                    (domain.min <= tensor_min) & (tensor_max <= domain.max)
                ):
                    raise ValueError(
                        f"Argument '{name}' must be within the specified domain "
                        f"[{domain.min}, {domain.max}]. Got min {tensor_min} and "
                        f"max {tensor_max}."
                    )

        return func(*args, **kwargs)
    return wrapper
