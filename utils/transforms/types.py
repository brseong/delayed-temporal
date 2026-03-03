from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeAlias, TypeVar
from functools import wraps

import torch
from torch import Tensor
from torch.types import Number

@dataclass
class Bounds(NamedTuple):
    min: Number
    max: Number
    # zero: Number
    @property
    def range(self) -> Number:
        return self.max - self.min

# @dataclass
class PotentialBounds(Bounds):pass

# @dataclass
class TimeBounds(Bounds): pass

T = TypeVar('T', bound=Bounds | PotentialBounds | TimeBounds)
NeuralTransform:TypeAlias = Callable[..., tuple[Tensor, T]]

# PotentialDomain = NewType("PotentialDomain", NamedTuple[Number, Number])
# TimeDomain = NewType("TimeDomain", NamedTuple[Number, Number])

def check_domain(func: NeuralTransform) -> NeuralTransform:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) < 2:
            raise ValueError("Expected at least two arguments: initial potentials and domain.")
        
        if not isinstance(args[0], Tensor):
            raise TypeError("First argument must be a torch.Tensor representing initial potentials or spikes.")
        input_value: Tensor = args[0]
        
        if not isinstance(args[1], Bounds):
            raise TypeError("Second argument must be an Interval representing the input domain.")
        domain: Bounds = args[1]
        
        assert domain.min <= input_value.min() and input_value.max() <= domain.max, "Initial potentials must be within the specified domain."
        return func(*args, **kwargs)
    return wrapper