from dataclasses import dataclass, field
from typing import Protocol
from functools import wraps

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


def check_domain[InT: OpenBounds, OutT: OpenBounds](func: NeuralTransform[InT, OutT]) -> NeuralTransform[InT, OutT]:
    @wraps(func)
    def wrapper(input_value: Tensor, domain: InT, **kwargs) -> tuple[Tensor, OutT]:
        try:
            assert domain.min <= input_value.min() and input_value.max() <= domain.max,\
               "Initial potentials must be within the specified domain."
        except AssertionError as e:
            breakpoint()
        return func(input_value, domain, **kwargs)
    return wrapper