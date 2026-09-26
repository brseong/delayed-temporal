import torch
from typing import Callable
from spikingjelly.activation_based.neuron import SimpleBaseNode
from spikingjelly.activation_based.surrogate import ATan

class StochasticLIFNode(SimpleBaseNode):
    def __init__(self,
                 tau:float,
                 decay_input: bool,
                 v_threshold: float = 1.,
                 v_reset: float | None = 0.,
                 surrogate_function: Callable = ATan(),
                 detach_reset: bool = False,
                 step_mode='s',):
        """
        Stochastic Leaky Integrate-and-Fire (LIF) neuron model with surrogate gradient.
        
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param surrogate: Surrogate gradient function to be used (default: Sigmoid)
        :param beta: Decay factor for membrane potential
        :param threshold: Firing threshold
        :param reset_mechanism: Reset mechanism after spike. Options: "subtract", "zero"
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.tau = tau
        self.decay_input = decay_input

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x