from spikingjelly.activation_based.layer import SynapseFilter, Linear
from spikingjelly.activation_based.base import MemoryModule
from jaxtyping import Float
import torch
from math import log

class LIF_Filter(MemoryModule):
    def __init__(self,
                 gamma_m:float = 10.508331944775,
                 gamma_s:float = 1.58197670686933,
                 step_mode:str = "m"):
        """
        Combined LIF neuron and Synapse Filter module.
        
        :param self: Self
        :param lif_tau: Time constant for the LIF neuron
        :param filter_tau: Time constant for the synapse filter
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param surrogate: Surrogate gradient function to be used (default: Sigmoid)
        """
        super(LIF_Filter, self).__init__()
        
        self.step_mode = step_mode
        self.gamma_m = gamma_m
        self.gamma_s = gamma_s
        self.lif_tau = 1/log(gamma_m / (gamma_m - 1))
        self.filter_tau = 1/log(gamma_s / (gamma_s - 1))
        self.synapse_filter = SynapseFilter(tau=gamma_s, step_mode=step_mode, learnable=False)
        self.lif_filter = SynapseFilter(tau=gamma_m, step_mode=step_mode, learnable=False)
        
    def forward(self, x:Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        if self.step_mode == "m":
            self.synapse_filter.reset()
            self.lif_filter.reset()
        else:
            raise NotImplementedError("Only 'm' (multi time-step) step_mode is implemented.")
        
        x = torch.cat([x, torch.zeros_like(x[0:1, ...])], dim=0)  # Pad zero at the end along the last dimension
        V_neg = self.synapse_filter(x)
        V_pos = self.lif_filter(x)
        V = self.filter_tau * (V_pos - V_neg) / (self.lif_tau - self.filter_tau)
        assert torch.any(V >= 0), "LIF_Filter output has negative values."
        dV = V[1:, ...] - V[:-1, ...]
        
        return dV