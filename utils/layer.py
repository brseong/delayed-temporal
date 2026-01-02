from typing import Literal
from spikingjelly.activation_based.layer import SynapseFilter, Linear
from spikingjelly.activation_based.base import MemoryModule
from jaxtyping import Float
import torch
from torch.profiler import record_function
from math import log

class LIF_Filter(MemoryModule):
    def __init__(self,
                 beta_m:float = 10.508331944775,
                 beta_s:float = 1.58197670686933,
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
        self.beta_m = beta_m
        self.beta_s = beta_s
        
        self.tau_mem = 1/log(beta_m / (beta_m - 1))
        self.tau_syn = 1/log(beta_s / (beta_s - 1))
        self.inv_tau_diff = 1 / (self.tau_mem - self.tau_syn)
        self.synapse_filter = SynapseFilter(tau=beta_s, step_mode=step_mode, learnable=False)
        self.lif_filter = SynapseFilter(tau=beta_m, step_mode=step_mode, learnable=False)
        
        self.dV_kernel:torch.Tensor|None = None
    
    def extra_repr(self):
        return super().extra_repr() + f', lif_tau={self.tau_mem}, filter_tau={self.tau_syn}, step_mode={self.step_mode}'
    
    def generate_dV_kernel(self, T:int, device:torch.device):
        x = torch.eye(T, device=device) # Shape: (T, T)
        x = torch.cat([torch.zeros(1, T, device=device), x], dim=0)  # Shape: (T+1, T)
        V = (self.tau_syn * self.inv_tau_diff) * (self.lif_filter(x) - self.synapse_filter(x)) # Shape: (T+1, T)
        dV_kernel = V[1:] - V[:-1]  # Shape: (T, T)
        # First dimension: time step of a kernel, Second dimension: The index of the kernel (time step when input spike occurs)
        dV_kernel = dV_kernel.transpose(0, 1).contiguous()  # Shape: (T, T)
        # Now dV_kernel[spike_time] gives the dV kernel for the spike_time
        return dV_kernel
    
    def forward(self, x:Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        if self.step_mode == "m":
            self.synapse_filter.reset()
            self.lif_filter.reset()
        else:
            raise NotImplementedError("Only 'm' (multi time-step) step_mode is implemented.")
        
        if self.dV_kernel is None or self.dV_kernel.shape[0] < x.shape[0]:
            self.dV_kernel = self.generate_dV_kernel(x.shape[0], x.device)  # Shape: (T, T+1)
            
        return self.forward_discrete(x)
    
    # @torch.compile
    def forward_discrete(self, x):
        """
        Simulate Leaky-Integrate and Fire Neuron with alpha-function synapse filter in discrete time.
        
        :param self: 설명
        :param x: Input spike tensor of shape (T, N, ...)
        :return: Output tensor after applying discrete LIF filter
        """
        V_neg = self.synapse_filter(x) # Shape: (T, N, ...)
        V_pos = self.lif_filter(x)
        V = (self.tau_syn * self.inv_tau_diff) * (V_pos - V_neg)  # Use inverse of time constant difference for computation efficiency
        assert torch.all(V >= 0), "LIF_Filter output has negative values."
        dV = V[1:] - V[:-1]
        return dV