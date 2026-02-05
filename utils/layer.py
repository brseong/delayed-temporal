from typing import Literal
from spikingjelly.activation_based.layer import SynapseFilter
from spikingjelly.activation_based.base import MemoryModule
from jaxtyping import Float
from torch.profiler import record_function
import torch, math

from .theory import tau2beta, tau2gamma

class LIF_Filter(MemoryModule):
    def __init__(self,
                 tau_m:float = 10.,
                 tau_s:float = 1.,
                 step_mode:str = "m"):
        """
        Combined LIF neuron and Synapse Filter module.
        
        :param self: Self
        :param tau_m: Time constant for the LIF neuron
        :param tau_s: Time constant for the synapse filter
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param surrogate: Surrogate gradient function to be used (default: Sigmoid)
        """
        super(LIF_Filter, self).__init__()
        
        self.step_mode = step_mode
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.gamma_m = tau2gamma(tau_m)
        self.gamma_s = tau2gamma(tau_s)
        
        self.synapse_filter = SynapseFilter(tau=self.gamma_s, step_mode=step_mode, learnable=False)
        self.mem_filter = SynapseFilter(tau=self.gamma_m, step_mode=step_mode, learnable=False)
        
    def extra_repr(self):
        return super().extra_repr() + f', lif_gamma={self.gamma_m}, filter_gamma={self.gamma_s}, step_mode={self.step_mode}, tau_m={self.tau_m}, tau_s={self.tau_s}'    
    
    def forward(self, x:Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        if self.step_mode == "m":
            self.synapse_filter.reset()
            self.mem_filter.reset()
        else:
            raise NotImplementedError("Only 'm' (multi time-step) step_mode is implemented.")
            
        return self.forward_discrete(x)
    
    # @torch.compile
    def forward_discrete(self, x):
        """
        Simulate Leaky-Integrate and Fire Neuron with alpha-function synapse filter in discrete time.
        
        :param self: Self
        :param x: Input spike tensor of shape (T, ...)
        :return: Output tensor after applying discrete LIF filter
        """
        x = torch.cat([torch.zeros_like(x[0:1]), x], dim=0)  # Pad initial zero for correct difference calculation # Shape: (T+1, ...)
        V_neg = self.synapse_filter(x) # Shape: (T+1, ...)
        V_pos = self.mem_filter(x) # Shape: (T+1, ...)
        V = (V_pos - V_neg) / (tau2beta(self.tau_m) - tau2beta(self.tau_s))  # Use inverse of time constant difference for computation efficiency
        # assert torch.all(V >= 0), "LIF_Filter output has negative values."
        dV = V[1:] - V[:-1] # Shape: (T, ...)
        return dV
    
# class SynapseFilterCustom(MemoryModule):
#     def __init__(self, beta=0.9, learnable=False, step_mode='s'):
#         super().__init__()
#         self.step_mode = step_mode
#         self.learnable = learnable
#         assert beta <= 1 and beta > 0, "Beta should be in (0, 1]."
#         if learnable:
#             init_w = math.log(beta/(1 - beta))
#             self.w = torch.nn.Parameter(torch.as_tensor(init_w))
#         else:
#             self.beta = beta
#         self.register_memory('out_i', 0.)

#     def extra_repr(self):
#         if self.learnable:
#             with torch.no_grad():
#                 beta = self.w.sigmoid()
#         else:
#             beta = self.beta

#         return f'beta={beta}, learnable={self.learnable}, step_mode={self.step_mode}'

#     @staticmethod
#     @torch.jit.script
#     def js_single_step_forward_learnable(x: torch.Tensor, w: torch.Tensor, out_i: torch.Tensor):
#         # inv_tau = w.sigmoid()
#         # out_i = out_i - (1. - x) * out_i * inv_tau + x
#         out_i = w.sigmoid() * out_i + x
#         return out_i

#     @staticmethod
#     @torch.jit.script
#     def js_single_step_forward(x: torch.Tensor, beta: float, out_i: torch.Tensor):
#         # inv_tau = 1. / tau
#         # out_i = out_i - (1. - x) * out_i * inv_tau + x
#         out_i = beta * out_i + x
#         return out_i

#     def single_step_forward(self, x: torch.Tensor):
#         if isinstance(self.out_i, float):
#             out_i_init = self.out_i
#             self.out_i = torch.zeros_like(x.data)
#             if out_i_init != 0.:
#                 torch.fill_(self.out_i, out_i_init)

#         if self.learnable:
#             self.out_i = self.js_single_step_forward_learnable(x, self.w, self.out_i)
#         else:
#             self.out_i = self.js_single_step_forward(x, self.beta, self.out_i)
#         return self.out_i

if __name__ == "__main__":
    # Test LIF_Filter
    from remote_plot import plt
    lif_filter = LIF_Filter(tau_m=10., tau_s=1., step_mode="m")
    T = 100
    x = torch.zeros(T)
    x[10] = 1.0  # Single spike at time step 10
    y = lif_filter(x.view(T, 1))
    plt.plot(y)
    plt.plot(y.cumsum(dim=0))