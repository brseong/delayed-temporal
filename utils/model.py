import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, IFNode, BaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import ATan
from spikingjelly.activation_based.monitor import OutputMonitor

from .module import SpikeAmplifier, SpikeAmplifierNetwork, JeffressLinear, TimePadding, TimeCrop
from .datasets import encode_temporal_th
from jaxtyping import Float

class L2Net(torch.nn.Module):
    def __init__(self,
                 time_padding:int,
                 vector_dim:int,
                 jeffress_radius:int,
                 jeffress_compression:int,
                 temporal_min:float = 0.0,
                 temporal_max:float = 1.0,
                 step_mode:str = "m",
                 backend:str = "torch",
                 out_neuron:BaseNode|None = None,
                 gamma_m:float = 20.,
                 gamma_s:float = 2.):
        """
        Cross-correlation network initialization.
        
        :param self: Self
        :param vector_dim: Vector dimension to be processed
        :param jeffress_radius: Number of Jeffress neurons 
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param neuron: Neuron model to be used (default: LIFNode)
        :param surrogate: Surrogate gradient function to be used (default: ATan)
        """
        super(L2Net, self).__init__()
        self.time_padding = time_padding
        self.vector_dim = vector_dim
        self.jeffress_radius = jeffress_radius
        self.jeffress_compression = jeffress_compression
        self.temporal_min = temporal_min
        self.temporal_max = temporal_max
        self.step_mode = step_mode
        self.backend = backend
        self.gamma_m = gamma_m
        self.gamma_s = gamma_s
        
        def _get_surrogate():
            return ATan(alpha=2.0)
        
        j_out_shape = 2 * ((jeffress_radius - 1) // jeffress_compression + 1)
        
        _linear = Linear(j_out_shape, 1, bias=False, step_mode=step_mode)
        with torch.no_grad():
            _linear.weight[:] = torch.linspace(
                temporal_min-temporal_max,
                temporal_max-temporal_min,
                j_out_shape).view(1, -1).square()
        
        self.jeffress_model = torch.nn.Sequential(
            TimePadding(time_padding),  # T,N,C,2 -> T+Tp,N,C,2
            JeffressLinear(jeffress_radius, compression=jeffress_compression), # T+Tp,N,C,2 -> T+Tp,N,C,J,2
            SpikeAmplifier(j_out_shape, backend=backend),  # T+Tp,N,C,J,2 -> T+Tp,N,C,J
            TimeCrop(time_padding),  # T+Tp,N,C,J -> T,N,C,J
            
            SynapseFilter(tau=gamma_s, step_mode=step_mode, learnable=True),
            _linear,  # T,N,C,J -> T,N,C,1
        )
        
        if out_neuron is None:
            self.out_neuron = LIFNode(tau=gamma_m, surrogate_function=_get_surrogate(), backend=backend, step_mode=step_mode)
        else:
            self.out_neuron = out_neuron
        
        # self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
    
    def forward(self, x:Float[torch.Tensor, "T N 2 C"],
                reset:bool=True,
                return_mean:bool=True) -> Float[torch.Tensor, "T N 1"]:
        """
        Compute the correlation between two input tensors.
        
        :param self: Self
        :param x: Input tensor of shape (T, N, 2, C)
        :param reset: Whether to reset the states of neurons before forward pass
        :param v_seq_pt: List to store returned membrane potential sequences
        :return: Output tensor of shape (T, N, 1)
        :type x: torch.Tensor
        :type reset: bool
        :type return_l2: list[torch.Tensor]|None
        :type return_v_seq: list[torch.Tensor]|None
        :rtype: torch.Tensor
        """
        if reset:
            for layer in self.modules():
                if isinstance(layer, (MemoryModule)):
                    layer.reset()
            # self.stats.clear_recorded_data()
        
        x = torch.transpose(x, -1, -2)  # T,N,C,2
        x = self.jeffress_model(x)  # T,N,C,1
        
        x = x.squeeze(-1).sum(dim=-1, keepdim=True) # T,N,C -> T,N,1
        x = self.out_neuron(x)  # T,N,1 -> N,1
        # If not non-spiking neuron, average over time dimension
        if return_mean:
            x = x.mean(dim=0)  # T,N,1 -> N,1
        
        # if return_v_seq is not None:
        #     return_v_seq.append(self.jeffress_model[1].v_seq.clone().detach())

        return x

