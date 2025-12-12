import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, BaseNode, IFNode, AdaptBaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter, PrintShapeModule, LinearRecurrentContainer
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import LeakyKReLU, MultiArgsSurrogateFunctionBase, SurrogateFunctionBase, ATan, Sigmoid, Rect
from spikingjelly.activation_based.monitor import OutputMonitor

from .module import SpikeAmplifier, TransposeLayer, JeffressLinear, EventPropLinear, TimePadding, TimeCrop, Squeeze, Unsqueeze
from jaxtyping import Float

class L2Net(torch.nn.Module):
    def __init__(self,
                 time_steps:int,
                 time_padding:int,
                 vector_dim:int,
                 jeffress_radius:int,
                 feature_dims:list[int],
                 step_mode:str = "m",
                 backend:str = "torch",
                 neuron = LIFNode,
                 tau_m:float = 10.0,
                 tau_s:float = 2.0):
        """
        Cross-correlation network initialization.
        
        :param self: Self
        :param vector_dim: Vector dimension to be processed
        :param J: Cross-correlation accuracy
        :param feature_dims: List of feature dimensions for each layer
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param neuron: Neuron model to be used (default: LIFNode)
        :param surrogate: Surrogate gradient function to be used (default: ATan)
        """
        super(L2Net, self).__init__()
        self.time_steps = time_steps
        self.time_padding = time_padding
        self.vector_dim = vector_dim
        self.jeffress_radius = jeffress_radius
        self.feature_dims = feature_dims
        self.step_mode = step_mode
        self.backend = backend
        self.neuron = neuron
        self.tau_m = tau_m
        self.tau_s = tau_s
        
        def _get_surrogate():
            return ATan(alpha=4.0)
        
        self.jeffress_model = torch.nn.Sequential(
            TimePadding(time_padding),  # T,N,C,2 -> T+Tp,N,C,2
            JeffressLinear(jeffress_radius, bias=False), # T+Tp,N,C,2 -> T+Tp,N,C,J
            SpikeAmplifier(2*jeffress_radius+1, 2*jeffress_radius+1, backend=backend),  # T+Tp,N,C,J -> T+Tp,N,C,1
            TimeCrop(time_padding),  # T+Tp,N,C,1 -> T,N,C,1
            
            TimePadding(time_padding),  # T,N,C,J -> T+Tp,N,C,J
            SynapseFilter(tau=2., step_mode=step_mode, learnable=True),
            Linear(jeffress_radius*2+1, 1, bias=False),  # T+Tp,N,C,J -> T+Tp,N,C,1
            LIFNode(tau=10., surrogate_function=_get_surrogate(), backend=backend, step_mode="m"),  # T,N,C,1 -> T,N,C,1
            TimeCrop(time_padding),  # T+Tp,N,C,1 -> T,N,C,1
        )

        self.weighted_filter = torch.nn.Sequential(
            SynapseFilter(tau=tau_s, step_mode=step_mode, learnable=True),
            Linear(1, 1, bias=False),
        )
        
        self.out_neuron = NonSpikingLIFNode(tau=10)
        
        self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
    
        # with torch.no_grad():
        #     for layer in self.modules():
        #         if isinstance(layer, Linear):
        #             layer.weight.div_(5.).abs_()
        #             if layer.bias is not None:
        #                 layer.bias.abs_()
    
    def forward(self, x:Float[torch.Tensor, "T N 2 C"],
                reset:bool=True,
                return_v_seq:list[torch.Tensor]|None=None) -> Float[torch.Tensor, "T N 1"]:
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
            self.stats.clear_recorded_data()
        
        x = torch.transpose(x, 2, 3)  # T,N,C,2
        x = self.jeffress_model(x)  # T,N,C,1
        
        # if return_l2 is not None:
        #     return_l2.append(x.clone())
        
        x = self.weighted_filter(x).squeeze(-1) # T,N,C,1 -> T,N,C
        x = x.sum(dim=2, keepdim=True) # T,N,C -> T,N,1
        x = self.out_neuron(x)  # T,N,1 -> N,1
        
        if return_v_seq is not None:
            return_v_seq.append(self.jeffress_model[1].v_seq.clone().detach())

        return x

