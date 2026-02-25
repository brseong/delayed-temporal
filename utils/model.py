from turtle import forward
import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, IFNode, BaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import ATan
from spikingjelly.activation_based.monitor import OutputMonitor
from utils.theory import tau2gamma

from .module import SpikeAmplifier, JeffressFilter, TimePadding, TimeCrop
from jaxtyping import Float

class AbstractL2Net(torch.nn.Module):
    def __init__(self,
                 time_steps:int,
                 jeffress_radius:int,
                 jeffress_compression:int,
                 input_window:tuple[float, float] = (0.0, 1.0),
                 output_window:tuple[float, float] = (1.0, 2.0)):
        """
        Abstract Cross-correlation network initialization.
        
        :param self: Self
        :param vector_dim: Vector dimension to be processed
        :param jeffress_radius: Number of Jeffress neurons 
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param neuron: Neuron model to be used (default: LIFNode)
        :param surrogate: Surrogate gradient function to be used (default: ATan)
        """
        super(AbstractL2Net, self).__init__()
        self.time_steps = time_steps
        self.jeffress_radius = jeffress_radius
        self.jeffress_compression = jeffress_compression
        self.input_window = input_window
        self.output_window = output_window
        
        self.j_out_shape = 2 * ((jeffress_radius - 1) // jeffress_compression + 1)
        self.log_w = torch.nn.Parameter(torch.ones(self.j_out_shape), requires_grad=True)
        self.tau_s = torch.nn.Parameter(torch.tensor(10.), requires_grad=True)
        # if out_neuron is None:
        #     self.out_neuron = LIFNode(tau=20., surrogate_function=ATan(), backend=backend, step_mode=step_mode)
        # else:
        #     self.out_neuron = out_neuron
        
        # self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
        self.spike_count = 0
    
    def forward(self, x:Float[torch.Tensor, "N 2 C"]) -> Float[torch.Tensor, "N 1"]:
        """
        Compute the correlation between two input tensors.
        
        :param self: Self
        :param x: Input tensor of shape (N, 2, C). Input values are expected to be in the range [0, 1], representing normalized spike times.
        :type x: torch.Tensor
        :return: Output tensor of shape (N, 1)
        :rtype: torch.Tensor
        """
        
        # Simulate input encoding process
        # Input time window is [0, 1], where 0 means the earliest spike and 1 means the latest spike.
        # `time_steps` is the precision of Jeffress layer.
        x = (self.input_window[1]-x) * (self.time_steps - 1) # *N,2,C
        x = x.floor() #  N,2,C
        
        # Get the index of activated neurons in Jeffress layer
        diff = x[...,0,:] - x[...,1,:]  # N,C -> N,C
        diff = diff.floor() 
        
        # Get the spike times of the Jeffress layer neurons
        jeffress_spk_out_time = x.max(dim=-2).values # N,C
        # Positive weight constraint: Sum squared difference is always non-negative
        log_current = self.log_w[diff.long()]  # N,C -> N,C
        # Assume that the size of the output spike time window is 1, and the spike time is determined by the membrane potential.
        # We can use an exponential function to simulate the relationship between the spike time and the membrane potential.
        current = torch.exp(log_current - (self.output_window[1] - jeffress_spk_out_time) * (1/self.tau_s))  # N,C -> N,C
        return current.sum(dim=-1, keepdim=True)  # N,C -> N,1

class L2Net(torch.nn.Module):
    def __init__(self,
                 time_steps:int,
                 vector_dim:int,
                 jeffress_radius:int,
                 jeffress_compression:int,
                 temporal_min:float = 0.0,
                 temporal_max:float = 1.0,
                 step_mode:str = "m",
                 backend:str = "torch",
                 out_neuron:BaseNode|None = None,
                 tau_m:float = 20.,
                 tau_s:float = 2.,
                 accelerated:bool = False):
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
        self.time_steps = time_steps
        self.vector_dim = vector_dim
        self.jeffress_radius = jeffress_radius
        self.jeffress_compression = jeffress_compression
        self.temporal_min = temporal_min
        self.temporal_max = temporal_max
        self.step_mode = step_mode
        self.backend = backend
        self.tau_m = tau_m
        self.tau_s = tau_s
        
        def _get_surrogate():
            return ATan()
        
        j_out_shape = 2 * ((jeffress_radius - 1) // jeffress_compression + 1)
        
        _linear = Linear(j_out_shape, 1, bias=False, step_mode=step_mode)
        with torch.no_grad():
            _linear.weight[:] = torch.linspace(
                temporal_min-temporal_max,
                temporal_max-temporal_min,
                j_out_shape).view(1, -1).square()
        
        self.jeffress_model = torch.nn.Sequential(
            TimePadding(time_steps),  # T,N,C,2 -> T+Tp,N,C,2
            JeffressFilter(time_steps,
                           jeffress_radius,
                           compression=jeffress_compression,
                           tau_s=tau_s,
                           tau_m=tau_m), # T+Tp,N,C,2 -> T+Tp,N,C,J,2
            SpikeAmplifier(j_out_shape, tau=tau_s, backend=backend, accelerated=accelerated),  # T+Tp,N,C,J,2 -> T+Tp,N,C,J
            TimeCrop(time_steps),  # T+Tp,N,C,J -> T,N,C,J
            
            SynapseFilter(tau=tau2gamma(tau_s), step_mode=step_mode, learnable=True),
            _linear,  # T,N,C,J -> T,N,C,1
        )
        
        if out_neuron is None:
            self.out_neuron = LIFNode(tau=tau2gamma(tau_m), surrogate_function=_get_surrogate(), backend=backend, step_mode=step_mode)
        else:
            self.out_neuron = out_neuron
        
        # self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
    
    def forward(self,
                x:Float[torch.Tensor, "T N 2 C"] | Float[torch.Tensor, "N C 2"],
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
        
        x = torch.transpose(x, -1, -2)  # (T,)N,C,2
        x = self.jeffress_model(x)  # T,N,C,1
        
        x = x.squeeze(-1).sum(dim=-1, keepdim=True) # T,N,C -> T,N,1
        x = self.out_neuron(x)  # T,N,1 -> N,1
        
        # If not non-spiking neuron, average over time dimension
        if return_mean:
            x = x.mean(dim=0)  # T,N,1 -> N,1
        
        return x
