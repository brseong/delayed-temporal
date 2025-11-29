import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, BaseNode, IFNode, AdaptBaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter, PrintShapeModule
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import LeakyKReLU, MultiArgsSurrogateFunctionBase, SurrogateFunctionBase, ATan, Sigmoid, Rect
from spikingjelly.activation_based.monitor import OutputMonitor

from .module import TransposeLayer, JeffressLinear, EventPropLinear
from jaxtyping import Float

class L2Net(torch.nn.Module):
    def __init__(self,
                 vector_dim:int,
                 jeffress_dim:int,
                 feature_dims:list[int],
                 step_mode:str = "m",
                 backend:str = "torch",
                 neuron = IFNode,
                 surrogate = ATan,
                 filter_tau:float = 2.0):
        """
        Cross-correlation network initialization.
        
        :param self: Self
        :param vector_dim: Vector dimension to be processed
        :param cc_acc: Cross-correlation accuracy
        :param feature_dims: List of feature dimensions for each layer
        :param step_mode: Step mode for spiking neurons. Options: "s" (single time-step), "m" (multi time-step)
        :param backend: Backend for spiking neurons. Options: "torch", "cupy"
        :param neuron: Neuron model to be used (default: LIFNode)
        :param surrogate: Surrogate gradient function to be used (default: ATan)
        """
        super(L2Net, self).__init__()
        self.vector_dim = vector_dim
        self.jeffress_dim = jeffress_dim
        self.feature_dims = feature_dims
        self.step_mode = step_mode
        self.backend = backend
        self.neuron = neuron
        self.surrogate = surrogate
        self.filter_tau = filter_tau
        
        def _make_synapse_filter():
            return SynapseFilter(tau=filter_tau, step_mode=step_mode, learnable=False)
        def _make_neuron():
            return neuron(v_reset=0., surrogate_function=surrogate(), backend=backend, step_mode=step_mode)
        
        self.jeffress_model = torch.nn.Sequential(
            JeffressLinear(jeffress_dim, tau=2., bias=False), # T,N,C,2 -> T,N,C,cc_acc
            LIFNode(tau=20., v_reset=0., surrogate_function=ATan(), backend=backend, step_mode="m", store_v_seq=True), # T,N,C,cc_acc -> T,N,C,cc_acc
            
            # _make_synapse_filter(), # T,N,C,cc_acc -> T,N,C,cc_acc
        )
        self.jeffress_integrator = None
        self.jeffress_integrator_neuron = _make_neuron()
            
        # Dimension-wise Linear Layer (or can be seen as convolution), to compute the similarity of each layer.
        self.square_model = torch.nn.Sequential(
            _make_synapse_filter(), # T,N,C,1 -> T,N,C,1
            Linear(1, 10, step_mode="m", bias=True), # T,N,C,1 -> T,N,C,10
            _make_neuron(), # T,N,C,10 -> T,N,C,10
            
            _make_synapse_filter(), # T,N,C,10 -> T,N,C,10
            Linear(10, 1, step_mode="m", bias=True), # T,N,C,10 -> T,N,C,1
            _make_neuron(), # T,N,C,1 -> T,N,C,1
        )
        
        self.sum_filter = _make_synapse_filter()  # T,N,C -> T,N,1
        self.sum_neuron = _make_neuron() # T,N,1 -> T,N,1
        self.sqrt_model = torch.nn.Sequential()
        feature_dims = [1] + self.feature_dims
        for in_dim, out_dim in zip(feature_dims[:-1], feature_dims[1:]):
            self.sqrt_model.extend(
                [
                    _make_synapse_filter(), # T,N,C -> T,N,C
                    Linear(in_dim, out_dim, step_mode="m", bias=True), # T,N,C_in -> T,N,C_out
                    _make_neuron()
                ]
            )
        
        self.sqrt_model.extend(
            [
                _make_synapse_filter(), # T,N,C -> T,N,C
                Linear(feature_dims[-1], 1, step_mode="m", bias=True), # T,N,C -> T,N,1
                NonSpikingIFNode()
            ]
        )
        
        self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
    
        # with torch.no_grad():
        #     for layer in self.modules():
        #         if isinstance(layer, Linear):
        #             layer.weight.div_(5.).abs_()
        #             if layer.bias is not None:
        #                 layer.bias.abs_()
    
    # def get_loss(self):
    #     return self.model[1].get_loss()
    
    def forward(self, x:Float[torch.Tensor, "T N 2 C"],
                reset:bool=True,
                return_l2:list[torch.Tensor]|None=None,
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
        x = self.jeffress_model(x)  # T,N,C,cc_acc
        print(x.sum(dim=(0,-1)).mean())
        if self.jeffress_integrator is None:
            _kernel = torch.arange(0, x.shape[-1], dtype=torch.float32, device=x.device) - x.shape[-1]//2
            _kernel = -torch.abs(_kernel) / self.filter_tau
            self.jeffress_integrator = (1/(1-torch.exp(_kernel))).reshape(1, -1)  # 1,cc_acc
        x = torch.nn.functional.linear(x, self.jeffress_integrator)
        x = self.jeffress_integrator_neuron(x)  # T,N,C,1
        
        x = self.square_model(x)  # T,N,C,1 -> T,N,C,1
        x = torch.flatten(x, start_dim=2) # T,N,C,1 -> T,N,C
        
        if return_l2 is not None:
            return_l2.append(x.clone())
        
        x = self.sum_filter(x)  # T,N,C
        x = x.sum(dim=2, keepdim=True)  # T,N,1
        x = self.sum_neuron(x)  # T,N,1
        
        x = self.sqrt_model(x.detach())  # T,N,1
        
        if return_v_seq is not None:
            return_v_seq.append(self.jeffress_model[1].v_seq.clone().detach())

        return x

