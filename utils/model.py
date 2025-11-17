import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, BaseNode, IFNode, AdaptBaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter, PrintShapeModule
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import LeakyKReLU, MultiArgsSurrogateFunctionBase, SurrogateFunctionBase, ATan, Sigmoid, Rect
from spikingjelly.activation_based.monitor import OutputMonitor

from .module import TransposeLayer, JeffressLinear
from jaxtyping import Float

class CCN(torch.nn.Module):
    def __init__(self,
                 vector_dim:int,
                 cc_acc:int,
                 feature_dims:list[int],
                 step_mode:str = "m",
                 backend:str = "torch",
                 neuron = IFNode,
                 surrogate = Rect()):
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
        super(CCN, self).__init__()
        self.vector_dim = vector_dim
        self.cc_acc = cc_acc
        self.feature_dims = feature_dims
        self.step_mode = step_mode
        self.backend = backend
        self.neuron = neuron
        self.surrogate = surrogate
        
        self.model = torch.nn.Sequential(
            TransposeLayer((2,3)), # T,N,2,C -> T,N,C,2
            JeffressLinear(cc_acc, tau=2., bias=False), # T,N,C,2 -> T,N,C,cc_acc
            LIFNode(tau=1.5, v_reset=0., surrogate_function=surrogate, backend=backend, step_mode="m", store_v_seq=True), # T,N,C,cc_acc -> T,N,C,cc_acc
            
            SynapseFilter(tau=10.0, step_mode="m", learnable=True), # T,N,C,cc_acc -> T,N,C,cc_acc
            # Dimension-wise Linear Layer, to compute the similarity of each layer.
            Linear(cc_acc, 1, step_mode="m", bias=False), # T,N,C,cc_acc -> T,N,C,1
            torch.nn.Flatten(start_dim=2), # T,N,C,1 -> T,N,C
            neuron(v_reset=0., surrogate_function=surrogate, backend=backend, step_mode="m"), # T,N,C -> T,N,C
        )

        feature_dims = [self.vector_dim] + self.feature_dims
        for in_dim, out_dim in zip(feature_dims[:-1], feature_dims[1:]):
            self.model.extend(
                [
                    SynapseFilter(tau=10.0, step_mode="m", learnable=True), # T,N,C,cc_acc -> T,N,C,cc_acc
                    Linear(in_dim, out_dim, step_mode="m", bias=False), # T,N,C,in_dim -> T,N,C,out_dim
                    neuron(v_reset=0., surrogate_function=surrogate, backend=backend, step_mode="m")
                ]
            )
        
        # Final linear layer to output a single value
        self.linear = Linear(feature_dims[-1], 1, step_mode="m", bias=True)
        # Final non-spiking neuron to accumulate the voltage
        self.out_neuron = NonSpikingIFNode()
        
        self.stats = OutputMonitor(self, (IFNode, LIFNode, ParametricLIFNode, SynapseFilter))
        
    #     self._loss = None
    
    # @property
    # def loss(self):
    #     if self._loss is not None:
    #         return self._loss
    #     else:
    #         raise ValueError("Loss has not been computed yet.")
    
    # def get_loss(self):
    #     return self.model[1].get_loss()
    
    def forward(self, x:Float[torch.Tensor, "T N 2 C"], reset:bool=True, v_seq_pt:list=[]):
        """
        Compute the correlation between two input tensors.
        
        :param self: Self
        :param x: Input tensor of shape (T, N, 2, C)
        :param reset: Whether to reset the states of neurons before forward pass
        :param v_seq_pt: List to store returned membrane potential sequences
        :return: Output tensor of shape (T, N, 1)
        :type x: torch.Tensor
        :type reset: bool
        :type v_seq_pt: list
        :rtype: torch.Tensor
        """
        if reset:
            for layer in self.model:
                if isinstance(layer, (BaseNode, MemoryModule)):
                    layer.reset()
            self.stats.clear_recorded_data()
            
        for i, layer in enumerate(self.model):
            x = layer(x)
        
        x = self.linear(x)
        x = self.out_neuron(x)
        
        v_seq_pt.append(self.model[2].v_seq)

        return x

