import torch
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode, BaseNode, IFNode, AdaptBaseNode, NonSpikingIFNode, NonSpikingLIFNode
from spikingjelly.activation_based.layer import Linear, SynapseFilter, PrintShapeModule
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.surrogate import LeakyKReLU, MultiArgsSurrogateFunctionBase, SurrogateFunctionBase, ATan, Sigmoid
from spikingjelly.activation_based.monitor import OutputMonitor

from .module import TransposeLayer, SDCLinear
from jaxtyping import Float

class CCN(torch.nn.Module):
    def __init__(self,
                 vector_dim:int,
                 cc_acc:int,
                 feature_dims:list[int],
                 step_mode:str = "m",
                 backend:str = "torch",
                 neuron = LIFNode,
                 surrogate = ATan):
        """
        Cross-correlation network initialization.
        
        :param self: 설명
        :param input_dim: ...
        :param n_acc: 설명
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
            TransposeLayer((2,3)), # T,N,2,D -> T,N,D,2
            SDCLinear(cc_acc, tau=2., bias=True), # T,N,D,2 -> T,N,D,cc_acc
            neuron(tau=2., v_reset=0., surrogate_function=surrogate(), backend=backend, step_mode="m"), # T,N,D,cc_acc -> T,N,D,cc_acc
            
            SynapseFilter(tau=10.0, step_mode="m", learnable=True), # T,N,D,cc_acc -> T,N,D,cc_acc
            Linear(cc_acc, 1, step_mode="m"), # T,N,D,cc_acc -> T,N,D,1
            torch.nn.Flatten(start_dim=2), # T,N,D,1 -> T,N,D
            neuron(tau=2., v_reset=0., surrogate_function=surrogate(), backend=backend, step_mode="m"), # T,N,D -> T,N,D
        )

        feature_dims = [self.cc_acc] + self.feature_dims
        for in_dim, out_dim in zip(feature_dims[:-1], feature_dims[1:]):
            self.model.extend(
                [
                    # SynapseFilter(tau=10.0, step_mode="m", learnable=True), # T,N,D,cc_acc -> T,N,D,cc_acc
                    Linear(in_dim, out_dim),
                    neuron(v_reset=0., surrogate_function=surrogate(), backend=self.backend)
                ]
            )
        
        self.linear = Linear(feature_dims[-1], 1)
        self.out_neuron = NonSpikingIFNode()
        
        # set_step_mode(self, step_mode)
        self.stats = OutputMonitor(self, (neuron, SynapseFilter))
        
        self._loss = None
    
    @property
    def loss(self):
        if self._loss is not None:
            return self._loss
        else:
            raise ValueError("Loss has not been computed yet.")
    
    def forward(self, x:Float[torch.Tensor, "T N 2D"], reset:bool=True, v_seq_pt:list=[]):
        """
        Compute the correlation between two input tensors.
        
        :param self: 설명
        :param x: 설명
        :type x: torch.Tensor
        """
        if reset:
            for layer in self.model:
                if isinstance(layer, (BaseNode, MemoryModule)):
                    layer.reset()
            
        for i, layer in enumerate(self.model):
            x = layer(x)
            # if i == 2:
            #     for c in range(x.shape[-2]):
            #         print(x[:, 0, c, :])
            #     self._loss = x.mean(dim=(0,-1)) - 0.25
            #     self._loss = x.square().mean()
        
        x = self.linear(x)
        x = self.out_neuron(x)
        
        v_seq_pt.append(self.model[2].v_seq)

        return x

