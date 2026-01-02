from itertools import product
from collections.abc import Sequence
import torch
from jaxtyping import Int64, Float
from spikingjelly.activation_based.layer import SynapseFilter
from spikingjelly.activation_based.surrogate import ATan
from spikingjelly.activation_based.neuron import IFNode
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import reset_net
from torch import Tensor
from torch.profiler import record_function
from .layer import LIF_Filter
from .theory import get_weight

class StochasticRound(torch.autograd.Function):
    """
    확률적 라운딩 (Stochastic Rounding)을 위한 Straight-Through Estimator (STE)
    
    - Forward Pass:  입력 x를 floor(x) 또는 ceil(x)로 확률적으로 붕괴시킴.
                     P(ceil(x)) = x - floor(x)
                     P(floor(x)) = 1 - (x - floor(x))
    - Backward Pass: 그래디언트를 1로 그대로 통과시킴 (Identity).
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass: 확률적 라운딩 수행
        """
        # 1. floor 값과 ceil 확률(p) 계산
        #    (예: x=2.7 -> x_floor=2.0, p=0.7)
        x_floor = torch.floor(x)
        p = x - x_floor  # ceil(x)가 될 확률
        
        # 2. 확률적 붕괴
        # r < p (이 이벤트는 p의 확률로 발생) 이면 ceil(x) = x_floor + 1
        # 아니면 (1-p의 확률로 발생) floor(x)
        rounded_x = torch.where(torch.bernoulli(p).bool(), x_floor + 1.0, x_floor)
        
        return rounded_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE):
        그래디언트를 그대로 통과시킴
        """
        # (d_loss / d_rounded_x) * (d_rounded_x / d_x)
        # STE는 d_rounded_x / d_x 를 1로 가정함
        return grad_output

class TimePadding(torch.nn.Module):
    def __init__(self, steps:int) -> None:
        """
        Pad the time dimension with zeros at the end.
        ex) (0,1,1) -> (0,1,1,0,0) for steps=2.
        
        :param self: 설명
        :param steps: 설명
        :type steps: int
        """
        super().__init__()
        self.steps = steps
        
    def extra_repr(self) -> str:
        return super().extra_repr() + f'steps={self.steps}'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat([input, torch.zeros(self.steps, *input.shape[1:], device=input.device)], dim=0)

class TimeCrop(torch.nn.Module):
    def __init__(self, steps:int) -> None:
        """
        Crop the time dimension by removing steps from the beginning.
        ex) (0,1,1,0,0) -> (1,0,0) for steps=2.
        
        :param self: 설명
        :param steps: 설명
        :type steps: int
        """
        super().__init__()
        self.steps = steps
        
    def extra_repr(self) -> str:
        return super().extra_repr() + f'steps={self.steps}'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[self.steps:,...]

class Squeeze(torch.nn.Module):
    def __init__(self, dim:int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.squeeze(self.dim)
    
class Unsqueeze(torch.nn.Module):
    def __init__(self, dim:int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.unsqueeze(self.dim)    
    
class FirstSpikeTime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        forward의 Docstring
        
        :param ctx: Context to save information for backward pass
        :param input: Tensor of shape (T, N, C)
        
        :return: Tensor of shape (N, C) representing first spike times
        """
        idx = torch.arange(input.shape[0], 0, -1).unsqueeze(-1).unsqueeze(-1).float().cuda()
        first_spike_times = torch.argmax(idx*input, dim=0).float()
        ctx.save_for_backward(input, first_spike_times.clone())
        first_spike_times[first_spike_times==0] = input.shape[0]-1
        return first_spike_times
    
    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        # grad_output: d_loss / d_first_spike_times, shape: (N, C)
        input, first_spike_times = ctx.saved_tensors
        k = torch.nn.functional.one_hot(first_spike_times.long(), input.shape[0]).float() # shape: (N, C, T)
        grad_input = k.permute(-1, *range(len(k.shape)-1)) * grad_output.unsqueeze(0)  # shape: (T, N, C) * (1, N, C) -> (T, N, C)
        return grad_input

class TransposeLayer(torch.nn.Module):
    dims: tuple[int, int]

    def __init__(self, dims: tuple[int, int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(*self.dims)

@torch.compile
def roll_spike(input: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """
    Apply synaptic delay to the one-hot coded input tensor.
    
    :param vec: The input tensor to apply delay to. shape: (T, ..., D_out, D_in)
    :type vec: torch.Tensor
    :param delay: The delay tensor to apply. shape: (D_out, D_in)
    :type delay: torch.Tensor
    :return: The output tensor after applying synaptic delay. shape: (T, ..., D_out, D_in)
    :rtype: Tensor
    """
    assert delay.dtype == torch.long, f"Delay tensor must be of type torch.long but got {delay.dtype}"
    with record_function("shift_vec"):
        T = input.shape[0]
        
        # Apply delay by shifting the time dimension, using torch.gather
        mat = torch.arange(T, device=input.device)[:,*([None] * (input.ndim-1))].expand(input.shape)  # Shape: (T, ..., D_out, D_in)
        output = torch.gather(input, 0, (mat - delay[*([None]*(input.ndim-2)),...]) % T)
        # Equivalently, we can use roll (very slow):
        # for d_out in range(D_out):
        #     for d_in in range(D_in):
        #         output[..., d_out, d_in] = torch.roll(output[..., d_out, d_in], shifts=delay[0,0,d_out,d_in].item(), dims=0)
    
    return output

class SpikeDelay(torch.autograd.Function):    
    @staticmethod
    @torch.compile
    def forward(ctx, input: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """Apply synaptic delay to the one-hot coded input tensor.
        
        :param ctx: Context to save information for backward pass
        :param input: The input tensor to apply delay to. shape: (T, ..., D_out, D_in)
        :type input: torch.Tensor
        :param delay: The `long` delay tensor to apply. Delay must be in range [0, T-1]. shape: (D_out, D_in)
        :type delay: torch.nn.Parameter
        :return: The output tensor after synaptic delay. shape: (T, ..., D_out, D_in)
        :rtype: torch.Tensor
        """
        assert delay.dtype == torch.long, f"Delay tensor must be of type torch.long but got {delay.dtype}"
        with record_function("SpikeDelay_forward"):
            T = input.shape[0]
            # Sample synaptic delays
            
            # batch_delay = delay[*([None]*(input.ndim-3)),...].expand(*input.shape[1:-2],-1,-1) # Shape: (..., D_out, D_in)
            indices = torch.arange(T, device=input.device)[:, *[None]*(input.ndim-1)]  # Shape: (T, ..., 1, 1)
            mask = indices >= T - delay[*([None]*(input.ndim-2)),...]  # Shape: (T, ..., D_out, D_in)
            input = input.masked_fill(mask, 0.)
            # batch_delay = batch_delay.clamp(max = (T - 1) - input.argmax(dim=0)) # Prevent delay overflow
            
            # For backward pass
            ctx.save_for_backward(delay.clone())
            
            # Apply delay by shifting the time dimension, using torch.gather
        return roll_spike(input, delay)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE): Return gradients estimator (Straight-Through Estimator) for input and delay.
        
        :param ctx: Context containing saved tensors from forward pass
        :param grad_output: Gradient d_loss / d_shifted_vec. shape: (T, N, C, D_out, 2)
        :return: Gradients for input and delay. shape: (T, N, C, D_out, 2), (D_out, 2)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        
        """
        delay, = ctx.saved_tensors
        
        # 'delay'에 대한 그래디언트만 계산 (vec의 그래디언트는 None)
        grad_input = None
        grad_delay = None
        
        if ctx.needs_input_grad[0]:
            grad_input = roll_spike(grad_output, -delay)
        
        # 'delay'가 그래디언트를 요구할 때만 (needs_input_grad[1]) 계산
        if ctx.needs_input_grad[1]:
            grad_delay = (grad_output).sum(dim=(0,1,2))
        
        # vec의 그래디언트(None), d의 그래디언트 순서로 반환
        return grad_input, grad_delay


class SingleStepDelay(MemoryModule):
    def __init__(self, queue_shape:Sequence[int], backend="torch") -> None:
        super().__init__()
        self.register_memory("spike_queue", torch.zeros(queue_shape, dtype=torch.float32)) # Shape: (T, N, C, D_out, D_in)
        self.backend = backend
        
    def single_step_forward(self, x: Tensor, delay: Tensor) -> Tensor:
        """
        Single step forward pass for the delay module.
        
        :param x: Input tensor of shape (N, C, D_out, D_in)
        :type x: torch.Tensor
        :param delay: Delay tensor of shape (D_out, D_in)
        :type delay: torch.Tensor (float in [0, 1])
        :return: Output tensor after applying delay
        """
        self.spike_queue[0, ...] = 0  # Drop last spike
        self.spike_queue = self.spike_queue.roll(shifts=-1, dims=0)  # Shift the queue to make space for new spike
        x_delayed = torch.zeros_like(self.spike_queue)
        x_delayed[0,:x.shape[0]] = x 
        x_delayed = SpikeDelay.apply(x_delayed, delay)  # Apply stochastic delay
        self.spike_queue += x_delayed
        return self.spike_queue[0,:x.shape[0]]  # Return the output spike at current time

class JeffressFilter(torch.nn.Module):
    def __init__(self, radius: int, compression: int = 1, accelerated: bool = True) -> None:
        """
        Synaptic Delay Convolutional Linear Layer.
        Applies synaptic delay convolution followed by a learnable synapse filter.
        Linear transformation is performed with synapse-wise delays.
        
        :param self: Self
        :param radius: Radius of cross correlation window
        :type radius: int
        :param compression: compression multiplier in correlation window.
        """
        super().__init__()
        self.in_features = 2
        self.radius = radius
        self.out_features = 2 * ((radius - 1) // compression + 1)
        _delay = torch.cat([torch.arange(-radius, 0, compression, dtype=torch.long),
                            torch.arange(radius, 0, -compression, dtype=torch.long).flip(0)],
                           dim=0).view(-1, 1)
        self.delay = torch.nn.Parameter(torch.cat([_delay, -_delay], dim=1).relu(), requires_grad=False) # (D_out, D_in(=2))
        self.weight = torch.nn.Parameter(torch.tensor(get_weight(1., 10., compression)), requires_grad=False)

        self.filter = LIF_Filter(step_mode="m")
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    def forward(self, input: torch.Tensor, reset: bool=True) -> torch.Tensor:
        """
        forward의 Docstring
        
        :param self: Self
        :param input: Input tensor, shape: (T, N, C, 2)
        :type input: torch.Tensor
        :param reset: Whether to reset the filter state before forward pass
        :type reset: bool
        :return: Output tensor after synaptic delay convolution and filtering
        :rtype: Tensor
        """
        if reset:
            self.filter.reset()
        with record_function("JeffressLinear_forward"):
            output = self.filter(input)  # Apply LIF_Filter, shape: (T, N, C, 2)
            
            # Duplicate last dimension for D_out and D_in, to apply synapse-wise delay.
            # a out spikes from each input neuron map to d_out output neurons: total d_out output delays.
            output = output[...,None,:].expand([-1] * (len(input.shape) - 1) + [self.out_features, -1]) # ..., 2 -> ..., d_out, 2
            
            # Apply synapse-wise delay
            output = SpikeDelay.apply(output, self.delay) # output shape: (T, N, C, D_out, 2)
            assert isinstance(output, torch.Tensor)
            
            # Apply synapse filter (postsynaptic current kernel) and weight
            # Sum over input dimension to get output current.
            output = torch.sum(output, dim=-1) # Shape: (T, N, C, D_out)
            output = torch.mul(output, self.weight) # Shape: (T, N, C, D_out)

        return output


class SpikeAmplifierNetwork(torch.nn.Module):
    def __init__(self,
                 num_features:int,
                 backend:str="torch") -> None:
        """
        Amplify the first spike to all subsequent time steps.
        
        :param self: Self
        :param num_features: Number of input and output features (J)
        :param step_mode: Step mode for spiking neurons
        :param backend: Backend for spiking neurons
        """
        super().__init__()
        self.num_features = num_features
        
        self.lateral_weight = torch.nn.Parameter(torch.full((num_features,), 10.), requires_grad=False) # Shape: (J,)
        self.single_step_delay = None
        self.neuron = IFNode(v_reset=0., surrogate_function=ATan(), backend=backend, step_mode="s", store_v_seq=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        A SNN to change all spikes after the first spike to 1.
        
        :param self: Self
        :param input: Input tensor, shape: (T, N, C, J)
        :type input: torch.Tensor
        :return: Output tensor after amplification
        :rtype: Tensor
        """
        
        if self.single_step_delay is None:
            self.single_step_delay = SingleStepDelay(queue_shape=input.shape[:-1]+(1,self.num_features), backend=self.neuron.backend).to(input.device)
        self.single_step_delay.reset()
        
        reset_net(self.neuron)
        
        T = input.shape[0]
        y_seq = torch.zeros(T+1, *input.shape[1:-1], self.num_features, device=input.device) # T+1, N, C, J
        h = torch.zeros(*input.shape[1:-1], self.num_features, device=input.device)  # N, C, J
        for t in range(T):
            h = h - (1. - y_seq[t]) * h + self.lateral_weight * y_seq[t]  # N, C, J
            x = input[t,...] + h  # N, C, J
            y = self.neuron(x) # N, C, J
            # y = self.single_step_delay(y.unsqueeze(-2), torch.nn.functional.softplus(self.recurrent_delay))  # N, C, 1, J
            # y = y.squeeze(-2) # N, C, J
            y_seq[t+1] = y # List of (N, C, J)
        return y_seq[1:] # T, N, C, J
    
class SpikeAmplifier(torch.nn.Module):
    def __init__(self,
                 num_features:int,
                 accelerated:bool=True,
                 backend:str="torch") -> None:
        """
        Amplify the first spike to all subsequent time steps.
        
        :param self: Self
        :param num_features: Number of input and output features (J)
        :param step_mode: Step mode for spiking neurons
        :param backend: Backend for spiking neurons
        """
        super().__init__()
        self.num_features = num_features
        self.accelerated = accelerated
        
        self.model = SpikeAmplifierNetwork(num_features=num_features, backend=backend)
    
    @torch.compile
    def forward_accelerated(self, input: torch.Tensor) -> torch.Tensor:
        """
        Change all spikes after the first spike to 1.
        
        :param self: Self
        :param input: Input tensor, shape: (T, ...)
        :type input: torch.Tensor
        :return: Output tensor after amplification
        :rtype: Tensor
        """
        assert self.accelerated, "Model is not in accelerated mode."
        
        output = torch.cumsum(input, dim=0)
        output = torch.ge(output, 1.0)
        output = torch.cummax(output, dim=0)[0].float()
        
        return output
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Change all spikes after the first spike to 1.
        
        :param self: Self
        :param input: Input tensor, shape: (T, ...)
        :type input: torch.Tensor
        :return: Output tensor after amplification
        :rtype: Tensor
        """
        
        if self.accelerated:
            return self.forward_accelerated(input)
        else:
            return self.model(input)
        