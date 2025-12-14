from itertools import product
from typing import Sequence, overload
import torch
from jaxtyping import Int64, Float
from spikingjelly.activation_based.layer import SynapseFilter
from spikingjelly.activation_based.surrogate import ATan
from spikingjelly.activation_based.neuron import IFNode
from spikingjelly.activation_based.base import MemoryModule
from spikingjelly.activation_based.functional import reset_net
from torch import Tensor
from .layer import LIF_Filter

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

def _shift_vec(input: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """
    Apply synaptic delay to the one-hot coded input tensor.
    
    :param vec: The input tensor to apply delay to. shape: (T, N, C, D_out, D_in)
    :type vec: torch.Tensor
    :param delay: The delay tensor to apply. shape: (N, C, D_out, D_in)
    :type delay: torch.Tensor
    :return: The output tensor after applying synaptic delay. shape: (T, N, C, D_out, D_in)
    :rtype: Tensor
    """
    assert delay.dtype == torch.long, f"Delay tensor must be of type torch.long but got {delay.dtype}"
    T, N, C, D_out, D_in = input.shape
    
    # Apply delay by shifting the time dimension, using torch.gather
    mat = torch.arange(T, device=input.device).view(T,1,1,1,1).repeat(1,N,C,D_out,D_in)
    output = torch.gather(input, 0, (mat - delay[None,...]) % T)
    # Equivalently, we can use roll (very slow):
    # for d_out in range(D_out):
    #     for d_in in range(D_in):
    #         output[..., d_out, d_in] = torch.roll(output[..., d_out, d_in], shifts=rounded_delay[0,0,d_out,d_in].item(), dims=0)
    
    return output

class WrapperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params, forward, backward):
        ctx.backward = backward
        pack, output = forward(input.permute(*range(1, len(input.shape)),0))  # T, N, C, D_in -> N, C, D_in, T
        ctx.save_for_backward(*pack)
        return output.permute(-1,*range(len(input.shape)-1))  # N, C, D_out, T -> T, N, C, D_out

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_input, grad_weight = backward(grad_output.permute(*range(1, len(grad_output.shape)),0), *pack)
        return grad_input.permute(-1,*range(len(grad_output.shape)-1)), grad_weight, None, None

class EventPropLinear(torch.nn.Module):
    def __init__(self, input_dim:int,
                 output_dim:int,
                 T:int,
                 dt:int = 1,
                 tau_m:float = 10.,
                 tau_s:float = 1.,
                 mu:float = 0.1) -> None:
        """
        EventProp Linear Layer Implementation with Spiking Neuron Dynamics.
        https://github.com/lolemacs/pytorch-eventprop
        
        :param self: 설명
        :param input_dim: 설명
        :param output_dim: 설명
        :param T: 설명
        :param dt: 설명
        :param tau_m: 설명
        :param tau_s: 설명
        :param mu: 설명
        """
        super(EventPropLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        
        self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        torch.nn.init.normal_(self.weight, mu, mu)
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, T={self.T}, dt={self.dt}, tau_m={self.tau_m}, tau_s={self.tau_s}'
    
    def forward(self, input): # type: ignore
        out = WrapperFunction.apply(input, self.weight, self.manual_forward, self.manual_backward)
        return out
        
    def manual_forward(self, input):
        steps = int(self.T / self.dt)

        other_dims = input.shape[:-2]
        V = torch.zeros(*other_dims, self.output_dim, steps).cuda()
        I = torch.zeros(*other_dims, self.output_dim, steps).cuda()
        output = torch.zeros(*other_dims, self.output_dim, steps).cuda()

        while True:
            for i in range(1, steps):
                t = i * self.dt
                V[...,i] = (1 - self.dt / self.tau_m) * V[...,i-1] + (self.dt / self.tau_m) * I[...,i-1]
                I[...,i] = (1 - self.dt / self.tau_s) * I[...,i-1] + torch.nn.functional.linear(input[...,i-1].float(), self.weight)
                spikes = (V[...,i] > 1.0).float()
                output[...,i] = spikes
                V[...,i] = (1-spikes) * V[...,i]
            if self.training:
                is_silent = output.sum(dim=-1).flatten(end_dim=-2).min(dim=0)[0] == 0
                self.weight.data[is_silent] = self.weight.data[is_silent] + 1e-1
                # if is_silent.sum() == 0:
                #     break
                break
            else:
                break

        return (input, I, output), output
    
    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)

        other_dims = input.shape[:-2]
        lV = torch.zeros(*other_dims, self.output_dim, steps).cuda()
        lI = torch.zeros(*other_dims, self.output_dim, steps).cuda()
        
        grad_input = torch.zeros(*other_dims, input.shape[-2], steps).cuda()
        grad_weight = torch.zeros(*other_dims, *self.weight.shape).cuda()
        
        for i in range(steps-2, -1, -1):
            t = i * self.dt
            delta = lV[...,i+1] - lI[...,i+1]
            grad_input[...,i] = torch.nn.functional.linear(delta, self.weight.t())
            lV[...,i] = (1 - self.dt / self.tau_m) * lV[...,i+1] + post_spikes[...,i+1] * (lV[...,i+1] + grad_output[...,i+1]) / (I[...,i] - 1 + 1e-10)
            lI[...,i] = lI[...,i+1] + (self.dt / self.tau_s) * (lV[...,i+1] - lI[...,i+1])
            spike_bool = input[...,i].float()
            grad_weight -= (spike_bool.unsqueeze(-2) * lI[...,i].unsqueeze(-1))
        return grad_input, grad_weight

class StochasticDelay(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, input: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """Apply synaptic delay to the one-hot coded input tensor.
        
        :param ctx: Context to save information for backward pass
        :param input: The input tensor to apply delay to. shape: (T, N, C, D_out, 2)
        :type input: torch.Tensor
        :param delay: The `float` delay tensor to apply. Delay must be in range [0, T-1].
            It will be rounded stochastically. shape: (D_out, 2)
        :type delay: torch.nn.Parameter
        :return: The output tensor after synaptic delay. shape: (T, N, C, D_out, 2)
        :rtype: torch.Tensor
        """
        output = input.clone()
        T, N, C, D_out, D_in = output.shape
        # Sample synaptic delays
        
        batch_delay_latent = delay[None,None,...].repeat(N, C, 1, 1) # Shape: (N, C, D_out, 2)
        
        batch_delay_floored = torch.floor(batch_delay_latent)
        p = batch_delay_latent - batch_delay_floored  # The probability that delay is ceiled
        
        batch_delay_rounded = torch.where(torch.bernoulli(p).bool(), batch_delay_floored + 1.0, batch_delay_floored)
        batch_delay_rounded = batch_delay_rounded.to(output.device) # Shape: (N, C, D_out, D_in)
        batch_delay_rounded = batch_delay_rounded.clamp(max= (T - 1) - output.argmax(dim=0)) # Prevent delay overflow
        batch_delay_rounded = batch_delay_rounded.long()
        
        # Apply delay by shifting the time dimension, using torch.gather
        output = _shift_vec(output, batch_delay_rounded)
        
        # For backward pass, use real-valued delay (not rounded)
        ctx.save_for_backward(input.clone(), output.clone(), batch_delay_rounded.clone())
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE): Return gradients estimator (Straight-Through Estimator) for input and delay.
        
        :param ctx: Context containing saved tensors from forward pass
        :param grad_output: Gradient d_loss / d_shifted_vec. shape: (T, N, C, D_out, 2)
        :return: Gradients for input and delay. shape: (T, N, C, D_out, 2), (D_out, 2)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        
        """
        
        #TODO: Gradient of clamped delay?
        input, output, delay, = ctx.saved_tensors
        # T, N, C, D_out, _2 = input.shape
        # T, N, C, D_out, _2 = output.shape
        # D_out, _2 = delay.shape
        
        # 'delay'에 대한 그래디언트만 계산 (vec의 그래디언트는 None)
        grad_input = None
        grad_delay = None
        
        if ctx.needs_input_grad[0]:
            grad_input = _shift_vec(grad_output, -delay)
        
        # 'delay'가 그래디언트를 요구할 때만 (needs_input_grad[1]) 계산
        if ctx.needs_input_grad[1]:
            # score = torch.argmax(output, dim=0)  # shape: (N, C, D_out, 2)
            # score = torch.square(score[...,0] - score[...,1])  # shape: (N, C, D_out)
            # score = torch.exp(-score)  # shape: (N, C, D_out)
            
            # score = torch.softmax(output, dim=0) # shape: (T, N, C, D_out, 2)
            # score = torch.linalg.vecdot(score[...,0], score[...,1], dim=0)  # shape: (N, C, D_out)
            
            # grad_delay = (grad_output * score[None,...,None])
            
            # delay_ceil_prob = delay - torch.floor(delay)  # P(ceil), shape: (D_out, 2)
            # delay_floor_prob = 1.0 - delay_ceil_prob      # P(floor), shape: (D_out, 2)
            # delay_ceil_map = torch.nn.functional.one_hot(torch.ceil(delay).long(), T) # shape: (D_out, 2, T_in)
            # delay_ceil_map = torch.nn.functional.one_hot(delay_ceil_map.transpose(0,-1), T).permute(0, 3, 2, 1) # shape: (T_in, T_out, D_out, 2)
            # delay_floor_map = torch.nn.functional.one_hot(torch.floor(delay).long(), T) # shape: (D_out, 2, T_in)
            # delay_floor_map = torch.nn.functional.one_hot(delay_floor_map.transpose(0,-1), T).permute(0, 3, 2, 1) # shape: (T_in, T_out, D_out, 2)
            # delay_map = delay_ceil_prob[None, None,...] * delay_ceil_map + delay_floor_prob[None, None,...] * delay_floor_map # shape: (T_in, T_out, D_out, 2)
            # delay_map = delay_ceil_map - delay_floor_map # shape: (T_in, T_out, D_out, 2)
            # grad_delay = (delay_map[:,:,None,None,:,:] * input[:,None,:,:,:,:]).sum(dim=0) # shape: (T_out, N, C, D_out, 2)
            # grad_delay = (grad_output * grad_delay).sum(dim=(0,1,2)) / input.shape[0]  # shape: (N, C, D_out, 2)
            
            # for indices in product(*[range(dim) for dim in (N, C, D_out, 2)]):
            #     delay_map[..., *indices]
            # diff = torch.argmax(output, dim=0)  # shape: (N, C, D_out, 2)
            # diff = (diff[...,1] - diff[...,0]) # shape: (N, C, D_out)
            # score = torch.exp(-torch.square(diff) / 2).unsqueeze(-1)  # shape: (N, C, D_out, 1)
            # diff = torch.stack([diff, -diff], dim=-1)  # shape: (N, C, D_out, 2)
            # grad_delay = (grad_output.sum(dim=0) * (score * diff)) # shape: (N, C, D_out, 2)
            
            # grad_delay = (torch.argmax(output, dim=0) - torch.argmax(input, dim=0)).float()  # shape: (N, C, D_out, 2)
            
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
        self.spike_queue = self.spike_queue.to(x.device)
        self.spike_queue[0, ...] = 0  # Drop last spike
        self.spike_queue = self.spike_queue.roll(shifts=-1, dims=0)  # Shift the queue to make space for new spike
        x_delayed = torch.zeros_like(self.spike_queue)
        x_delayed[0,:x.shape[0]] = x 
        x_delayed = StochasticDelay.apply(x_delayed, delay)  # Apply stochastic delay
        self.spike_queue += x_delayed
        return self.spike_queue[0,:x.shape[0]]  # Return the output spike at current time

class TrainableDelay(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int|None=None, init_delay:float=0) -> None:
        """
        Trainable Synaptic Delay Module.
        
        :param self: Self
        :param init_delay: Initial delay in seconds
        """
        super().__init__()
        self.init_delay = init_delay
        self.synapsewise = out_features is not None
        
        if self.synapsewise:
            assert out_features is not None
            self.delay = torch.nn.Parameter(torch.full((in_features, out_features), init_delay), requires_grad=True)
        else:
            self.delay = torch.nn.Parameter(torch.full((in_features, 1), init_delay), requires_grad=True)
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f'init_delay={self.init_delay}'
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = StochasticDelay.apply(torch.sigmoid(input), self.delay)
        if not self.synapsewise:
            output = torch.squeeze(output, -1)
        return output

class JeffressLinear(torch.nn.Module):
    def __init__(self, radius: int, bias = False) -> None:
        """
        Synaptic Delay Convolutional Linear Layer.
        Applies synaptic delay convolution followed by a learnable synapse filter.
        Linear transformation is performed with synapse-wise delays.
        
        :param self: Self
        :param out_features: Number of output features
        :type out_features: int
        :param tau: Time constant for the synapse filter
        :type tau: float
        :param bias: Whether to include a bias term
        """
        super().__init__()
        self.in_features = 2
        self.radius = radius
        self.out_features = 2 * radius + 1
        self.has_bias = bias
        _delay = torch.arange(-radius, radius+1, dtype=torch.float32, requires_grad=True).view(-1,1)
        self._delay = torch.nn.Parameter(_delay, requires_grad=False) # For symmetry
        self._weight = torch.nn.Parameter(torch.tensor(6.53543197272069), requires_grad=False)
        if bias:
            self.log_bias = torch.nn.Parameter(torch.tensor(0.))

        self.filter = LIF_Filter(step_mode="m")
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.radius}'
    
    @property
    def delay(self):
        return torch.cat([self._delay, -self._delay], dim=1).relu()

    @property
    def weight(self):
        return self._weight

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
        
        # Duplicate last dimension for D_out and D_in, to apply synapse-wise delay.
        # a out spikes from each input neuron map to d_out output neurons: total d_out output delays.
        output = input.unsqueeze(-2).repeat([1] * (len(input.shape) - 1) + [self.out_features, 1]) # ..., 2 -> ..., d_out, 2
        
        # Apply synapse-wise delay
        output = StochasticDelay.apply(output, self.delay) # output shape: (T, N, C, D_out, 2)
        assert isinstance(output, torch.Tensor)
        
        output = self.filter(output)  # Apply LIF_Filter, shape: (T, N, C, D_out, 2)
        
        # Apply synapse filter (postsynaptic current kernel) and weight
        output = torch.mul(output, self.weight) # For only one weight
        
        # Sum over input dimension to get output current.
        output = torch.sum(output, dim=-1)
        
        if self.has_bias:
            output += self.log_bias.exp()

        return output

class SpikeAmplifier(torch.nn.Module):
    def __init__(self,
                 num_features:int,
                 backend:str="torch") -> None:
        """
        Jeffress Model for Temporal Correlation Detection.
        
        :param self: Self
        :param in_features: Number of input features (J)
        :param out_features: Number of output features (J)
        :param step_mode: Step mode for spiking neurons
        :param backend: Backend for spiking neurons
        """
        super().__init__()
        self.num_features = num_features
        
        self.lateral_weight = torch.nn.Parameter(torch.full((num_features,), 10.), requires_grad=False) # Shape: (J,)
        self.single_step_delay = None
        self.neuron = IFNode(v_reset=0., surrogate_function=ATan(), backend=backend, step_mode="s", store_v_seq=True)
        self.i_seq = []
        self.v_seq = []
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward의 Docstring
        
        :param self: Self
        :param input: Input tensor, shape: (T, N, C, J)
        :type input: torch.Tensor
        :return: Output tensor after Jeffress processing
        :rtype: Tensor
        """
        
        if self.single_step_delay is None:
            self.single_step_delay = SingleStepDelay(queue_shape=input.shape[:3]+(1,self.num_features), backend=self.neuron.backend)
        self.single_step_delay.reset()
        self.i_seq = input.unbind(dim=0)  # Time series of (N, C, J)
        self.v_seq = []
        
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
            self.v_seq.append(self.neuron.v.clone())
        return y_seq[1:] # T, N, C, J