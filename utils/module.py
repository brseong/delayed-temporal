from itertools import product
import torch
from jaxtyping import Int64, Float
from spikingjelly.activation_based.layer import SynapseFilter
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
    
    :param vec: The input tensor to apply delay to. shape: (T, N, C, D_out, 2)
    :type vec: torch.Tensor
    :param delay: The delay tensor to apply. shape: (N, C, D_out, 2)
    :type delay: torch.Tensor
    :return: The output tensor after applying synaptic delay. shape: (T, N, C, D_out, 2)
    :rtype: Tensor
    """
    T, N, C, D_out, _ = input.shape
    
    # Apply delay by shifting the time dimension, using torch.gather
    mat = torch.arange(T, device=input.device).view(T,1,1,1,1).repeat(1,N,C,D_out,2)
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
        pack, output = forward(input)
        ctx.save_for_backward(*pack)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_input, grad_weight = backward(grad_output, *pack)
        return grad_input, grad_weight, None, None

class EventPropLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s, mu):
        """
        EventProp Linear Layer with Spiking Neuron Dynamics.
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
        
    def forward(self, input): # type: ignore
        return WrapperFunction.apply(input, self.weight, self.manual_forward, self.manual_backward)
        
    def manual_forward(self, input):
        steps = int(self.T / self.dt)
        input = input.permute(1,2,3,0)  # N, C_in, D, T
    
        V = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        I = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        output = torch.zeros(input.shape[0], self.output_dim, steps).cuda()

        while True:
            for i in range(1, steps):
                t = i * self.dt
                V[:,:,i] = (1 - self.dt / self.tau_m) * V[:,:,i-1] + (self.dt / self.tau_m) * I[:,:,i-1]
                I[:,:,i] = (1 - self.dt / self.tau_s) * I[:,:,i-1] + torch.nn.functional.linear(input[:,:,i-1].float(), self.weight)
                spikes = (V[:,:,i] > 1.0).float()
                output[:,:,i] = spikes
                V[:,:,i] = (1-spikes) * V[:,:,i]

            if self.training:
                is_silent = output.sum(2).min(0)[0] == 0
                self.weight.data[is_silent] = self.weight.data[is_silent] + 1e-1
                if is_silent.sum() == 0:
                    break
            else:
                break

        return (input, I, output), output
    
    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)
                
        lV = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        lI = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        
        grad_input = torch.zeros(input.shape[0], input.shape[1], steps).cuda()
        grad_weight = torch.zeros(input.shape[0], *self.weight.shape).cuda()
        
        for i in range(steps-2, -1, -1):
            t = i * self.dt
            delta = lV[:,:,i+1] - lI[:,:,i+1]
            grad_input[:,:,i] = torch.nn.functional.linear(delta, self.weight.t())
            lV[:,:,i] = (1 - self.dt / self.tau_m) * lV[:,:,i+1] + post_spikes[:,:,i+1] * (lV[:,:,i+1] + grad_output[:,:,i+1]) / (I[:,:,i] - 1 + 1e-10)
            lI[:,:,i] = lI[:,:,i+1] + (self.dt / self.tau_s) * (lV[:,:,i+1] - lI[:,:,i+1])
            spike_bool = input[:,:,i].float()
            grad_weight -= (spike_bool.unsqueeze(1) * lI[:,:,i].unsqueeze(2))

        return grad_input, grad_weight

class JeffressDelay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """Apply synaptic delay to the one-hot coded input tensor.
        
        :param ctx: Context to save information for backward pass
        :param input: The input tensor to apply delay to. shape: (T, N, C, D_out, 2)
        :type input: torch.Tensor
        :param delay: The delay tensor to apply. shape: (D_out, 2)
        :type delay: torch.nn.Parameter
        :return: The output tensor after applying synaptic delay. shape: (T, N, C, D_out, 2)
        :rtype: Tensor
        """
        output = input.clone()
        T, N, C, D_out, _ = output.shape
        # Sample synaptic delays
        
        # batch_delay = (T-1) * delay[None,None,...].repeat(N, C, 1, 1) # Shape: (N, C, D_out, 2)
        batch_delay = delay[None,None,...].repeat(N, C, 1, 1) # Shape: (N, C, D_out, 2)
        
        delay_floor = torch.floor(batch_delay)
        p = batch_delay - delay_floor  # ceil(x)가 될 확률
        
        # 확률적 붕괴
        # r < p (이 이벤트는 p의 확률로 발생) 이면 ceil(x) = x_floor + 1
        # 아니면 (1-p의 확률로 발생) floor(x)
        rounded_delay = torch.where(torch.bernoulli(p).bool(), delay_floor + 1.0, delay_floor)
        rounded_delay = rounded_delay.to(output.device) # Shape: (N, C, D_out, D_in)
        rounded_delay = rounded_delay.clamp(max= (T - 1) - output.argmax(dim=0)) # Prevent delay overflow
        
        # Apply delay by shifting the time dimension, using torch.gather
        output = _shift_vec(output, rounded_delay.long())
        
        # mat = torch.arange(T, device=output.device).view(T,1,1,1,1).repeat(1,N,C,D_out,2)
        # output = torch.gather(output, 0, (mat - delay[None,...]) % T)
        # Equivalently, we can use roll (very slow):
        # for d_out in range(D_out):
        #     for d_in in range(D_in):
        #         output[..., d_out, d_in] = torch.roll(output[..., d_out, d_in], shifts=rounded_delay[0,0,d_out,d_in].item(), dims=0)
        
        # For backward pass, use real-valued delay (not rounded)
        ctx.save_for_backward(input.clone(), output.clone(), delay.clone())
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE): 'delay'에 대한 가짜 그래디언트를 계산하여 반환
        
        grad_output: Loss에 대한 shifted_vec의 그래디언트 (d_loss / d_shifted_vec)
        """
        
        #TODO: Gradient of clamped delay?
        input, output, delay, = ctx.saved_tensors
        # T, N, C, D_out, _2 = input.shape
        T, N, C, D_out, _2 = output.shape
        # D_out, _2 = delay.shape
        
        # # 'delay'에 대한 그래디언트만 계산 (vec의 그래디언트는 None)
        grad_input = None
        grad_delay = None
        
        if ctx.needs_input_grad[0]:
            grad_input = _shift_vec(grad_output, -delay)
        
        # # 'delay'가 그래디언트를 요구할 때만 (needs_input_grad[1]) 계산
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

class JeffressLinear(torch.nn.Module):
    def __init__(self, radius: int, tau: float = 2., bias = False) -> None:
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
        
        self.min_diff = None

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
            
        T, N, C, _ = input.shape
        
        # Duplicate last dimension for D_out and D_in, to apply synapse-wise delay.
        # a out spikes from each input neuron map to d_out output neurons: total d_out output delays.
        output = input.unsqueeze(-2).repeat([1] * (len(input.shape) - 1) + [self.out_features, 1]) # ..., 2 -> ..., d_out, 2
        
        # Apply synapse-wise delay
        output = JeffressDelay.apply(output, self.delay) # output shape: (T, N, C, D_out, 2)
        assert isinstance(output, torch.Tensor)
        
        output = self.filter(output)  # Apply LIF_Filter, shape: (T, N, C, D_out, 2)
        
        # Apply synapse filter (postsynaptic current kernel) and weight
        output = torch.mul(output, self.weight) # For only one weight
        
        # Sum over input dimension to get output current.
        output = torch.sum(output, dim=-1)
        
        if self.has_bias:
            output += self.log_bias.exp()

        return output
