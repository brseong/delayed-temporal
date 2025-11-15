import torch
from jaxtyping import Int64, Float
from spikingjelly.activation_based.layer import SynapseFilter
from torch import Tensor

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
        
        return rounded_x.long()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE):
        그래디언트를 그대로 통과시킴
        """
        # (d_loss / d_rounded_x) * (d_rounded_x / d_x)
        # STE는 d_rounded_x / d_x 를 1로 가정함
        print("StochasticRound backward called")
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

class JeffressDelay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, delay: torch.nn.Parameter) -> torch.Tensor:
        """Apply synaptic delay to the one-hot coded input tensor.
        
        :param ctx: Context to save information for backward pass
        :param input: The input tensor to apply delay to. shape: (T, N, C, D_out, 2)
        :type input: torch.Tensor
        :param delay: The delay tensor to apply. shape: (N, C, D_out, 2)
        :type delay: torch.nn.Parameter
        :return: The output tensor after applying synaptic delay. shape: (T, N, C, D_out, 2)
        :rtype: Tensor
        """
        output = input.clone()
        T, N, C, D_out, _ = output.shape
        
        # Apply delay by shifting the time dimension, using torch.gather
        output = _shift_vec(output, delay)
        
        # mat = torch.arange(T, device=output.device).view(T,1,1,1,1).repeat(1,N,C,D_out,2)
        # output = torch.gather(output, 0, (mat - delay[None,...]) % T)
        # Equivalently, we can use roll (very slow):
        # for d_out in range(D_out):
        #     for d_in in range(D_in):
        #         output[..., d_out, d_in] = torch.roll(output[..., d_out, d_in], shifts=rounded_delay[0,0,d_out,d_in].item(), dims=0)
        
        ctx.save_for_backward(output.clone(), delay.clone())
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type: ignore
        """
        Backward Pass (STE): 'delay'에 대한 가짜 그래디언트를 계산하여 반환
        
        grad_output: Loss에 대한 shifted_vec의 그래디언트 (d_loss / d_shifted_vec)
        """
        
        #TODO: Gradient of clamped delay?
        output, delay, = ctx.saved_tensors
        # T, N, C, D_out, _ = output.shape
        # N, C, D_out, D_in = delay.shape
        
        # # 'delay'에 대한 그래디언트만 계산 (vec의 그래디언트는 None)
        grad_input = None
        grad_delay = None
        
        if ctx.needs_input_grad[0]:
            grad_input = _shift_vec(grad_output, -delay)
        
        # # 'delay'가 그래디언트를 요구할 때만 (needs_input_grad[1]) 계산
        if ctx.needs_input_grad[1]:
            # arange = torch.arange(output.shape[0], device=output.device).view(-1,1,1,1,1)
            # score = torch.sum(output * arange, dim=0)  # shape: (N, C, D_out, 2)
            # score = torch.square(score[...,0] - score[...,1])  # shape: (N, C, D_out)
            # score = torch.exp(-score)  # shape: (N, C, D_out)
            
            # score = torch.softmax(output, dim=0) # shape: (T, N, C, D_out, 2)
            # score = torch.linalg.vecdot(score[...,0], score[...,1], dim=0)  # shape: (N, C, D_out)
                
            # grad_delay = (grad_output * score[None,...,None]).sum(dim=(0, 1, 2))
            
            # vec의 그래디언트(None), d의 그래디언트 순서로 반환
            grad_delay = output * grad_output
        
        print("JeffressDelay backward called", grad_input.shape, grad_delay.shape)
        return grad_input, grad_delay.sum(dim=(0))

class JeffressLinear(torch.nn.Module):
    def __init__(self, out_features: int, tau: float = 2., bias = False) -> None:
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
        self.out_features = out_features
        self.bias = bias
        self._log_delay = torch.nn.Parameter(torch.linspace(-4, 0, out_features).view(-1,1).float()) # For symmetry
        # self.weight = torch.nn.Parameter(torch.rand(out_features, 2).exp())
        self._log_weight = torch.nn.Parameter(torch.log(torch.tensor(2.0))) # For only one weight
        # if bias:
        #     self.log_bias = torch.nn.Parameter(torch.tensor(0.))

        self.filter = SynapseFilter(tau=tau, learnable=False, step_mode="m")
        self._loss = None

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    @property
    def delay(self):
        # self._delay_mask = torch.cat([
        #     torch.zeros(self.out_features//2),
        #     torch.ones(self.out_features//2)
        # ])
        # self._delay_mask = torch.stack([self._delay_mask, self._delay_mask.flip(0)], dim=-1)  # Shape: (out_features, 2)
        # return torch.cat([self._log_delay, self._log_delay.flip(0)], dim=1).exp() * self._delay_mask
        return torch.cat([self._log_delay.exp(), self._log_delay.flip(0).exp()], dim=1)

    @property
    def weight(self):
        return self._log_weight.exp()

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
        
        # Sample synaptic delays
        rounded_delay:Int64[Tensor, "N C D_out D_in"]\
            = StochasticRound.apply(
                    self.delay[None,None,...].to(output.device)\
                        .repeat(N, C, 1, 1)
                    ) # type: ignore
                # Shape: (N, C, D_out, D_in)
        rounded_delay = rounded_delay.clamp(max= (T - 1) - output.argmax(dim=0)) # Prevent delay overflow
        
        # Apply synapse-wise delay
        output = JeffressDelay.apply(output, rounded_delay.to(output.device)); assert isinstance(output, torch.Tensor)
        # Apply synapse filter (postsynaptic current kernel) and weight
        # output = self.filter(output)
        output.mul_(self.weight) # For only one weight
        # output.mul_(self.weight[None, None, None, :, :])
        # Sum over input dimension to get output current.
        output = output.sum(dim=-1)
        # if self.bias:
        #     output += self.log_bias.exp()

        return output
