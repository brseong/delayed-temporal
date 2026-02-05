import torch

torch.jit.script
def discounted_cumsum(input: torch.Tensor, discount: float) -> torch.Tensor:
    """
    Compute the discounted cumulative sum of the input tensor along the time dimension (dim=0).
    
    :param input: Input tensor of shape (T, ...)
    :param discount: Discount factor (float)
    :return: Discounted cumulative sum tensor of the same shape as input
    """
    T = input.shape[0]
    output = torch.zeros_like(input)
    
    running_sum = torch.zeros_like(input[0])
    for t in range(T):
        running_sum = input[t] + discount * running_sum
        output[t] = running_sum
    
    return output