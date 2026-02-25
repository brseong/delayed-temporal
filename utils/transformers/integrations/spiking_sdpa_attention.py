from functools import lru_cache
from typing import Literal

import torch, math, wandb
from time import perf_counter
# from torch.utils.tensorboard import SummaryWriter

from transformers.utils import is_torch_npu_available, is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal
from utils.datasets import encode_temporal_th, unnormalize_net_output
from utils.load import AbstractL2Net, load_l2net_model, load_abst_l2net_model

logger = logging.get_logger(__name__)
# writer = SummaryWriter(log_dir="runs/spiking_sdpa_attention")

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()

class NetCache:
    def __init__(self):
        self.cache = {}
        self._hex = None
        self._parallel = True
    
    @property
    def hex(self):
        assert self._hex is not None, "L2Net hash is not registered."
        return self._hex
    @hex.setter
    def hex(self, hex:str):
        assert self._hex is None, "L2Net hash is already registered."
        self._hex = hex
        
    @property
    def parallel(self):
        return self._parallel
    @parallel.setter
    def parallel(self, parallel:bool):
        self._parallel = parallel
        
    def __getitem__(self, device:torch.device) -> tuple[AbstractL2Net, dict[str, object]]:
        if device not in self.cache:
            self.cache[device] = load_abst_l2net_model(self.hex, device=device)
        return self.cache[device]

netcache = NetCache()

def get_sse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    get_l2_distance의 Docstring
    
    :param a: Shape: (N, S, D)
    :param b: Shape: (N, S, D)
    :return: Shape: (N, S)
    """
    l2net, l2net_cfg = netcache[a.device]
    min_val, max_val = float(l2net_cfg["MIN_VAL"]), float(l2net_cfg["MAX_VAL"]) # type: ignore
    time_steps, vector_dim = int(l2net_cfg["TIME_STEPS"]), int(l2net_cfg["VECTOR_DIM"]) # type: ignore
    
    a = encode_temporal_th(a.clamp(min=min_val, max=max_val), time_steps, time_pad=0, min_val=min_val, max_val=max_val)
    b = encode_temporal_th(b.clamp(min=min_val, max=max_val), time_steps, time_pad=0, min_val=min_val, max_val=max_val)
    input_ab = torch.stack([a,b], dim=-2)  # To make shape in form (T, N, S, 2, D)
    out = l2net(input_ab).squeeze(-1)  # (T, N, S, 2, D) -> (N, S)
    # out = unnormalize_net_output(out, vector_dim, min_val, max_val)
    return out

def get_scaled_sse_abst(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    get_l2_distance의 Docstring
    
    :param a: Shape: (N, S, D)
    :param b: Shape: (N, S, D)
    :return: Shape: (N, S)
    """
    l2net, l2net_cfg = netcache[a.device]
    min_val, max_val = float(l2net_cfg["MIN_VAL"]), float(l2net_cfg["MAX_VAL"]) # type: ignore
    time_steps, vector_dim = int(l2net_cfg["TIME_STEPS"]), int(l2net_cfg["VECTOR_DIM"]) # type: ignore
    
    input_ab = torch.zeros(*a.shape[:-1], 2, a.shape[-1], device=a.device, dtype=a.dtype)
    input_ab[..., 0, :] = (a.clamp(min = min_val, max = max_val) - min_val) / (max_val - min_val)
    input_ab[..., 1, :] = (b.clamp(min = min_val, max = max_val) - min_val) / (max_val - min_val)
    
    out = l2net(input_ab).squeeze(-1)  # (N, S, 2, D) -> (N, S)
    # out = unnormalize_net_output(out, vector_dim, min_val, max_val)
    return out


def get_scaled_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    get_inner의 Docstring
    
    :param a: Shape: (B, H, S, D)
    :param b: Shape: (B, H, S, D)
    """
    batch, num_attention_heads, seqlen, head_dim = a.shape
    output = torch.zeros((batch, num_attention_heads, seqlen, seqlen), device=a.device, dtype=a.dtype)
    
    # Inefficient version:
    # zeros = torch.zeros((seqlen, head_dim), device=a.device, dtype=a.dtype)
    # for n in range(batch):
    #     for h in range(num_attention_heads):
            # for s in range(seqlen):
            #     Q, k = a[n,h], b[n,h,s] # Shape: (S, D), (,D)
            #     K = k.unsqueeze(0).expand(seqlen, head_dim) # Shape: (S, D)
            #     output[n,h,s,:] = (
            #         get_sum_square_error(Q, zeros) + get_sum_square_error(K, zeros) - get_sum_square_error(Q, K)
            #     ) / 2
    
    # More efficient version:
    zeros = None
    for h in range(num_attention_heads):
        for s in range(seqlen):
            t = perf_counter()
            Q_hs = a[:, h, s:s+1].expand(-1, seqlen, -1) # Shape: (N, S, D)
            K_h = b[:, h]  # Shape: (N, S, D)
            
            if zeros is None:
                zeros = torch.zeros_like(K_h)
            sse_Q = get_scaled_sse_abst(Q_hs, zeros)  # Shape: (N, S, D) -> (N, S)
            sse_K = get_scaled_sse_abst(K_h, zeros)  # Shape: (N, S, D) -> (N, S)
            sse_QK = get_scaled_sse_abst(Q_hs, K_h)     # Shape: (N, S, D) -> (N, S)
            
            # inner_product = (sse_Q + sse_K - sse_QK) / 2  # Shape: (N, S)
            output[:, h, s] = sse_Q + sse_K - sse_QK # Shape: (N, S)
            
            wandb.log({"spiking_sdpa_attention/get_inner_time_per_head": perf_counter() - t})
    
    return output
    

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# Efficient implementation equivalent to the following:
def spiking_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    # scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = get_scaled_inner(query, key) # * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

# att_min = float("inf")
# att_max = float("-inf")

def spiking_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # To record the distribution of query, key, value tensors to train Jeffress network
    # writer.add_histogram("spiking_sdpa_attention/query", query)
    # writer.add_histogram("spiking_sdpa_attention/key", key)
    # writer.add_histogram("spiking_sdpa_attention/value", value)
    # writer.flush()
    # print("Spiking SDPA Attention - query.shape:", query.shape, "key.shape:", key.shape, "value.shape:", value.shape)
    
    attn_output = spiking_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
