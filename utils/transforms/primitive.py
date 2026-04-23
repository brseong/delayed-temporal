import torch
from jaxtyping import Float, Int
from math import log, exp
from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain

@check_domain
def pulse_width_modulation_operator(
    t_A: torch.Tensor, 
    domain_t_A: TimeBounds,
    t_B: torch.Tensor | float, 
    domain_t_B: TimeBounds | float,
    V: torch.Tensor,
    domain_V: PotentialBounds
) -> tuple[torch.Tensor, PotentialBounds]:
    """Pulse-Width Modulation operator (psi_PWM)
    Processes two distinct time intervals simultaneously. Computes interval arithmetic for output bounds unconditionally.
    """
    res = V * (t_B - t_A)
    
    t_B_min = domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.min
    t_B_max = domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.max
    
    # Temporal difference bounds: dt = t_B - t_A
    dt_min = t_B_min - domain_t_A.max
    dt_max = t_B_max - domain_t_A.min
    
    v_min, v_max = domain_V.min, domain_V.max
    
    # Interval arithmetic for multiplication V * dt
    p1 = v_min * dt_min
    p2 = v_min * dt_max
    p3 = v_max * dt_min
    p4 = v_max * dt_max
    
    return res, PotentialBounds(min(p1, p2, p3, p4), max(p1, p2, p3, p4))