
import numpy as np
import sympy as sp
from typing import SupportsFloat

def get_gamma(tau:float) -> float:
        return 1/(1 - np.exp(-1/tau))

def get_weight(TAU_S:SupportsFloat, TAU_M:SupportsFloat, TOLERANCE:SupportsFloat) -> SupportsFloat:
        w, t, tau_s, tau_m, tolerance = sp.symbols('w t tau_s tau_m tolerance', real=True, positive=True)

        psp = w * tau_s /(tau_m - tau_s) * ((sp.exp(-t/tau_m)-sp.exp(-t/tau_s)) + (sp.exp(-(t-tolerance)/tau_m)-sp.exp(-(t-tolerance)/tau_s)))
        psp_dot = sp.diff(psp, t)
        
        eq_argmax_t = sp.Eq(psp_dot.expand(), 0)
        argmax_t = sp.solve(eq_argmax_t, t)[0]
        psp_max = sp.simplify(psp.subs(t, argmax_t))
        psp_max = psp_max.subs({tau_m: TAU_M, tau_s: TAU_S})

        w_eq = sp.Eq(psp_max.subs(tolerance, TOLERANCE), 1)

        return float(sp.solve(w_eq, w)[0])

if __name__ == "__main__":
    TAU_S = 1.0
    TAU_M = 10.0
    TOLERANCE = 1.0

    weight = get_weight(TAU_S, TAU_M, TOLERANCE)
    print(f"Computed synaptic weight: {weight:.4f}")

