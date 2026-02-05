
import numpy as np
import sympy as sp

def tau2gamma(tau:float) -> float:
        return 1/(1 - tau2beta(tau))

def tau2beta(tau:float) -> float:
        return np.exp(-1/tau)

def get_weight(TAU_S:float, TAU_M:float, TOLERANCE:float) -> float:
        BETA_S = tau2beta(TAU_S)
        BETA_M = tau2beta(TAU_M)
        w, t = sp.symbols('w t', real=True, positive=True)

        # psp = w * TAU_S /(TAU_M - TAU_S) * ((sp.exp(-t/TAU_M)-sp.exp(-t/TAU_S)) + (sp.exp(-(t-TOLERANCE)/TAU_M)-sp.exp(-(t-TOLERANCE)/TAU_S)))
        psp = w * ((sp.exp(-t/TAU_M)-sp.exp(-t/TAU_S)) + (sp.exp(-(t-TOLERANCE)/TAU_M)-sp.exp(-(t-TOLERANCE)/TAU_S)))
        psp_dot = sp.diff(psp, t)

        eq_argmax_t = sp.Eq(psp_dot.expand(), 0)
        argmax_t = sp.solve(eq_argmax_t, t)[0]
        psp_max = sp.simplify(psp.subs(t, argmax_t))

        w_eq = sp.Eq(psp_max, 1)

        soln = float(sp.solve(w_eq, w)[0])
        # print(f"soln: {soln}")

        soln *= BETA_M - BETA_S # To correct the difference between analytic soln and numerical soln.
        
        return soln

if __name__ == "__main__":
    TAU_S = 1.0
    TAU_M = 10.0
    TOLERANCE = 1.0

    print(tau2gamma(TAU_S), tau2gamma(TAU_M))
    weight = get_weight(TAU_S, TAU_M, TOLERANCE)
    print(f"Computed synaptic weight: {weight:.4f}")

