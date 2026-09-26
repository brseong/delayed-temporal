import sympy as sp
import torch
from .layer import SynapseFilter
from .transforms.parameter import tau2beta, tau2gamma

def get_extremum_time(TAU_S:float, TAU_M:float, TOLERANCE:float) -> float:
        w, t = sp.symbols('w t', real=True, positive=True)

        psp = (TAU_S /(TAU_M - TAU_S)) * ((sp.exp(-t/TAU_M)-sp.exp(-t/TAU_S)) + (sp.exp(-(t-TOLERANCE)/TAU_M)-sp.exp(-(t-TOLERANCE)/TAU_S)))
        psp_dot = sp.diff(psp, t)

        eq_argmax_t = sp.Eq(psp_dot.expand(), 0)
        argmax_t = sp.solve(eq_argmax_t, t)[0]

        return float(argmax_t)

def get_weight(TAU_S:float, TAU_M:float, TOLERANCE:float, time_steps:int) -> float:
        
        ### Previous symbolic solution (commented out) ###
        # BETA_S = tau2beta(TAU_S)
        # BETA_M = tau2beta(TAU_M)
        # w, t = sp.symbols('w t', real=True, positive=True)

        # # psp = w * TAU_S /(TAU_M - TAU_S) * ((sp.exp(-t/TAU_M)-sp.exp(-t/TAU_S)) + (sp.exp(-(t-TOLERANCE)/TAU_M)-sp.exp(-(t-TOLERANCE)/TAU_S)))
        # psp = w * ((sp.exp(-t/TAU_M)-sp.exp(-t/TAU_S)) + (sp.exp(-(t-TOLERANCE)/TAU_M)-sp.exp(-(t-TOLERANCE)/TAU_S)))
        # psp_dot = sp.diff(psp, t)

        # eq_argmax_t = sp.Eq(psp_dot.expand(), 0)
        # argmax_t = sp.solve(eq_argmax_t, t)[0]
        # psp_max = sp.simplify(psp.subs(t, argmax_t))

        # w_eq = sp.Eq(psp_max, 1)

        # soln = float(sp.solve(w_eq, w)[0])

        # soln *= BETA_M - BETA_S # To correct the difference between analytic soln and numerical soln.

        argmax_t = get_extremum_time(TAU_S, TAU_M, TOLERANCE)
        
        synapse_filter = SynapseFilter(beta=tau2beta(TAU_S), step_mode='m', learnable=False)
        mem_filter = SynapseFilter(beta=tau2beta(TAU_M), step_mode='m', learnable=False)
        spike = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=time_steps).view(-1, 1) # Spike at t=0
        V = mem_filter(synapse_filter(spike.float()))
        soln = V[int(argmax_t)].item()
        
        print(f"weight soln: {soln}")
        
        return soln

if __name__ == "__main__":
    TAU_S = 1.0
    TAU_M = 10.0
    TOLERANCE = 1.0
    TIME_STEPS = 20

    print(tau2gamma(TAU_S), tau2gamma(TAU_M))
    weight = get_weight(TAU_S, TAU_M, TOLERANCE, TIME_STEPS)
    print(f"Computed synaptic weight: {weight:.4f}")

