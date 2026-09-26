import numpy as np

def tau2gamma(tau:float) -> float:
        return 1/(1 - tau2beta(tau))

def tau2beta(tau:float) -> float:
        return np.exp(-1/tau)