import numpy as np
from core.math.basis import BaseBasis

class SoftmaxRadialBasis(BaseBasis):
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray, beta:float) -> np.ndarray:

        weights = np.exp(-beta*dists)
        weights = weights/ np.sum(weights)

        return weights

