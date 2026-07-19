import numpy as np
from core.math.basis.base_basis import BaseBasis

class SigmoidalBasis(BaseBasis):
    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray, alpha:float, c:float) -> np.ndarray:
        
        weights = 1 / (1 + np.exp(alpha*(dists - c) ) )
        
        weights /= weights

        return weights