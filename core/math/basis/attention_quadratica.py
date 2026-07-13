import numpy as np 
from core.math.basis import BaseBasis

class AttentionQuadraticBasis(BaseBasis):
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray) -> np.ndarray:
        dists = np.where(dists == 0, 1e-10, dists) 
        
        weights = 1/(1 + dists**2)
        weights /= weights.sum() 

        return weights