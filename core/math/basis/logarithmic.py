import numpy as np 
from core.math.basis import BaseBasis

class LogarithimicBasis(BaseBasis):
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray, c:float) -> np.ndarray:

        weights = 1/(1 + dists**2)
        weights /= weights.sum() 

        return weights