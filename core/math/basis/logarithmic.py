import numpy as np 
from core.math.basis import BaseBasis

class LogarithimicBasis(BaseBasis):
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray, c:float) -> np.ndarray:

        weights = 1 / ( np.log(1 + dists + c) )
        weights /= np.sum(weights)
        
        return weights