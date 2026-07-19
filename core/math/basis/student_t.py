import numpy as np
from core.math.basis import BaseBasis

class StudentTBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, nu:float) -> np.ndarray:

        weights = ( 1 + (dists**2 / nu) ) ** -((nu+1)/2) 

        weights /= np.sum(weights)
        
        return weights