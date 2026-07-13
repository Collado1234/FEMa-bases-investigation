import numpy as np
from core.math.basis import BaseBasis

class HarmonicBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, nu:float) -> np.ndarray:

        weights = ( 1 + (dists**2) / nu ) ** -( (nu+ 1.0)/2.0 )

        weights /= np.sum(weights)