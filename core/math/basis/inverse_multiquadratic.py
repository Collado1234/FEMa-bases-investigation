import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class InverseMultiquadraticBasis(BaseBasis):
    """
    Inverse Multiquadric radial basis: w_i = 1 / sqrt(d_i^2 + c^2), normalizada.
    """
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self,
                        dists:np.ndarray,
                        params: BasisParameters
                        ) -> np.ndarray:
        
        c = self._require(params)["c"]

        weights = 1 / ( np.sqrt((dists)**2 + c**2) + DELTA)
        
        weights /= np.sum(weights) # particao de unidade

        return weights
        
