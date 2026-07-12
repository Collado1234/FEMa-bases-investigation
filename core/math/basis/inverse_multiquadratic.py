import numpy as np
from .base_basis import BaseBasis

class InverseMultiquadraticBasis(BaseBasis):
    """
    Inverse Multiquadric radial basis.

    w_i = 1 / sqrt(d_i² + c²)

    Após o cálculo, os pesos são normalizados para formar
    uma partição da unidade.
    """
    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self,
                        dists:np.ndarray,
                        c:float = 1.0
                        ) -> np.ndarray:
        
        weights = 1 / ( np.sqrt((dists)**2 + c**2) + DELTA)
        weights /= np.sum(weights) # particao de unidade

        return weights
        
