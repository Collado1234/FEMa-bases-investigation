import numpy as np
from .base_basis import BaseBasis

class MultiquadraticBasis(BaseBasis):
    """
    Multiquadric radial basis.

    w_i = sqrt(d_i² + c²)

    Após o cálculo, os pesos são normalizados para formar
    uma partição da unidade.
    """
    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self,
                        dists:np.ndarray,
                        c:float = 1.0
                        ) -> np.ndarray:
        
        weights = np.sqrt((dists)**2 + c**2)
        weights /= np.sum(weights) # particao de unidade

        return weights
        
