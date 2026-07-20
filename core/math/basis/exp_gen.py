import numpy as np
from core.math.basis import BaseBasis, DELTA
from .parameters import BasisParameters

class ExponentialGenBasis(BaseBasis):
    """
    Generalized exponential: w_i = exp(-epsilon * d_i^p), normalizada.
    """
    PARAMS = ("epsilon", "p")

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        epsilon, p = values["epsilon"], values["p"]

        dists = np.where(dists == 0, DELTA, dists)

        weights = np.exp( -epsilon * ( dists ** p ))

        weights /= np.sum(weights) # particao de unidade

        return weights