import numpy as np
from core.math.basis import BaseBasis
from .parameters import BasisParameters
class EntropicBasis(BaseBasis):
    """
    w_i = exp(-beta * d_i^2), normalizada.

    Corrigidos aqui: faltava `return` (a base sempre devolvia None) e a
    normalização usava `weights /= weights` (divisão de cada peso por si
    mesmo, resultando em [1,1,...,1] em vez de partição da unidade).
    """
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray, params: BasisParameters) -> np.ndarray:

        beta = self._require(params)["beta"]

        weights = np.exp( - (beta * dists ** 2) )

        weights /= np.sum(weights)

        return weights