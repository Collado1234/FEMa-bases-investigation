import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class SigmoidalBasis(BaseBasis):
    """
    w_i = 1 / (1 + exp(alpha*(d_i - c))), normalizada.

    Corrigido aqui: a normalização usava `weights /= weights` (divisão de
    cada peso por si mesmo, resultando em [1,1,...,1] em vez de partição
    da unidade).
    """
    PARAMS = ("alpha", "c")

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        alpha, c = values["alpha"], values["c"]

        weights = 1 / (1 + np.exp(alpha * (dists - c)))
        weights /= np.sum(weights)
        return weights
