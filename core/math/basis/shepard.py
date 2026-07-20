import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters
from ..neighboor_search import BaseSearch

class ShepardBasis(BaseBasis):
    """
    Base de Shepard para interpolação baseada em vizinhos.

        w_i = 1 / d_i^z ,  depois normalizado (partição da unidade).

    Interpoladora: quando d_i = 0, o peso correspondente domina e tende
    a 1, os demais a 0.
    """
    PARAMS = ("z",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        z = self._require(params)["z"]
        dists = np.where(dists == 0, DELTA, dists)  # evita divisao por zero
        weights = 1.0 / (dists ** z)
        weights /= np.sum(weights)  # particao de unidade
        return weights
