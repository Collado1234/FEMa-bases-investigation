import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class ShepardBasis(BaseBasis):
    """
    Base de Shepard: phi(d) = 1 / d^z. Normalizada (compute_weights,
    herdado), é interpoladora: quando d_i = 0, o peso correspondente
    domina e tende a 1, os demais a 0.
    """
    PARAMS = ("z",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        z = self._require(params)["z"]
        dists = np.where(dists == 0, DELTA, dists)  # evita divisao por zero
        return 1.0 / (dists ** z)
