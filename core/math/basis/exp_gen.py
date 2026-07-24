import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class ExponentialGenBasis(BaseBasis):
    """phi(d) = exp(-epsilon * d^p) (Generalized exponential)."""
    PARAMS = ("epsilon", "p")

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        epsilon, p = values["epsilon"], values["p"]
        dists = np.where(dists == 0, DELTA, dists)
        return np.exp(-epsilon * (dists ** p))
