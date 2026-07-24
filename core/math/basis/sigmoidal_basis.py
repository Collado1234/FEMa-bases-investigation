import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class SigmoidalBasis(BaseBasis):
    """phi(d) = 1 / (1 + exp(alpha*(d - c)))."""
    PARAMS = ("alpha", "c")

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        alpha, c = values["alpha"], values["c"]
        return 1 / (1 + np.exp(alpha * (dists - c)))
