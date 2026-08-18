import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class EntropicBasis(BaseBasis):
    """phi(d) = exp(-beta * d^2)."""
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        soma = np.sum(np.exp(-beta * dists), axis=1, keepdims=True)
        return np.exp(-beta * dists) / soma
