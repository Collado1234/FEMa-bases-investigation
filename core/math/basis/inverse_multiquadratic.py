import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class InverseMultiquadraticBasis(BaseBasis):
    """phi(d) = 1 / sqrt(d^2 + c^2)."""
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        c = self._require(params)["c"]
        return 1 / (np.sqrt(dists ** 2 + c ** 2) + DELTA)
