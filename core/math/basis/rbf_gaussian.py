import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class RbfGaussianBasis(BaseBasis):
    """phi(d) = exp(-(epsilon*d)^2)."""
    PARAMS = ("epsilon",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        dists = np.where(dists == 0, DELTA, dists)
        return np.exp(-(epsilon * dists) ** 2)
