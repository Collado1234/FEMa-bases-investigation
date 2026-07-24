import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class CauchyKernelBasis(BaseBasis):
    """phi(d) = 1 / (1 + (epsilon*d)^2). (Basak, 2008)"""
    PARAMS = ("epsilon",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        return 1.0 / (1.0 + (epsilon * dists) ** 2)
