import numpy as np
from core.math.basis import BaseBasis
from core.math.basis.parameters import BasisParameters

class CauchyKernelBasis(BaseBasis):
    """
    w_i = 1 / (1 + (epsilon * d_i)^2), normalizada. (Basak, 2008)
    """
    PARAMS = ("epsilon",)
    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        weights = 1.0 / (1.0 + (epsilon * dists)**2)
        weights /= weights.sum()

        return weights