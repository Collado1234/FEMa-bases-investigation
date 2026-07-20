import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class LaplacianKernelBasis(BaseBasis):
    """
    w_i = exp(-epsilon * d_i), normalizada.
    Ver nota de duplicação em softmax_radial.py.
    """
    PARAMS = ("epsilon",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        weights = np.exp(-epsilon * dists)
        weights /= weights.sum()
        return weights
