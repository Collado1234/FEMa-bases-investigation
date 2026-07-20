import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class RbfGaussianBasis(BaseBasis):
    """
    Base radial gaussiana: w_i = exp(-(epsilon*d_i)^2), normalizada.
    """
    PARAMS = ("epsilon",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        dists = np.where(dists == 0, DELTA, dists)

        weights = np.exp(-(epsilon * dists) ** 2)
        weights /= np.sum(weights)  # particao de unidade
        return weights
