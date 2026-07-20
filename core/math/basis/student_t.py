import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class StudentTBasis(BaseBasis):
    """
    Kernel Student-t: w_i = (1 + d_i^2/nu)^(-(nu+1)/2), normalizada.
    (van der Maaten & Hinton, 2008 - t-SNE usa essa mesma forma)
    """
    PARAMS = ("nu",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        nu = self._require(params)["nu"]
        weights = (1 + (dists ** 2 / nu)) ** -((nu + 1) / 2)
        weights /= np.sum(weights)
        return weights
