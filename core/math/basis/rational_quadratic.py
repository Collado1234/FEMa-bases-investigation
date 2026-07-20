import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class RationalQuadraticBasis(BaseBasis):
    """
    Rational Quadratic kernel:
        w_i = (1 + d_i^2 / (2*alpha*l^2))^(-alpha), normalizada.
    (Rasmussen & Williams, Gaussian Processes for Machine Learning, 2006, eq. 4.19)
    """
    PARAMS = ("alpha", "l")

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        alpha, l = values["alpha"], values["l"]

        weights = (1 + (dists ** 2) / (2 * alpha * l ** 2)) ** (-alpha)
        weights /= weights.sum()
        return weights
