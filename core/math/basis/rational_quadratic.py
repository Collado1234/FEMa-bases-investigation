import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class RationalQuadraticBasis(BaseBasis):
    """
    phi(d) = (1 + d^2 / (2*alpha*l^2))^(-alpha).
    (Rasmussen & Williams, GPML, 2006, eq. 4.19)
    """
    PARAMS = ("alpha", "l")

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        values = self._require(params)
        alpha, l = values["alpha"], values["l"]
        return (1 + (dists ** 2) / (2 * alpha * l ** 2)) ** (-alpha)
