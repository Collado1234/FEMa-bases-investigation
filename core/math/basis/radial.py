import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class RadialBasis(BaseBasis):
    """
    Função-mãe sino gaussiano: phi(r) = exp(-0.5 * (r / r0)^2), r0 = z.
    """
    PARAMS = ("z",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        r0 = self._require(params)["z"]
        return np.exp(-0.5 * (dists / r0) ** 2)
