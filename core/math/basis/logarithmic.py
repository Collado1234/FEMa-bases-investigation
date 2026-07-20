import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class LogarithmicBasis(BaseBasis):
    """
    w_i = 1 / log(1 + d_i + c), normalizada.

    Guarda contra log(1+d+c) -> 0 (ex.: d=0 com c pequeno), que antes
    causava divisão por zero silenciosa.
    """
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        c = self._require(params)["c"]
        denom = np.log(1 + dists + c)
        denom = np.where(np.abs(denom) < DELTA, DELTA, denom)

        weights = 1 / denom
        weights /= np.sum(weights)
        return weights
