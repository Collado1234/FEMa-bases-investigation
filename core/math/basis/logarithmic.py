import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class LogarithmicBasis(BaseBasis):
    """
    phi(d) = 1 / log(1 + d + c).
    Guarda contra log(1+d+c) -> 0 (divisão por zero).
    """
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        c = self._require(params)["c"]
        denom = np.log(1 + dists + c)
        denom = np.where(np.abs(denom) < DELTA, DELTA, denom)
        return 1 / denom
