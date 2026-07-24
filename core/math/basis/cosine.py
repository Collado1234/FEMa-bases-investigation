import numpy as np
from ._compactSupport import CompactSupportBasis
from .parameters import BasisParameters

class CosineBasis(CompactSupportBasis):
    """
    Raised-cosine, suporte compacto: phi(r) = cos(pi*r/2)^2, r = d/h,
    0 caso r > 1. (variante ao quadrado do cosine kernel de KDE; cf.
    Silverman, Density Estimation for Statistics and Data Analysis, 1986)
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        r = self._normalized_distance(dists, params.h)
        return np.where(r <= 1, np.cos(np.pi * r / 2) ** 2, 0.0)
