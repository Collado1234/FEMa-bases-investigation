import numpy as np
from ._compactSupport import CompactSupportBasis
from .parameters import BasisParameters

class QuarticSplineBasis(CompactSupportBasis):
    """
    Suporte compacto: phi(r) = (1-r)^4, r = d/h, 0 caso r > 1.
    NOTA: hoje idêntica a CubicSplineBasis (ver ali) — decisão de
    fórmula pendente com o orientador.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        r = self._normalized_distance(dists, params.h)
        return np.where(r <= 1, (1 - r) ** 4, 0.0)
