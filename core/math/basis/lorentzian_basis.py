import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class LorentzianBasis(BaseBasis):
    """
    phi(d) = 1 / (1 + d^4).
    NOTA: na literatura, "Lorentzian" costuma ser sinônimo de Cauchy
    (1/(1+d^2) com escala ajustável) — decisão de fórmula pendente.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        dists = np.where(dists == 0, DELTA, dists)
        return 1 / (1 + dists ** 4)
