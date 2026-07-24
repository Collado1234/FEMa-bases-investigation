import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class AttentionQuadraticBasis(BaseBasis):
    """
    phi(d) = 1 / (1 + d^2).
    NOTA: idêntica ao Cauchy Kernel com epsilon=1 fixo — decisão de
    fórmula pendente com o orientador (ver cauchy_kernel.py).
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        dists = np.where(dists == 0, DELTA, dists)
        return 1 / (1 + dists ** 2)
