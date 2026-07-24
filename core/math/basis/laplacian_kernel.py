import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class LaplacianKernelBasis(BaseBasis):
    """
    phi(d) = exp(-epsilon * d).
    NOTA: idêntica a SoftmaxRadialBasis com beta=epsilon — decisão de
    fórmula pendente com o orientador (ver softmax_radial.py).
    """
    PARAMS = ("epsilon",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        epsilon = self._require(params)["epsilon"]
        return np.exp(-epsilon * dists)
