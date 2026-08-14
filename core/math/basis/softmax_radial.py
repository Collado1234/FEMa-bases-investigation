import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class SoftmaxRadialBasis(BaseBasis):
    """
    phi(d) = exp(-beta * d).
    NOTA: matematicamente idêntica ao Laplacian Kernel com epsilon=beta
    depois de normalizada — ver nota em laplacian_kernel.py.
    """
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        soma = np.sum(np.exp(-beta * dists), axis=1, keepdims=True)

        return np.exp((-beta * dists)/soma)
