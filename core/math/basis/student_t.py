import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class StudentTBasis(BaseBasis):
    """
    Kernel Student-t: phi(d) = (1 + d^2/nu)^(-(nu+1)/2).
    (van der Maaten & Hinton, 2008 - t-SNE usa essa mesma forma)
    """
    PARAMS = ("nu",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        nu = self._require(params)["nu"]
        return (1 + (dists ** 2 / nu)) ** -((nu + 1) / 2)
