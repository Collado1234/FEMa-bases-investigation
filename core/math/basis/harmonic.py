import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class HarmonicBasis(BaseBasis):
    """
    phi(d) = (1 + d^2/nu)^(-(nu+1)/2).
    NOTA: formalmente idêntica ao Student-t kernel (ver student_t.py) —
    decisão de fórmula pendente com o orientador.
    """
    PARAMS = ("nu",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        nu = self._require(params)["nu"]
        return (1 + (dists ** 2) / nu) ** -((nu + 1.0) / 2.0)
