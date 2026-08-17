import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class HarmonicBasis(BaseBasis):
    """
    phi(d) = 1 / (1 + d).

    Kernel harmônico clássico de interpolação, parente do potencial de
    Newton 1/r. É formalmente um "parente distante" do Student-t (ver
    student_t.py): StudentTBasis com nu->infinito e expoente linear em
    vez de quadrático se reduz a algo próximo, mas não são a mesma
    fórmula — mantidas como bases separadas.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        return 1 / (1 + dists)

