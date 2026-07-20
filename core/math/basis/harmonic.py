import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class HarmonicBasis(BaseBasis):
    """
    w_i = (1 + d_i^2/nu)^(-(nu+1)/2), normalizada.

    NOTA: esta é, formalmente, a mesma fórmula do Student-t kernel (ver
    student_t.py) — confirmado empiricamente. "Harmonic" na literatura
    costuma se referir a outra coisa (ex. média harmônica de distâncias,
    ou 1/(1+d^2)^p). Corrigido aqui apenas o bug estrutural (faltava
    `return`, então a base sempre devolvia None); a duplicação
    conceitual com Student-t continua e precisa de decisão de fórmula.
    """
    PARAMS = ("nu",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, params: BasisParameters) -> np.ndarray:
        nu = self._require(params)["nu"]
        weights = ( 1 + (dists**2) / nu ) ** -( (nu+ 1.0)/2.0 )

        weights /= np.sum(weights)

        return weights