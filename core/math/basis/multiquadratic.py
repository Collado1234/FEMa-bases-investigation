import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class MultiquadraticBasis(BaseBasis):
    """
    phi(d) = sqrt(d^2 + c^2). Monotonicamente CRESCENTE em d por
    construção (Hardy, 1971) — favorece vizinhos distantes quando usada
    como peso normalizado; é objeto de estudo do projeto, não um bug.
    """
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        c = self._require(params)["c"]
        return np.sqrt(dists ** 2 + c ** 2)
