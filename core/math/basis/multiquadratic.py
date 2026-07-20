import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class MultiquadraticBasis(BaseBasis):
    """
    Multiquadric radial basis: w_i = sqrt(d_i^2 + c^2), normalizada.

    NOTA (propositalmente preservada): esta forma é monotonicamente
    CRESCENTE em d, então, usada diretamente como peso normalizado
    (e não resolvendo o sistema linear A*lambda=y da interpolação
    multiquadric clássica de Hardy 1971), ela favorece vizinhos mais
    distantes. Isso é um objeto de estudo do projeto, não um bug.
    """
    PARAMS = ("c",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        c = self._require(params)["c"]
        weights = np.sqrt(dists ** 2 + c ** 2)
        weights /= np.sum(weights)  # particao de unidade
        return weights
