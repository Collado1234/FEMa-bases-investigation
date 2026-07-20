import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters
from ..neighboor_search import BaseSearch

class RadialBasis(BaseBasis):
    """
    Base radial normalizada com função-mãe sino gaussiano:
        Psi(r) = exp(-0.5 * (r / r0)^2)

    Onde r = distância entre x e xi, r0 = raio efetivo (parâmetro z).
    Normalizada, torna-se partição da unidade, mas não é interpoladora.
    """
    PARAMS = ("z",)

    def __init__(self, search):
        super().__init__(search)

    @staticmethod
    def _mother_function(r: np.ndarray, r0: float) -> np.ndarray:
        return np.exp(-0.5 * (r / r0) ** 2)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        r0 = self._require(params)["z"]
        weights = self._mother_function(dists, r0=r0)

        phi_sum = np.sum(weights)
        if phi_sum == 0:
            return np.ones(len(dists)) / len(dists)  # fallback uniforme

        weights /= phi_sum  # particao de unidade
        return weights
