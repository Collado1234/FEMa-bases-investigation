import numpy as np
from ._compactSupport import CompactSupportBasis
from .parameters import BasisParameters

class WendlandC2Basis(CompactSupportBasis):
    """
    Wendland C2 (Wendland, 1995), suporte compacto, classe C^2:
        w[r] = (1-r)^4 (4r+1),  r = d/h,  0 caso r > 1

    h é opcional: se None, usa max(dists) como raio de suporte (ver
    CompactSupportBasis._normalized_distance). Por isso h NÃO passa por
    self._require — None é um valor válido ("modo automático"), não
    "parâmetro faltando".
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        r = self._normalized_distance(dists, params.h)

        weights = np.where(
            r <= 1,
            (1 - r) ** 4 * (4 * r + 1),
            0.0
            )

        total = weights.sum()
        if total > 0:
            weights /= total

        return weights
