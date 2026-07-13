import numpy as np
from core.math.basis import BaseBasis

class WendlandC2Basis(BaseBasis):
    """
    Funcao de base:
        w_i[r] = (1-r)^4(4r+1)

        onde r = d/h sendo d, distancia e h raio de suporte.

        Verificar se é isso mesmo
    """
    def __init__(self, search):
        super().__init__(search)

    def compute_weights(
        self,
        dists: np.ndarray,
        h: float | None = None
    ) -> np.ndarray:

        if h is None:   # verificar isso daqui
            h = np.max(dists)

        r = dists / h

        weights = np.where(
            r <= 1,
            (1 - r) ** 4 * (4 * r + 1),
            0.0
        )

        total = weights.sum()

        if total > 0:
            weights /= total

        return weights

        
