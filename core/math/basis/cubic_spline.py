import numpy as np
from core.math.basis._compactSupport import CompactSupportBasis
from .parameters import BasisParameters

class CubicSplineBasis(CompactSupportBasis):
    """
    Suporte compacto: w[r] = (1-r)^4, r = d/h, 0 caso r > 1.

    NOTA: ver o mesmo alerta em quartic_spline.py — esta base está
    idêntica à Quartic Spline hoje e não corresponde à forma clássica de
    cubic spline por partes. Precisa de decisão de fórmula.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray,
                        params: BasisParameters
                        )-> np.ndarray:
        

        r = self._normalized_distance(dists, params.h)

        weights = np.where(
            r <= 1,
            (1-r)**4,
            0.0
        )

        total = weights.sum()

        if total > 0:
            weights /= total

        return weights

