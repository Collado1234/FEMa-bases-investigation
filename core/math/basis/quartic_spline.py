import numpy as np
from ._compactSupport import CompactSupportBasis
from .parameters import BasisParameters

class QuarticSplineBasis(CompactSupportBasis):
    """
    Suporte compacto: w[r] = (1-r)^4, r = d/h, 0 caso r > 1.

    NOTA: hoje esta fórmula é idêntica à de CubicSplineBasis, e nenhuma
    das duas corresponde às formas clássicas de spline cúbico/quártico
    por partes (ex. B-spline cúbico/quártico usado em SPH, ou Fasshauer,
    "Meshfree Approximation Methods with MATLAB", 2007, cap. 2-3) — a
    fórmula atual é, na verdade, o núcleo do Wendland C2 sem o fator
    (4r+1). Precisa de decisão de fórmula com o orientador antes de
    seguir para os experimentos comparativos; deixado como estava para
    não inventar matemática nesta refatoração.
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
            (1 - r) ** 4,
            0.0
        )

        total = weights.sum()

        if total > 0:
            weights /= total

        return weights

