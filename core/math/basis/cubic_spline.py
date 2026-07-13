import numpy as np
from core.math.basis._compactSupport import CompactSupportBasis

class CubicSplineBasis(CompactSupportBasis):

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray,
                        h:float | None
                        )-> np.ndarray:
        

        r = self._normalized_distance(dists, h)

        weights = np.where(
            r <= 1,
            (1-r)**4,
            0.0
        )

        total = weights.sum()

        if total > 0:
            weights /= total

        return weights

