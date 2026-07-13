import numpy as np
from .base_basis import BaseBasis

class CompactSupportBasis(BaseBasis):

    def _normalized_distance(
        self,
        dists: np.ndarray,
        h: float | None = None
    ):
        if h is None:
            h = np.max(dists)

        return dists / h