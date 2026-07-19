import numpy as np
from core.math.basis import _compactSupport

class CosineBasis(Co):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, alpha:float, h) -> np.ndarray:

        r = dists / h

        weights = np.where(
            r <= 1,
            np.cos(np.pi*r/2)**2,
            0.0
        )

        weights /= weights.sum()

        return weights