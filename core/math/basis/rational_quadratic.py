import numpy as np
from core.math.basis import BaseBasis

class RationalQuadraticBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, alpha:float, l) -> np.ndarray:

        weights = (
            1 + (dists**2)/(2*alpha*l**2)
        ) ** (-alpha)

        weights /= weights.sum()

        return weights