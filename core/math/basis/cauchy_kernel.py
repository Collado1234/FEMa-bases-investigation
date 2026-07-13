import numpy as np
from core.math.basis import BaseBasis

class CauchyKernelBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, epsilon:float) -> np.ndarray:

        weights = 1.0 / (1.0 + (epsilon * dists)**2)
        weights /= weights.sum()

        return weights