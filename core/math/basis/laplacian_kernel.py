import numpy as np
from core.math.basis import BaseBasis

class LaplacianKernelBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, epsilon:float) -> np.ndarray:

        weights = np.exp(-epsilon * dists)
        weights /= weights.sum()

        return weights