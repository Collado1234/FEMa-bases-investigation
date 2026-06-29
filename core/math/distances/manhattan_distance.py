import numpy as np
from .base_distance import BaseDistance

class ManhattanDistance(BaseDistance):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if x1.ndim == 1 and x2.ndim == 2:
            return np.sum(np.abs(x1 - x2), axis=1)
        return np.sum(np.abs(x1 - x2))
