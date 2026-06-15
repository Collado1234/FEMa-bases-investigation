import numpy as np

class Distance:
    @staticmethod
    def euclidean_distance(X: np.ndarray, sample: np.ndarray):
        X = np.asarray(X, dtype=float)
        sample = np.asarray(sample, dtype=float)

        if X.ndim == 1:
            return float(np.linalg.norm(X - sample))

        return np.linalg.norm(X - sample, axis=1)