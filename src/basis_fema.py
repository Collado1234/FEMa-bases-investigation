import numpy as np

def _euclidian_distance(x_train:np.ndarray, x_sample:np.ndarray) -> np.ndarray:
    return np.linalg.norm(x_train - x_sample, axis=1)

def _find_k_nearest_indices(distances: np.ndarray, k: int) -> np.ndarray:
    if k != 0:
        return np.argpartition(distances, k)[:k]
    else:
        return np.arange(len(distances))

def _shepard_compute(dist: np.ndarray, z: int) -> np.ndarray:
    weights = 1.0 / (dist ** z)
    weights /= np.sum(weights)
    return weights

class Basis:
    def __init__(self) -> None:
        pass

    def sheppardBasis(self, X_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, k: int = 2, z: int = 2):
        """
        Compute the Shepard's interpolation weights for a vector x and a matrix Y.

        Parameters:
        x (numpy.ndarray): A 1D array of shape (d,).
        Y (numpy.ndarray): A 2D array of shape (n, d).
        p (float): The power parameter for Shepard's interpolation.

        Returns:
        numpy.ndarray: A 1D array of shape (n,) containing the Shepard's interpolation weights.
        """
        distances = _euclidian_distance(X_train, x_test)
        k_nearest_indices = _find_k_nearest_indices(distances, k)
        x_train_k_nearest = X_train[k_nearest_indices]
        y_train_k_nearest = y_train[k_nearest_indices]

        dist = np.where(dist == 0, 1e-10, dist)  # Evita divisão por zero

        weights = _shepard_compute(dist, z)
        predicted = np.sum(weights * y_train_k_nearest)

        if np.isnan(predicted):
            predicted = np.mean(y_train)
        
        return predicted

