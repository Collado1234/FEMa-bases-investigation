import numpy as np
import math

class Basis:
    def __init__(self) -> None:
        pass

    @staticmethod
    def shepardBasis(train_x: np.array, train_y: np.array, test_one_sample: np.array, k: int, z: int) -> float:
        dist = np.linalg.norm(train_x - test_one_sample, axis=1)
        
        if k != 0:
            k_nearest_indices = np.argpartition(dist, k)[:k]
        else:
            k_nearest_indices = np.arange(len(train_x))
        
        dist = dist[k_nearest_indices]
        train_y_k_nearest = train_y[k_nearest_indices]

        dist = np.where(dist == 0, 1e-10, dist)  # Evita divisão por zero
        weights = 1.0 / (dist ** z)
        weights /= np.sum(weights)

        predicted = np.sum(weights * train_y_k_nearest)

        if math.isnan(predicted):
            predicted = np.mean(train_y)

        return predicted

    @staticmethod
    def radialBasis(train_x: np.array, train_y: np.array, test_one_sample: np.array, k: int, z: int) -> float:
        dist = np.linalg.norm(train_x - test_one_sample, axis=1)

        if k != 0:
            k_nearest_indices = np.argpartition(dist, k)[:k]
        else:
            k_nearest_indices = np.arange(len(train_x))
        
        dist = dist[k_nearest_indices]
        train_y_k_nearest = train_y[k_nearest_indices]

        rbf = np.exp(-(z * dist) ** 2)
        weights = rbf / np.sum(rbf)

        predicted = np.sum(weights * train_y_k_nearest)

        if math.isnan(predicted):
            predicted = np.mean(train_y)

        return predicted
