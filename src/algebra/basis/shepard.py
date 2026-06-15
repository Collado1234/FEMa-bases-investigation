import numpy as np
from algebra.basis.basis import Basis
from algebra.distances.base_distance import BaseDistance
from algebra.neighboor_search.base_search import BaseSearch


class ShepardBasis(Basis):
    """
    Base de Sheppard para interpolação.

    Utiliza pesos inversamente proporcionais à distância:
        w_i = 1 / d_i^z
    Normalizado para partição da unidade:
        w_i = w_i / Σ w_j
    """

    def __init__(self, distance: BaseDistance, search: BaseSearch):
        super().__init__(distance, search)

    def predict(self, sample: np.ndarray, k: int, z: float) -> float:
        """
        Interpola o valor para uma amostra usando a base de Sheppard.

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)
            z:      Expoente da distância inversa (z > 0)

        Returns:
            Valor interpolado.
        """
        dists = self.distance.compute(self.X_train, sample)
        indices = self.search.search(dists, k)

        k_dists = dists[indices]
        k_labels = self.y_train[indices]

        k_dists = np.where(k_dists == 0, 1e-10, k_dists)  # evita divisão por zero
        weights = 1.0 / (k_dists ** z)
        weights /= np.sum(weights)  # partição de unidade

        predicted = np.dot(weights, k_labels)

        if np.isnan(predicted):
            return float(np.mean(self.y_train))

        return float(predicted)