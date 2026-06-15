from typing import Tuple
import numpy as np
from algebra.neighboor_search.base_search import BaseSearch
from algebra.distances.base_distance import BaseDistance
from algebra.distances.euclidean_distance import EuclideanDistance

class BruteForceSearch(BaseSearch):
    """
    Busca de vizinhos por força bruta com distância euclidiana.

    Calcula a distância de sample para todos os pontos de treino
    e seleciona os k menores. Complexidade O(n) por query.

    Para datasets grandes, prefira KDTreeSearch ou BallTreeSearch.
    """

    def __init__(self):
        self.X_train = None

    def build(self, X: np.ndarray) -> None:
        """
        Armazena X_train para uso nas queries.

        Args:
            X: Features de treino (n_samples, n_features)
        """
        self.X_train = X

    def query(self, sample: np.ndarray, k: int, distance: BaseDistance = EuclideanDistance()) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula distância euclidiana para todos os pontos e retorna os k menores.

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)
            distance: Métrica de distância a ser utilizada

        Returns:
            indices: Índices dos k vizinhos (k,)
            dists:   Distâncias euclidianas dos k vizinhos (k,)
        """
        if self.X_train is None:
            raise RuntimeError("Chame build() antes de query().")

        dists = distance.compute(self.X_train, sample)

        if k == 0:
            indices = np.arange(len(self.X_train))
        else:
            indices = np.argpartition(dists, k)[:k]

        return indices, dists[indices]