import numpy as np
from algebra.basis.basis import Basis
from algebra.neighboor_search.base_search import BaseSearch


class ShepardBasis(Basis):
    """
    Base de Shepard para interpolação.

    Utiliza pesos inversamente proporcionais à distância:
        w_i = 1 / d_i^z
    Normalizado para partição da unidade:
        w_i = w_i / Σ w_j
    """

    def __init__(self, search: BaseSearch):
        self.search = search

    def compute_weights(self, dists: np.ndarray, z: float) -> np.ndarray:
        """
        Calcula os pesos de Shepard a partir das distâncias.

        Args:
            dists: Distâncias dos k vizinhos (k,)
            z:     Expoente da distância inversa (z > 0)

        Returns:
            Pesos normalizados (k,)
        """
        dists = np.where(dists == 0, 1e-10, dists)  # evita divisão por zero
        weights = 1.0 / (dists ** z)
        weights /= np.sum(weights)  # partição de unidade
        return weights