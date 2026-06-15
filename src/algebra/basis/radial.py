import numpy as np
from algebra.basis.basis import Basis
from algebra.distances.base_distance import BaseDistance
from algebra.neighboor_search.base_search import BaseSearch


class RadialBasis(Basis):
    """
    Base radial com função-mãe sino gaussiano:
        Ψ(r) = exp(-0.5 * (r / r0)²)

    Onde:
        r  = distância entre x e xi
        r0 = raio efetivo (parâmetro z)

    Não é interpoladora nem partição da unidade na forma pura.
    Ao normalizar os pesos, torna-se partição da unidade (base radial normalizada).
    """

    def __init__(self, distance: BaseDistance, search: BaseSearch):
        super().__init__(distance, search)

    @staticmethod
    def _mother_function(r: np.ndarray, r0: float) -> np.ndarray:
        """
        Função-mãe sino gaussiano.
        Considerada nula para r >= 4*r0 na prática.

        Args:
            r:  vetor de distâncias
            r0: raio efetivo (> 0)

        Returns:
            vetor com Ψ(r) para cada distância
        """
        return np.exp(-0.5 * (r / r0) ** 2)

    def predict(self, sample: np.ndarray, k: int, z: float) -> float:
        """
        Interpola o valor para uma amostra usando base radial normalizada.

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)
            z:      Raio efetivo r0 da função-mãe (z > 0)

        Returns:
            Valor interpolado.
        """
        dists = self.distance.compute(self.X_train, sample)
        indices = self.search.search(dists, k)

        k_dists = dists[indices]
        k_labels = self.y_train[indices]

        weights = self._mother_function(k_dists, r0=z)

        phi_sum = np.sum(weights)
        if phi_sum == 0:
            return float(np.mean(self.y_train))

        weights /= phi_sum  # base radial normalizada — partição da unidade

        predicted = np.dot(weights, k_labels)

        if np.isnan(predicted):
            return float(np.mean(self.y_train))

        return float(predicted)