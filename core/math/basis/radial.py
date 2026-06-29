import numpy as np
from .base_basis import BaseBasis
from ..neighboor_search import BaseSearch

class RadialBasis(BaseBasis):
    def __init__(self, search):
        """
        Base radial normalizada com função-mãe sino gaussiano:
            Ψ(r) = exp(-0.5 * (r / r0)²)

        Onde:
            r  = distância entre x e xi
            r0 = raio efetivo (parâmetro z)

        Normalizada, torna-se partição da unidade, mas não é interpoladora.
        """
        super().__init__(search)

    @staticmethod
    def _mother_function(r: np.ndarray, r0: float) -> np.ndarray:
        """
        Função-mãe sino gaussiano.
        Considerada nula para r >= 4*r0 na prática.

        Args:
            r:  vetor de distâncias (k,)
            r0: raio efetivo (> 0)

        Returns:
            vetor com Ψ(r) para cada distância (k,)
        """
        return np.exp(-0.5 * (r/r0)**2)

    def compute_weights(self, dists: np.ndarray, z:float) -> np.ndarray:
        """
        Calcula os pesos radiais normalizados a partir das distâncias.

        Args:
            dists: Distâncias dos k vizinhos (k,)
            z:     Raio efetivo r0 da função-mãe (z > 0)

        Returns:
            Pesos normalizados (k,)
        """
        weights = self._mother_function(dists, r0=z)

        phi_sum = np.sum(weights)
        if phi_sum == 0:
            return np.ones(len(dists)) / len(dists) #fallback uniforme
        
        weights /= phi_sum # particao de unidade
        return weights
