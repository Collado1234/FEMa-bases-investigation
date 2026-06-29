import numpy as np
from .base_basis import BaseBasis
from ..neighboor_search import BaseSearch

class SheppardBasis(BaseBasis):
    """
    Base de Shepard para interpolação.

    Utiliza pesos inversamente proporcionais à distância:
        w_i = 1 / d_i^z

    Normalizado para partição da unidade:
        w_i = w_i / Σ w_j

    É interpoladora e partição da unidade quando normalizada.
    """
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray, z:float) -> np.ndarray:
        """
        Calcula os pesos de Shepard a partir das distâncias.

        Args:
            dists: Distâncias dos k vizinhos (k,)
            z:     Expoente da distância inversa (z > 0)

        Returns:
            Pesos normalizados (k,)
        """
        dists = np.where(dists == 0, 1e-10, dists) #evita divisao por zero
        weights = 1.0 / (dists**z)
        weights /= np.sum(weights) # particao de unidade
        return weights
