import numpy as np
from .base_basis import BaseBasis
from ..neighboor_search import BaseSearch

class ShepardBasis(BaseBasis):
    """
    Base de Shepard para interpolação baseada em vizinhos.

    Esta classe atribui pesos inversamente proporcionais às distâncias de
    um ponto de avaliação em relação aos seus vizinhos. Os pesos são
    calculados como

        w_i = 1 / d_i^z

    e, em seguida, normalizados para formar uma partição da unidade:

        w_i = w_i / sum_j w_j

    Com essa normalização, a base é interpoladora e satisfaz a propriedade
    de partição da unidade, o que a torna adequada para métodos de
    aproximação e reconstrução espacial.
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
