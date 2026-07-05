import numpy as np
from .base_basis import BaseBasis

class RbfGaussianBasis(BaseBasis):
    """
    Base radial gaussiana para interpolação baseada em vizinhos.

    Esta classe atribui pesos baseados na função gaussiana das distâncias de
    um ponto de avaliação em relação aos seus vizinhos. Os pesos são
    calculados como

        w_i = exp(-(epsilon*dists)^2)

    e, em seguida, normalizados para formar uma partição da unidade:

        w_i = w_i / sum_j w_j
    """
    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray, epsilon:float) -> np.ndarray:
        dists = np.where(dists == 0, 1e-10, dists)

        weights = np.exp(-(epsilon*dists)**2)

        weights /= np.sum(weights) # particao de unidade

        return weights
        