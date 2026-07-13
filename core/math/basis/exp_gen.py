import numpy as np
from core.math.basis import BaseBasis

class ExponentialGenBasis(BaseBasis):

    def __init__(self, search):
        super().__init__(search)
    

    def compute_weights(self, dists:np.ndarray, epsilon:float, p:float = 2.0):
        dists = np.where(dists == 0, 1e-10, dists)

        weights = np.exp(-epsilon*(dists**p))

        weights /= np.sum(weights) # particao de unidade

        return weights