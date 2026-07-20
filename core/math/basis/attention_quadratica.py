import numpy as np 
from core.math.basis import BaseBasis, DELTA
from .parameters import BasisParameters

class AttentionQuadraticBasis(BaseBasis):
    """
    w_i = 1 / (1 + d_i^2), normalizada.

    NOTA: esta é exatamente a forma do Cauchy Kernel com epsilon=1 fixo
    (ver cauchy_kernel.py) — confirmado empiricamente, os pesos saem
    idênticos. Hoje não existe parâmetro que distinga "attention" de
    "cauchy" no código; se a intenção é modelar atenção (softmax de um
    score de similaridade, não de distância), a fórmula precisa mudar.
    """
    PARAMS = ()
    def __init__(self, search):
        super().__init__(search)
    
    def compute_weights(self, dists:np.ndarray,  params: BasisParameters) -> np.ndarray:
        self._require(params)

        dists = np.where(dists == 0, DELTA, dists) 
        weights = 1 / (1 + dists ** 2)
        weights /= weights.sum() 
        return weights