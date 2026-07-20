import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class LorentzianBasis(BaseBasis):
    """
    w_i = 1 / (1 + d_i^4), normalizada.

    NOTA: na literatura, "Lorentzian" costuma ser sinônimo de Cauchy
    (distribuição Cauchy-Lorentz), tipicamente 1/(1+d^2) com escala
    ajustável — não d^4 sem parâmetro. Vale decidir se essa base deveria
    ganhar um parâmetro de escala (ex.: 1/(1+(d/l)^4)) para deixar de ser
    um caso fixo e se diferenciar de fato do Cauchy Kernel. Corrigido
    aqui apenas o bug estrutural: a normalização usava `weights /=
    weights` (resultando em [1,1,...,1] em vez de partição da unidade).
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        dists = np.where(dists == 0, DELTA, dists)
        weights = 1 / (1 + dists ** 4)
        weights /= np.sum(weights)
        return weights
