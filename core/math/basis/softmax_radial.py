import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class SoftmaxRadialBasis(BaseBasis):
    """
    w_i = exp(-beta * d_i), normalizada.

    NOTA: dentro do FEMa, que já normaliza tudo como partição da unidade,
    esta forma é matematicamente idêntica ao Laplacian Kernel (ver
    laplacian_kernel.py) com epsilon=beta — "softmax" aqui é só o nome
    do formato de peso exponencial-negativo, não uma propriedade
    distintiva adicional. Vale decidir com o orientador se uma das duas
    deve ganhar uma forma funcional diferente (ex.: temperatura separada
    de uma similaridade, não da distância bruta) para se justificar como
    base própria no estudo comparativo.
    """
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        weights = np.exp(-beta * dists)
        weights /= np.sum(weights)
        return weights
