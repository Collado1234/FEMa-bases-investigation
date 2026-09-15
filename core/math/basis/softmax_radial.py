import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class SoftmaxRadialBasis(BaseBasis):
    """
    phi(d) = exp(-beta * d).

    Depois de normalizada por BaseBasis.compute_weights() (partição da
    unidade: w_i = phi(d_i) / sum_j phi(d_j)), isso vira exatamente o
    softmax padrão sobre as distâncias negadas:

        w_i = exp(-beta*d_i) / sum_j exp(-beta*d_j)

    Matematicamente idêntica ao Laplacian Kernel com epsilon=beta depois
    de normalizada (ver laplacian_kernel.py) — a diferença de nome
    existe porque aqui o enquadramento pretendido é "softmax sobre
    distâncias" (útil ao comparar com kernels de atenção), enquanto
    Laplacian é o kernel de SVM clássico (Genton, 2001) antes de
    qualquer normalização.
    """
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        return np.exp(-beta * dists)
