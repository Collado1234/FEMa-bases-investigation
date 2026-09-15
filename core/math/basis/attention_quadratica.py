import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters

class AttentionQuadraticBasis(BaseBasis):
    """
    phi(d) = 1 / (1 + d^2).

    Idêntica ao Cauchy Kernel com epsilon=1 fixo (ver cauchy_kernel.py) —
    mantida como base separada porque não expõe epsilon como
    hiperparâmetro tunável (é, por design, a variante "sem escala").

    NOTA DE BUG CORRIGIDO: esta base normalizava por linha
    (`np.sum(..., axis=1)`) dentro do próprio evaluate(). Como o
    pipeline real (FEMaClassifier/FEMaRegressor) sempre passa `dists`
    como vetor 1D de distâncias dos k vizinhos de UM ponto de consulta
    (nunca uma matriz 2D par-a-par), `axis=1` não existe nesse vetor —
    a base quebrava em runtime com AxisError. Além disso, mesmo se o
    shape fosse 2D, normalizar aqui duplicaria a normalização já feita
    por BaseBasis.compute_weights(). evaluate() agora devolve só o phi(d)
    cru; a normalização (partição da unidade) é feita uma única vez,
    na classe base.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        return 1.0 / (1.0 + dists ** 2)
