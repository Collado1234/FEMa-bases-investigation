import numpy as np
from .base_basis import BaseBasis, DELTA
from .parameters import BasisParameters

class LorentzianBasis(BaseBasis):
    """
    phi(d) = 1 / (1 + d^4).

    ATENÇÃO AO NOME: na literatura, "kernel Lorentziano" costuma ser
    sinônimo do kernel de Cauchy, 1/(1+(epsilon*d)^2) — que já existe
    neste pacote como CauchyKernelBasis (e como AttentionQuadraticBasis
    com epsilon=1 fixo). Esta base, com expoente 4 em vez de 2, NÃO é
    essa forma padrão; é mais próxima de um kernel "bi-quadrático" tipo
    Tukey biweight adaptado. Mantida com o nome "lorentzian" (chave de
    registro em factory_basis.py) por compatibilidade com experimentos
    e configs já existentes — mas ao reportar resultados, não a rotule
    como o kernel de Lorentz/Cauchy da literatura. Se for reorganizar o
    conjunto de bases, considere renomear para algo como
    "QuarticCauchyBasis" nessa limpeza.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        self._require(params)
        dists = np.where(dists == 0, DELTA, dists)
        return 1 / (1 + dists ** 4)
