import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters


class EntropicBasis(BaseBasis):
    """phi(d) = exp(-beta * d^2); apos a normalizacao de compute_weights, w_i e' o softmax de -beta*d_i^2.

    CORRECAO: antes, evaluate() normalizava por conta propria (soma dos exp). Quando todos os exp davam
    underflow (beta grande ou distancias grandes, tipico em alta dimensao) a conta virava 0/0 e os pesos
    saiam NaN SEM AVISO (63 de 150 consultas com beta=50 em `digits`), fazendo o classificador escolher
    sempre a classe 0 nessas consultas. Agora evaluate() devolve o phi cru (regra da BaseBasis) e
    _log_phi() permite o softmax estavel em log-espaco.

    NOTA (equivalencia): apos normalizar, entropic(beta) == rbf_gaussian(epsilon=sqrt(beta))
    == radial(z=1/sqrt(2*beta)). Sao a MESMA familia gaussiana reparametrizada; diferencas de
    desempenho entre as tres vem das faixas de busca, nao da forma da funcao.
    """
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        return np.exp(-beta * np.asarray(dists, dtype=float) ** 2)

    def _log_phi(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]
        return -beta * np.asarray(dists, dtype=float) ** 2
