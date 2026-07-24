from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from ..neighboor_search import BaseSearch
from .parameters import BasisParameters

DELTA = 1e-10


class BaseBasis(ABC):
    """
    Classe base abstrata para bases de interpolação do FEMa.

    Duas camadas, propositalmente separadas:

    - evaluate(dists, params): valor CRU (não normalizado) da função de
      base phi(d). É o que entra na matriz do sistema linear A@lambda=y
      da interpolação RBF clássica (ver core/math/linear_system.py).
      Cada subclasse implementa só isso — é a fórmula da base em si,
      sem se preocupar com normalização.

    - compute_weights(dists, params): phi normalizado (partição da
      unidade) — w = phi(d) / sum(phi(d)) — usado pelo FEMa como peso de
      vizinhos. Implementado UMA ÚNICA VEZ aqui na classe base, em cima
      de evaluate(). Nenhuma subclasse deveria reimplementar isso: é
      justamente onde viviam os bugs de auto-divisão (`weights /=
      weights`) encontrados na revisão anterior — centralizar elimina
      essa classe inteira de bug por construção.

    Busca de vizinhos e cálculo de distâncias continuam delegados ao
    Search, injetado no construtor.
    """

    #: Nomes dos campos de BasisParameters que esta base efetivamente usa.
    #: Toda subclasse concreta deve sobrescrever isso. Permite validar,
    #: em uma única chamada (self._require), que os hiperparâmetros
    #: necessários foram fornecidos.
    PARAMS: Tuple[str, ...] = ()

    #: Campos que a base usa mas que são OPCIONAIS (None é um valor válido
    #: com significado próprio, ex.: h=None em bases de suporte compacto
    #: significa "raio automático" — não passam por self._require). Existe
    #: separado de PARAMS para que ferramentas externas (ex.: grade de
    #: busca de hiperparâmetros em models/fema_plugin.py) saibam que esses
    #: campos ainda valem a pena tunar, mesmo não sendo obrigatórios.
    OPTIONAL_PARAMS: Tuple[str, ...] = ()

    def __init__(self, search: BaseSearch):
        self.search = search

    @abstractmethod
    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        """
        Valor cru de phi(d), aplicado elemento a elemento em `dists`.

        Funciona tanto para um vetor 1D de distâncias de k vizinhos
        (uso do FEMa) quanto para uma matriz 2D de distâncias par-a-par
        n x n (uso do sistema linear de interpolação clássica) — todas
        as implementações são operações NumPy vetorizadas elemento a
        elemento, então o shape de entrada é preservado no shape de
        saída.
        """
        raise NotImplementedError

    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        """
        Pesos normalizados (partição da unidade): w = phi(d) / sum(phi(d)).

        Se sum(phi(d)) == 0 (ex.: base de suporte compacto onde nenhum
        vizinho cai dentro do raio), cai para pesos uniformes em vez de
        propagar um NaN silencioso.
        """
        phi = np.asarray(self.evaluate(dists, params), dtype=float)
        total = phi.sum()

        if total == 0:
            return np.full_like(phi, 1.0 / phi.size)

        return phi / total

    def _require(self, params: BasisParameters) -> Dict[str, float]:
        """
        Extrai de `params` os campos declarados em self.PARAMS e levanta
        um ValueError claro se algum estiver None (não configurado).
        """
        values: Dict[str, float] = {}
        missing = []

        for name in self.PARAMS:
            value = getattr(params, name)
            if value is None:
                missing.append(name)
            values[name] = value

        if missing:
            raise ValueError(
                f"{type(self).__name__} requer os parâmetros {self.PARAMS}, "
                f"mas {missing} não foram configurados em BasisParameters "
                f"(estão None)."
            )

        return values
