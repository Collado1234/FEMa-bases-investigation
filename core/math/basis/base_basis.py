from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from ..neighboor_search import BaseSearch
from .parameters import BasisParameters

DELTA = 1e-10

class BaseBasis(ABC):
    """
    Classe base abstrata para bases de interpolação do FEMa.

    Basis é puramente matemática — não guarda dados de treino.
    Recebe distâncias prontas e um objeto BasisParameters ("classe
    coringa" com todos os hiperparâmetros possíveis) e devolve pesos
    normalizados.

    A busca de vizinhos e o cálculo de distâncias são delegados
    ao Search, que é injetado no construtor.
    """

    #: Nomes dos campos de BasisParameters que esta base efetivamente usa.
    #: Toda subclasse concreta deve sobrescrever isso. É o que permite
    #: validar, em uma única chamada (self._require), que os
    #: hiperparâmetros necessários foram fornecidos — sem que esta base
    #: precise saber nada sobre as demais, e sem que quem chama precise
    #: saber a fórmula de cada base para montar os argumentos certos.
    PARAMS: Tuple[str, ...] = ()

    def __init__(self, search: BaseSearch):
        self.search = search

    @abstractmethod
    def compute_weights(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        """
        Calcula os pesos a partir das distâncias dos k vizinhos.

        Args:
            dists:  Distâncias dos k vizinhos (k,)
            params: Hiperparâmetros da base (ver BasisParameters).
                    Cada base lê apenas os campos listados em self.PARAMS.

        Returns:
            weights: Pesos normalizados (k,)
        """
        raise NotImplementedError

    def _require(self, params: BasisParameters) -> Dict[str, float]:
        """
        Extrai de `params` os campos declarados em self.PARAMS e levanta
        um ValueError claro se algum estiver None (não configurado).

        Uso típico no início de compute_weights:

            values = self._require(params)
            epsilon = values["epsilon"]
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
