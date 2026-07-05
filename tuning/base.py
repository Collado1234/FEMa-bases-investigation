"""Strategy pattern: interface comum para qualquer estratégia de busca de
hiperparâmetros. O pipeline só chama generate_combinations(); nunca sabe
se é grid, random ou (no futuro) bayesiana."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class TuningStrategy(ABC):
    @abstractmethod
    def generate_combinations(self, search_space: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        """Recebe o search_space declarativo do modelo (BaseModel.get_search_space())
        e devolve uma lista de dicionários prontos para create_model()."""
        raise NotImplementedError
