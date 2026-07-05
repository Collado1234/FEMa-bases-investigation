"""Registry das estratégias de tuning disponíveis, selecionadas pela config."""
from __future__ import annotations

from tuning.base import TuningStrategy
from tuning.grid_search import GridSearch
from tuning.random_search import RandomSearch


def get_tuning_strategy(name: str, **kwargs) -> TuningStrategy:
    if name == "grid_search":
        return GridSearch()
    if name == "random_search":
        return RandomSearch(n_iter=kwargs.get("n_iter", 20))
    raise ValueError(f"Estratégia de tuning desconhecida: '{name}'")
