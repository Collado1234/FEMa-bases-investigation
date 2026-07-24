"""
Ponto unico de entrada para expandir o espaco de hiperparametros de um
plugin em uma lista de combinacoes concretas, de acordo com a
`tuning_strategy` do experimento ("grid_search" ou "random_search").

O pipeline (pipeline/run_model.py) so conhece esta funcao - nao decide
sozinho entre grid_search.py e random_search.py.
"""
from __future__ import annotations

from typing import Any, Dict, List

from models.base import ModelPlugin
from tuning.grid_search import expand_param_grid
from tuning.random_search import sample_param_space

VALID_STRATEGIES = {"grid_search", "random_search"}

def generate_combinations(
        plugin: ModelPlugin,
        strategy: str,
        seed: int,
        n_iter: int = 20,
        fixed_hyperparameters: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Gera as combinacoes de hiperparametros a avaliar para `plugin`.

    Se `fixed_hyperparameters` for informado, ignora a estrategia de tuning
    e usa exatamente essa combinacao (util para reproduzir um run especifico
    ou rodar sem busca de hiperparametros).
    """
    if fixed_hyperparameters is not None:
        return [fixed_hyperparameters]

    if strategy == "grid_search":
        return expand_param_grid(plugin.parameter_grid())
    if strategy == "random_search":
        return sample_param_space(plugin.random_search_space(), n_iter=n_iter, seed=seed)
    raise ValueError(f"Estrategia de tuning desconhecida: '{strategy}'. Disponiveis: {VALID_STRATEGIES}")
