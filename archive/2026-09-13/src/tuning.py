"""Geração de combinações de hiperparâmetros a partir do search_space
declarativo de um modelo (ver src/models.py). Duas funções, sem
strategy pattern: grid_search (só parâmetros de escolha discreta) e
random_search (aceita também distribuições contínuas).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List

import numpy as np


def grid_search(search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Produto cartesiano de todos os parâmetros. Só aceita listas de
    valores discretos — parâmetros contínuos precisam de random_search."""
    names, value_lists = [], []
    for param, spec in search_space.items():
        if not isinstance(spec, list):
            raise ValueError(
                f"grid_search só aceita parâmetros com lista de valores. "
                f"'{param}' é uma distribuição contínua — use random_search.")
        names.append(param)
        value_lists.append(spec)
    return [dict(zip(names, values)) for values in itertools.product(*value_lists)]


def _sample_one(spec, rng: np.random.Generator):
    if isinstance(spec, list):
        return spec[rng.integers(0, len(spec))]
    kind, low, high = spec
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "loguniform":
        return float(np.exp(rng.uniform(math.log(low), math.log(high))))
    if kind == "randint":
        return int(rng.integers(low, high + 1))
    raise ValueError(f"Tipo de distribuição desconhecido: '{kind}'")


def random_search(search_space: Dict[str, Any], n_iter: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [
        {param: _sample_one(spec, rng) for param, spec in search_space.items()}
        for _ in range(n_iter)
    ]


def generate_combinations(strategy: str, search_space: Dict[str, Any], seed: int, n_iter: int = 20):
    if strategy == "grid_search":
        return grid_search(search_space)
    if strategy == "random_search":
        return random_search(search_space, n_iter=n_iter, seed=seed)
    raise ValueError(f"Estratégia de tuning desconhecida: '{strategy}'.")
