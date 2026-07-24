"""
Motor de Random Search.

Complementa tuning/grid_search.py: aceita tambem distribuicoes continuas
no formato ("uniform"|"loguniform", low, high) e ("randint", low, high),
alem de listas de escolha discreta. Necessario porque alguns modelos deste
projeto (FEMa: z/epsilon/c/...; logreg: C; mlp: learning_rate, dropout) tem
hiperparametros continuos que um grid discreto representaria mal.

Suporta os mesmos dois formatos de espaco de busca que grid_search.py:
Dict[str, spec] (incondicional) ou List[Dict[str, spec]] (condicional,
uma ramificacao por dict — ver models/fema.py). No caso condicional,
cada amostra sorteia primeiro QUAL ramificacao usar (com peso igual entre
elas) e so entao amostra os campos DAQUELA ramificacao — nunca mistura
campos de ramificacoes diferentes numa mesma amostra.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Union

import numpy as np

SearchSpace = Union[Dict[str, Any], List[Dict[str, Any]]]


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
    raise ValueError(f"Tipo de distribuicao desconhecido: '{kind}'")


def _sample_branch(branch: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    return {param: _sample_one(spec, rng) for param, spec in branch.items()}


def sample_param_space(search_space: SearchSpace, n_iter: int, seed: int) -> List[Dict[str, Any]]:
    """Amostra `n_iter` combinacoes de hiperparametros do espaco de busca,
    de forma deterministica dada a `seed`."""
    rng = np.random.default_rng(seed)

    if isinstance(search_space, list):
        branches = search_space
        return [_sample_branch(branches[rng.integers(0, len(branches))], rng) for _ in range(n_iter)]

    return [_sample_branch(search_space, rng) for _ in range(n_iter)]
