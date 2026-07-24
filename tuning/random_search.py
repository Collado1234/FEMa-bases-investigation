"""
Motor de Random Search.

Complementa tuning/grid_search.py: aceita tambem distribuicoes continuas
no formato ("uniform"|"loguniform", low, high) e ("randint", low, high),
alem de listas de escolha discreta. Necessario porque alguns modelos deste
projeto (FEMa: z; logreg: C; mlp: learning_rate, dropout) tem
hiperparametros continuos que um grid discreto representaria mal.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


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


def sample_param_space(search_space: Dict[str, Any], n_iter: int, seed: int) -> List[Dict[str, Any]]:
    """Amostra `n_iter` combinacoes de hiperparametros do espaco de busca,
    de forma deterministica dada a `seed`."""
    rng = np.random.default_rng(seed)
    return [
        {param: _sample_one(spec, rng) for param, spec in search_space.items()}
        for _ in range(n_iter)
    ]
