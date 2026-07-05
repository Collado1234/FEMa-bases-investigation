"""Random search: amostra n_iter combinações do search_space declarativo,
suportando choice / uniform / loguniform / randint."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from tuning.base import TuningStrategy


class RandomSearch(TuningStrategy):
    def __init__(self, n_iter: int = 20):
        self.n_iter = n_iter

    def generate_combinations(self, search_space: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        combinations = []
        for _ in range(self.n_iter):
            combo: Dict[str, Any] = {}
            for param, spec in search_space.items():
                combo[param] = self._sample_one(spec, rng)
            combinations.append(combo)
        return combinations

    @staticmethod
    def _sample_one(spec: Dict[str, Any], rng: np.random.Generator):
        kind = spec["type"]
        if kind == "choice":
            values = spec["values"]
            idx = rng.integers(0, len(values))
            return values[idx]
        if kind == "uniform":
            return float(rng.uniform(spec["low"], spec["high"]))
        if kind == "loguniform":
            low, high = math.log(spec["low"]), math.log(spec["high"])
            return float(math.exp(rng.uniform(low, high)))
        if kind == "randint":
            return int(rng.integers(spec["low"], spec["high"] + 1))
        raise ValueError(f"Tipo de parâmetro desconhecido: '{kind}'")
