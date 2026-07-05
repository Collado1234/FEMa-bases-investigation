"""
Grid search. Só aceita parâmetros do tipo "choice" (valores discretos
explícitos) — parâmetros contínuos (uniform/loguniform) precisam ser
discretizados na própria config antes de usar grid, ou usar random_search.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List

from tuning.base import TuningStrategy


class GridSearch(TuningStrategy):
    def generate_combinations(self, search_space: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        names = []
        value_lists = []
        for param, spec in search_space.items():
            if spec["type"] != "choice":
                raise ValueError(
                    f"GridSearch só aceita parâmetros 'choice'. "
                    f"'{param}' é '{spec['type']}' — use RandomSearch ou "
                    f"defina uma lista discreta de valores no YAML."
                )
            names.append(param)
            value_lists.append(spec["values"])

        combinations = []
        for values in itertools.product(*value_lists):
            combinations.append(dict(zip(names, values)))
        return combinations
