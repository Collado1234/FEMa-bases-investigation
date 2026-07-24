"""
Motor generico de Grid Search.

Nao conhece nenhum algoritmo especifico: apenas expande o
parameter_grid() do plugin (dict de listas) em todas as combinacoes
possiveis (produto cartesiano). O pipeline cruza essas combinacoes com os
splits de CV.
"""

import itertools
from typing import Any, Dict, List

def expand_param_grid(param_grid: Dict[str,list]) -> List[Dict]:
    """Transforma {"C": [1, 10], "penalty": ["l2"]} em:
    [{"C": 1, "penalty": "l2"}, {"C": 10, "penalty": "l2"}]
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def combo_id(params: Dict[str, Any]) -> str:
    """Gera um identificador estavel e legivel para uma combinacao de
    hiperparametros, usado como chave de checkpoint e no ranking do
    summary."""
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return ",".join(parts)

