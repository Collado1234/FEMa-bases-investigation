"""
Motor generico de Grid Search.

Nao conhece nenhum algoritmo especifico: apenas expande o
parameter_grid() do plugin em todas as combinacoes possiveis.

Suporta dois formatos de parameter_grid():

    - Dict[str, list]: espaco INCONDICIONAL classico, onde todo campo e'
      um eixo independente (ex.: logreg, mlp) -> produto cartesiano unico
      de tudo, como sempre foi.

    - List[Dict[str, list]]: espaco CONDICIONAL, uma "ramificacao" por
      dict -> produto cartesiano DENTRO de cada dict, resultados
      concatenados ENTRE ramificacoes (sem produto cruzado entre elas).
      Necessario quando um hiperparametro so existe para certos valores
      de outro — caso do FEMa, onde 'epsilon' so faz sentido quando
      basis_function='rbf_gaussian', 'alpha'+'l' so quando
      basis_function='rational_quadratic', etc. Um produto cartesiano
      unico sobre todos os campos de todas as bases geraria uma explosao
      combinatoria de combinacoes redundantes (mesma base+k repetida uma
      vez para cada valor irrelevante de parametro de OUTRA base) — ver
      models/fema.py para como isso e montado.
"""

import itertools
from typing import Any, Dict, List, Union

ParamGrid = Union[Dict[str, list], List[Dict[str, list]]]


def _expand_single_grid(param_grid: Dict[str, list]) -> List[Dict]:
    """Produto cartesiano de um único dict {chave: [valores]} (caso incondicional)."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def expand_param_grid(param_grid: ParamGrid) -> List[Dict]:
    """Transforma {"C": [1, 10], "penalty": ["l2"]} em:
    [{"C": 1, "penalty": "l2"}, {"C": 10, "penalty": "l2"}]

    Ou, se param_grid for uma lista de dicts (espaço condicional),
    expande cada um separadamente e concatena os resultados — nunca faz
    produto cruzado ENTRE ramificações diferentes.
    """
    if isinstance(param_grid, list):
        combos: List[Dict] = []
        for branch in param_grid:
            combos.extend(_expand_single_grid(branch))
        return combos

    return _expand_single_grid(param_grid)


def combo_id(params: Dict[str, Any]) -> str:
    """Gera um identificador estavel e legivel para uma combinacao de
    hiperparametros, usado como chave de checkpoint e no ranking do
    summary."""
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return ",".join(parts)
