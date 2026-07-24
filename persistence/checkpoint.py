"""
Sistema de checkpoint.

Antes de iniciar o Grid/Random Search, varre os run_*.json ja existentes
no diretorio de resultados do experimento e reconstroi o conjunto de
combinacoes (hiperparametros, fold, repeticao) ja executadas. O motor do
pipeline (pipeline/run_model.py) usa isso para pular o que ja foi feito e
continuar exatamente de onde parou.

A chave de identificacao de uma combinacao e (combo_id, repeat_idx,
fold_idx) - combo_id vem de tuning/grid_search.py::combo_id(), garantindo
que a MESMA combinacao de hiperparametros sempre gere a MESMA chave,
independente da ordem de iteracao.
"""

from typing import Set, Tuple

from persistence.run_writer import load_all_runs


def get_completed_keys(model_name: str, experiment_name: str) -> Set[Tuple[str, int, int]]:
    runs = load_all_runs(model_name, experiment_name)
    completed = set()
    for run in runs:
        key = (run["combo_id"], run["repeat_idx"], run["fold_idx"])
        completed.add(key)
    return completed


def is_done(completed_keys: Set[Tuple[str, int, int]], combo_id: str, repeat_idx: int, fold_idx: int) -> bool:
    return (combo_id, repeat_idx, fold_idx) in completed_keys
