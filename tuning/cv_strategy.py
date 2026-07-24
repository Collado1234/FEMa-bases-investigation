"""
Estrategia de validacao cruzada.

Suporta "stratified_kfold" (uma unica passada) e "repeated_stratified_kfold"
(repete com seeds diferentes para reduzir a variancia da estimativa - util
com datasets pequenos ou desbalanceados). Nested CV nao e oferecido de
proposito: a separacao treino/val (tuning, via estes splits) de teste
holdout (avaliacao final, ver evaluation/test_evaluator.py) ja cumpre o
mesmo papel sem custo multiplicado.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from utils.seeding import derive_seed

VALID_STRATEGIES = {"stratified_kfold", "repeated_stratified_kfold"}

def build_cv_splits( X, y, strategy: str, n_splits: int, n_repeats: int,
                    master_seed: int) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Gera splits (repeat_idx, fold_idx, train_idx, val_idx).

    A seed do splitter e derivada da seed mestre, entao o MESMO conjunto de
    splits e gerado sempre que o pipeline for reexecutado com a mesma
    config - condicao necessaria para o checkpoint funcionar corretamente.
    """
    cv_seed = derive_seed(master_seed, "cv_strategy", strategy)

    if strategy == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state= cv_seed)
        effective_repeats = 1
    elif strategy == "repeated_stratified_kfold":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
        effective_repeats = n_repeats
    else:
        raise ValueError(f"Estrategia de CV desconhecida: '{strategy}'. Disponiveis: {VALID_STRATEGIES}")

    for split_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        repeat_idx = split_idx // n_splits
        fold_idx = split_idx % n_splits
        yield repeat_idx, fold_idx, train_idx, val_idx
