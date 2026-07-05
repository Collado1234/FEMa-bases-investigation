"""
Estratégias de validação cruzada. Strategy pattern: a config escolhe qual
usar; training/trainer.py apenas itera sobre o que get_cv_splitter retorna,
sem saber qual estratégia está por trás.

Padrão do framework: Repeated Stratified K-Fold (ver arquitetura.md,
seção 1, para a justificativa estatística). Nested CV não é oferecido
aqui de propósito — o design já separa treino/val (tuning) de teste
holdout (evaluation), cumprindo o mesmo papel sem o custo multiplicado.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def get_cv_splitter(strategy: str, n_splits: int, seed: int, n_repeats: int = 1):
    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    raise ValueError(
        f"Estratégia de CV desconhecida: '{strategy}'. "
        "Disponíveis: 'stratified_kfold', 'repeated_stratified_kfold'."
    )


def iterate_folds(
    splitter, X: np.ndarray, y: np.ndarray, n_splits: int
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Produz (fold_index, repetition_index, X_tr, y_tr, X_va, y_va).
    Funciona tanto para StratifiedKFold (1 repetição) quanto para
    RepeatedStratifiedKFold (várias), calculando o índice de repetição a
    partir da posição absoluta do split.
    """
    for absolute_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        fold = absolute_idx % n_splits
        repetition = absolute_idx // n_splits
        yield (
            fold,
            repetition,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
        )
