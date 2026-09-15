"""Validação cruzada: Stratified K-Fold (com ou sem repetição). Nested CV
não é oferecido de propósito — a separação treino/val (tuning) de teste
holdout (avaliação final) já cumpre o mesmo papel sem custo multiplicado.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def iterate_folds(X: np.ndarray, y: np.ndarray, strategy: str, n_splits: int,
                   seed: int, n_repeats: int = 1
                   ) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Produz (fold, repetição, X_tr, y_tr, X_val, y_val)."""
    if strategy == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif strategy == "repeated_stratified_kfold":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    else:
        raise ValueError(
            f"Estratégia de CV desconhecida: '{strategy}'. "
            "Disponíveis: 'stratified_kfold', 'repeated_stratified_kfold'.")

    for absolute_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        yield (absolute_idx % n_splits, absolute_idx // n_splits,
               X[train_idx], y[train_idx], X[val_idx], y[val_idx])
