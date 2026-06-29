import numpy as np
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split as sk_train_test_split


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide aleatoriamente em treino e teste.

    Parâmetros
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    test_ratio : float
        Fração do conjunto para teste.
    random_state : int
        Semente para reprodutibilidade.
    stratify : Optional[np.ndarray]
        Se fornecido, faz split estratificado usando esses rótulos.
    """
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio deve estar entre 0 e 1")

    return sk_train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify
    )


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide aleatoriamente em treino, validação e teste.

    Parâmetros
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    val_ratio : float
        Fração do conjunto para validação.
    test_ratio : float
        Fração do conjunto para teste.
    random_state : int
        Semente para reprodutibilidade.
    stratify : Optional[np.ndarray]
        Se fornecido, faz split estratificado usando esses rótulos.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio e test_ratio devem ser >= 0 e soma < 1")

    X_temp, X_test, y_temp, y_test = sk_train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify
    )

    if val_ratio == 0:
        return X_temp, y_temp, np.empty((0, *X.shape[1:]), dtype=X.dtype), np.empty((0,), dtype=y.dtype), X_test, y_test

    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    stratify_temp = stratify
    if stratify is not None:
        _, stratify_temp = sk_train_test_split(
            stratify,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify
        )

    X_train, X_val, y_train, y_val = sk_train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide sequencialmente em treino e teste, mantendo ordem temporal.

    Útil para séries temporais e dados com dependência de ordem.
    """
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio deve estar entre 0 e 1")

    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def temporal_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide sequencialmente em treino, validação e teste, mantendo ordem temporal.

    Útil para séries temporais e dados com dependência de ordem.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio e test_ratio devem ser >= 0 e soma < 1")

    n = len(X)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = train_end + int(n * val_ratio)

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )