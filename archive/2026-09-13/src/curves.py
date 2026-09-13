"""Curvas ROC e precision-recall para classificação binária (cálculo manual,
sem dependências além de numpy). Usadas opcionalmente por src/plots.py."""
from __future__ import annotations

import numpy as np


def _check_binary(y: np.ndarray) -> None:
    if len(np.unique(y)) > 2:
        raise ValueError("Curvas só suportam classificação binária.")


def roc_curve(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    _check_binary(y_true)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    fpr = np.concatenate(([0.0], fp / N if N else fp * 0.0, [1.0]))
    tpr = np.concatenate(([0.0], tp / P if P else tp * 0.0, [1.0]))
    return {"fpr": fpr, "tpr": tpr}


def precision_recall_curve(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    _check_binary(y_true)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    P = np.sum(y_true == 1)

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / P if P else tp * 0.0
    return {"precision": precision, "recall": recall}
