"""Métricas de classificação e regressão. Em vez de reimplementar cada
métrica na mão, usamos sklearn.metrics diretamente (já é dependência do
projeto) — mais simples e correto em multiclasse. METRICS mapeia nome
(usado na config YAML) -> função (y_true, y_pred_ou_score) -> float.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def _auc_roc(y_true, y_score):
    """AUC-ROC: None quando não há score ou só existe uma classe em y_true."""
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    multi_class = "ovr" if len(np.unique(y_true)) > 2 else "raise"
    return float(roc_auc_score(y_true, y_score, multi_class=multi_class))


# nome (config YAML) -> função. average="macro" trata multiclasse dando o
# mesmo peso a todas as classes, independente do desbalanceamento.
METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
    "recall": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
    "f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    "balanced_accuracy": balanced_accuracy_score,
    "mcc": matthews_corrcoef,
    "auc_roc": _auc_roc,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
    "r2": r2_score,
    "mape": lambda yt, yp: float(mean_absolute_percentage_error(yt, yp) * 100),
}

# métricas que recebem score/probabilidade em vez de rótulo predito
SCORE_METRICS = {"auc_roc"}


def compute_metrics(names: list[str], y_true, y_pred, y_score=None) -> dict:
    results = {}
    for name in names:
        if name not in METRICS:
            raise KeyError(f"Métrica '{name}' desconhecida. Disponíveis: {sorted(METRICS)}")
        try:
            results[name] = METRICS[name](y_true, y_score if name in SCORE_METRICS else y_pred)
        except ValueError:
            results[name] = None  # ex.: auc_roc sem probabilidade adequada
    return results
