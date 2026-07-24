"""
Implementacoes concretas de metricas de regressao (via sklearn.metrics).

Seguem o mesmo contrato de metrics/contracts.py (y_true, y_pred, y_score,
n_classes), mas ignoram y_score e n_classes - existem apenas para que o
mesmo motor de avaliacao (metrics/registry.py::compute_all) sirva tanto
para tarefas de classificacao quanto de regressao.
"""

from typing import Optional

import numpy as np
from sklearn import metrics as skm


def mae(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.mean_absolute_error(y_true, y_pred))


def mse(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.r2_score(y_true, y_pred))


def mape(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.mean_absolute_percentage_error(y_true, y_pred) * 100)
