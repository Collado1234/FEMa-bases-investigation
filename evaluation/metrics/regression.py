"""
evaluation/metrics/regression.py
---------------------------------
Métricas de regressão registradas como plugins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .registry import register_metric
from .common import _normalize


@register_metric("mae")
def mae(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


@register_metric("mse")
def mse(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


@register_metric("rmse")
def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


@register_metric("r2")
def r2(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / (ss_tot + 1e-12)


@register_metric("mape")
def mape(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)

    return float(
        np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    )