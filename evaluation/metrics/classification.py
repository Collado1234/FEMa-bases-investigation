"""
evaluation/metrics/classification.py
-------------------------------------
Métricas de classificação registradas como plugins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from .registry import register_metric
from .common import _normalize


# -------------------------
# Core helpers
# -------------------------

def _confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


# -------------------------
# Metrics (plugins)
# -------------------------

@register_metric("accuracy")
def accuracy(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)
    return float(np.mean(y_true == y_pred))


@register_metric("precision")
def precision(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)

    tp, _, fp, _ = _confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


@register_metric("recall")
def recall(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)

    tp, _, _, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


@register_metric("f1")
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


@register_metric("mcc")
def mcc(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)

    tp, tn, fp, fn = _confusion(y_true, y_pred)

    denom = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    return (tp * tn - fp * fn) / denom if denom else 0.0


@register_metric("auc_roc")
def auc_roc(y_true, y_score):
    y_true = _normalize(y_true)
    y_score = _normalize(y_score)

    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))