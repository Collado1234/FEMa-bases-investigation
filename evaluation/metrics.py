"""
fema/metrics.py
---------------
Métricas de avaliação para classificação binária no contexto do projeto FEMa.

Este módulo foi adaptado para suportar a validação de modelos de
classificação binária usados nas investigações de bases FEMa, com foco
em análise de desempenho, limiarização e avaliação de probabilidades.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict
import numpy as np
from numpy.typing import NDArray

MetricDict = TypedDict(
    "MetricDict",
    {
        "accuracy": float,
        "precision": float,
        "recall": float,
        "specificity": float,
        "balanced_accuracy": float,
        "f1": float,
        "f2": float,
        "mcc": float,
        "auc_roc": Optional[float],
        "TP": int,
        "TN": int,
        "FP": int,
        "FN": int,
    },
)

Strategy = Literal["predict", "predict_proba", "decision_function"]


def _confusion(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> tuple[int, int, int, int]:
    """Retorna (TP, TN, FP, FN) para classificação binária."""
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    return TP, TN, FP, FN


def _auc_roc(y_true: NDArray[np.int64], y_prob: NDArray[np.floating]) -> float:
    """Calcula AUC-ROC pelo método trapezoidal."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    pos = max(np.sum(y_true == 1), 1)
    neg = max(np.sum(y_true == 0), 1)
    tpr = np.concatenate([[0.0], np.cumsum(y_sorted == 1) / pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(y_sorted == 0) / neg, [1.0]])
    return float(np.trapz(tpr, fpr))


def _normalize_array(arr: NDArray) -> NDArray:
    return np.asarray(arr).ravel()


def compute_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_prob: Optional[NDArray] = None,
    beta: float = 2.0,
) -> MetricDict:
    """
    Calcula as métricas de classificação binária mais usadas em FEMa.

    Parâmetros
    ----------
    y_true : NDArray
        Rótulos verdadeiros binários (0 ou 1).
    y_pred : NDArray
        Predições binárias (0 ou 1).
    y_prob : NDArray | None
        Probabilidades da classe positiva, usadas para AUC-ROC.
    beta : float
        Exponente do F-beta.
    """
    y_true = _normalize_array(y_true).astype(int)
    y_pred = _normalize_array(y_pred).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")

    TP, TN, FP, FN = _confusion(y_true, y_pred)
    n = len(y_true)

    accuracy = (TP + TN) / n if n > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    b2 = beta ** 2
    fbeta = (
        (1 + b2) * precision * recall / (b2 * precision + recall)
        if (b2 * precision + recall) > 0 else 0.0
    )

    denom_mcc = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = float((TP * TN - FP * FN) / denom_mcc) if denom_mcc > 0 else 0.0

    auc = None
    if y_prob is not None:
        y_prob = _normalize_array(y_prob).astype(float)
        if len(y_prob) == len(y_true):
            auc = _auc_roc(y_true, y_prob)

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "f1": round(f1, 6),
        "f2": round(fbeta, 6),
        "mcc": round(mcc, 6),
        "auc_roc": round(auc, 6) if auc is not None else None,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def evaluate(
    model,
    X: NDArray,
    y_true: NDArray,
    strategy: Strategy = "predict",
    threshold: float = 0.5,
    beta: float = 2.0,
) -> dict:
    """
    Avalia um modelo no conjunto (X, y_true) usando a estratégia escolhida.

    Estratégias suportadas:
      - "predict"           : usa model.predict()
      - "predict_proba"     : usa probabilidade da classe positiva
      - "decision_function" : usa função de decisão se disponível
    """
    y_true = _normalize_array(y_true).astype(int)

    if strategy == "predict_proba":
        if not hasattr(model, "predict_proba"):
            raise AttributeError("modelo não implementa predict_proba")
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    elif strategy == "decision_function":
        if not hasattr(model, "decision_function"):
            raise AttributeError("modelo não implementa decision_function")
        scores = model.decision_function(X)
        y_prob = np.asarray(scores).astype(float)
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = _normalize_array(model.predict(X)).astype(int)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)[:, 1]
            except Exception:
                y_prob = None

    metrics = compute_metrics(y_true, y_pred, y_prob, beta)
    metrics["strategy"] = strategy
    metrics["threshold"] = threshold if strategy in {"predict_proba", "decision_function"} else None
    return metrics