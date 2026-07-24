"""
Avaliacao por fold de CV.

Recebe um plugin, uma combinacao de hiperparametros e os indices de um
fold especifico. Treina no split de treino do fold, avalia no split de
validacao, e retorna as metricas calculadas - sem saber nada sobre
persistencia (isso e responsabilidade do pipeline/orquestrador).
"""

from typing import Any, Dict, List

import numpy as np

from metrics.registry import compute_all
from models.base import ModelPlugin
from training.trainer import train_estimator


def evaluate_fold(
    plugin: ModelPlugin,
    params: Dict[str, Any],
    X_train_fold,
    y_train_fold,
    X_val_fold,
    y_val_fold,
    metric_names: List[str],
    random_state: int,
) -> Dict[str, Any]:
    estimator = train_estimator(plugin, params, X_train_fold, y_train_fold, random_state)

    y_pred = plugin.predict(estimator, X_val_fold)
    y_score = plugin.predict_proba(estimator, X_val_fold)

    n_classes = int(len(np.unique(y_train_fold)))

    return compute_all(
        metric_names,
        y_true=y_val_fold,
        y_pred=y_pred,
        y_score=y_score,
        n_classes=n_classes,
    )
