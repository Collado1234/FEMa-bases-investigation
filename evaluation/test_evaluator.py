"""
Avaliacao final no conjunto de teste.

Chamado UMA UNICA VEZ, depois do retreino final (training/final_fit.py). O
conjunto de teste nunca e usado antes deste ponto - nem no grid/random
search, nem no CV, nem em nenhuma decisao de tuning.
"""

from typing import Any, Dict, List

import numpy as np

from metrics.registry import compute_all
from models.base import ModelPlugin


def evaluate_on_test(
    plugin: ModelPlugin,
    estimator,
    X_test,
    y_test,
    y_train_reference,
    metric_names: List[str],
) -> Dict[str, Any]:
    y_pred = plugin.predict(estimator, X_test)
    y_score = plugin.predict_proba(estimator, X_test)

    n_classes = int(len(np.unique(y_train_reference)))

    return compute_all(
        metric_names,
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        n_classes=n_classes,
    )
