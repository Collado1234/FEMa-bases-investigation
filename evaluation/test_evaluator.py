"""
Avaliacao final no conjunto de teste.

Chamado UMA UNICA VEZ, depois do retreino final (training/final_fit.py). O
conjunto de teste nunca e usado antes deste ponto - nem no grid/random
search, nem no CV, nem em nenhuma decisao de tuning.
"""

from typing import Any, Dict, List

import numpy as np

from metrics.classification import per_class_report
from metrics.registry import compute_all
from models.base import ModelPlugin

# Metricas cuja presenca em metric_names indica que este e' um experimento
# de CLASSIFICACAO (n_classes/per_class fazem sentido). Usado so' para
# decidir se calculamos o detalhamento por classe abaixo - nao interfere
# nas metricas agregadas, que continuam vindo de metrics.registry.
_CLASSIFICATION_METRIC_HINTS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"}


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

    results = compute_all(
        metric_names,
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        n_classes=n_classes,
    )

    if _CLASSIFICATION_METRIC_HINTS.intersection(metric_names):
        try:
            results["per_class"] = per_class_report(y_test, y_pred)
        except Exception:
            # nao interrompe a avaliacao agregada se o detalhamento por
            # classe falhar por algum motivo inesperado.
            results["per_class"] = None

    return results
