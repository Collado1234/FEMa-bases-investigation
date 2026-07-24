"""
Treinamento generico.

Responsabilidade unica: dado um plugin de modelo, um conjunto de
hiperparametros e dados de treino (e, opcionalmente, de validacao),
instanciar e treinar o estimator. Nao sabe nada sobre CV, grid/random
search, metricas ou persistencia.
"""

from typing import Any, Dict, Optional

from models.base import ModelPlugin


def train_estimator(
    plugin: ModelPlugin,
    params: Dict[str, Any],
    X_train,
    y_train,
    random_state: int,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
):
    estimator = plugin.create_model(params, random_state=random_state)
    estimator = plugin.fit(estimator, X_train, y_train, X_val=X_val, y_val=y_val)
    return estimator
