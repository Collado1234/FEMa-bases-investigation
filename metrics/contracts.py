"""
Contrato de metrica.

Toda metrica registrada em metrics/registry.py deve ser uma funcao com a
assinatura:

    metric(y_true, y_pred, y_score, n_classes) -> float | None

- y_true: array de labels/valores verdadeiros
- y_pred: array de labels preditos (classe) ou valores preditos (regressao)
- y_score: matriz (n_amostras, n_classes) de probabilidades (classificacao)
  ou None se nao houver/nao suportado; ignorado por metricas de regressao
- n_classes: numero de classes distintas (classificacao); ignorado por
  metricas de regressao

Retornar None quando a metrica nao puder ser calculada (ex: roc_auc sem
y_score disponivel, ou so uma classe no fold) - o pipeline trata isso
registrando o valor como null no JSON, sem quebrar a execucao.
"""

from typing import Callable, Optional

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], int], Optional[float]]
