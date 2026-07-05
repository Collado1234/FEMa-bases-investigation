"""
evaluation/evaluator.py
-----------------------

Orquestrador central do sistema de avaliação.

Responsabilidades
-----------------
- Executar a inferência do modelo apenas uma vez;
- Manter cache das predições;
- Calcular apenas as métricas solicitadas;
- Calcular curvas quando solicitado;
- Gerar gráficos quando solicitado.

Este módulo NÃO implementa métricas, curvas ou gráficos.
Apenas coordena os respectivos módulos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics.registry import get_metric

from .curves.roc import compute_roc_curve
from .curves.precision_recall import compute_pr_curve

from .plots.classification import ClassificationPlots
from .plots.regression import RegressionPlots


class Evaluator:
    """
    Orquestrador da avaliação.

    Uma única inferência é realizada para cada modelo/dataset,
    reutilizando os resultados em todas as métricas, curvas e gráficos.
    """

    def __init__(
        self,
        model,
        X,
        y_true,
    ):

        self.model = model
        self.X = X
        self.y_true = np.asarray(y_true)

        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def infer(self):

        if "y_pred" in self._cache:
            return

        self._cache["y_pred"] = self.model.predict(self.X)

        if hasattr(self.model, "predict_proba"):

            try:
                self._cache["y_score"] = self.model.predict_proba(self.X)[:, 1]

            except Exception:

                self._cache["y_score"] = None

        else:

            self._cache["y_score"] = None

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def evaluate_metrics(
        self,
        metrics: list[str],
    ) -> dict:

        self.infer()

        results = {}

        y_pred = self._cache["y_pred"]
        y_score = self._cache["y_score"]

        for metric_name in metrics:

            metric = get_metric(metric_name)

            if metric_name == "auc_roc":

                results[metric_name] = metric(
                    self.y_true,
                    y_score,
                )

            else:

                results[metric_name] = metric(
                    self.y_true,
                    y_pred,
                )

        return results

    # ------------------------------------------------------------------
    # Curvas
    # ------------------------------------------------------------------

    def evaluate_curves(
        self,
        curves: list[str],
    ) -> dict:

        self.infer()

        y_score = self._cache["y_score"]

        if y_score is None:
            raise RuntimeError(
                "O modelo não fornece scores/probabilidades."
            )

        results = {}

        for curve in curves:

            if curve == "roc":

                results["roc"] = compute_roc_curve(
                    self.y_true,
                    y_score,
                )

            elif curve == "precision_recall":

                results["precision_recall"] = compute_pr_curve(
                    self.y_true,
                    y_score,
                )

            else:

                raise ValueError(
                    f"Curva '{curve}' desconhecida."
                )

        return results

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def evaluate_plots(
        self,
        plots: list[str],
        labels: list[str] | None = None,
        show: bool = True,
    ):

        curves = {}

        if "roc" in plots:

            curves["roc"] = self.evaluate_curves(
                ["roc"]
            )["roc"]

        if "precision_recall" in plots:

            curves["precision_recall"] = self.evaluate_curves(
                ["precision_recall"]
            )["precision_recall"]

        self.infer()

        y_pred = self._cache["y_pred"]

        for plot in plots:

            if plot == "roc":

                ClassificationPlots.roc_curve(
                    curves["roc"]["fpr"],
                    curves["roc"]["tpr"],
                    show=show,
                )

            elif plot == "precision_recall":

                ClassificationPlots.precision_recall_curve(
                    curves["precision_recall"]["recall"],
                    curves["precision_recall"]["precision"],
                    show=show,
                )

            elif plot == "confusion":

                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(
                    self.y_true,
                    y_pred,
                )

                ClassificationPlots.confusion_matrix(
                    cm,
                    labels=labels or ["0", "1"],
                    show=show,
                )

            else:

                raise ValueError(
                    f"Plot '{plot}' desconhecido."
                )

    # ------------------------------------------------------------------
    # API completa
    # ------------------------------------------------------------------

    def run(
        self,
        metrics: list[str] | None = None,
        curves: list[str] | None = None,
    ) -> dict:

        results = {}

        if metrics:

            results["metrics"] = self.evaluate_metrics(
                metrics
            )

        if curves:

            results["curves"] = self.evaluate_curves(
                curves
            )

        return results

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def clear_cache(self):

        self._cache.clear()