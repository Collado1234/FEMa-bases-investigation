"""
evaluation/plots/regression.py
------------------------------

Gráficos para problemas de regressão.

Este módulo é responsável apenas pela visualização dos resultados.
Não calcula métricas, resíduos nem realiza inferência.
"""

from __future__ import annotations

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .common import (
    create_figure,
    configure_axes,
    export_figure,
)

from .styles import COLORS


class RegressionPlots:
    """
    Conjunto de gráficos para regressão.
    """

    # ------------------------------------------------------------------
    # Actual vs Predicted
    # ------------------------------------------------------------------

    @staticmethod
    def actual_vs_predicted(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        ax.scatter(
            y_true,
            y_pred,
            color=COLORS.primary,
            alpha=0.75,
        )

        minimum = min(np.min(y_true), np.min(y_pred))
        maximum = max(np.max(y_true), np.max(y_pred))

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            "--",
            color=COLORS.diagonal,
            label="Ideal",
        )

        configure_axes(
            ax,
            title="Actual vs Predicted",
            xlabel="Actual",
            ylabel="Predicted",
            legend=True,
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    @staticmethod
    def residuals(
        y_pred: np.ndarray,
        residuals: np.ndarray,
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        ax.scatter(
            y_pred,
            residuals,
            color=COLORS.secondary,
            alpha=0.75,
        )

        ax.axhline(
            0,
            linestyle="--",
            color=COLORS.diagonal,
        )

        configure_axes(
            ax,
            title="Residual Plot",
            xlabel="Predicted",
            ylabel="Residual",
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax

    # ------------------------------------------------------------------
    # Residual Histogram
    # ------------------------------------------------------------------

    @staticmethod
    def residual_histogram(
        residuals: np.ndarray,
        bins: int = 30,
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        ax.hist(
            residuals,
            bins=bins,
            color=COLORS.primary,
            alpha=0.8,
        )

        configure_axes(
            ax,
            title="Residual Distribution",
            xlabel="Residual",
            ylabel="Frequency",
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax

    # ------------------------------------------------------------------
    # Error Distribution
    # ------------------------------------------------------------------

    @staticmethod
    def error_distribution(
        errors: np.ndarray,
        bins: int = 30,
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        ax.hist(
            errors,
            bins=bins,
            color=COLORS.danger,
            alpha=0.8,
        )

        configure_axes(
            ax,
            title="Prediction Error Distribution",
            xlabel="Prediction Error",
            ylabel="Frequency",
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax