"""
evaluation/plots/classification.py
----------------------------------

Gráficos para classificação binária.

Este módulo não calcula métricas nem curvas.
Ele apenas recebe os dados e os desenha.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .common import (
    create_figure,
    configure_axes,
    export_figure,
)

from .styles import COLORS


class ClassificationPlots:
    """
    Conjunto de gráficos para classificação.
    """

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------
    @staticmethod
    def confusion_matrix(
        cm: np.ndarray,
        labels: list[str],
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        im = ax.imshow(
            cm,
            cmap=COLORS.confusion_cmap,
        )

        plt.colorbar(im)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                )

        configure_axes(
            ax,
            title="Confusion Matrix",
            xlabel="Predicted",
            ylabel="True",
        )

        export_figure(
            fig,
            filename=filename,
            show=show,
        )

        return fig, ax

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------
    @staticmethod
    def roc_curve(
        fpr,
        tpr,
        auc: float | None = None,
        filename: str | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:

        fig, ax = create_figure()

        label = (
            f"ROC (AUC={auc:.4f})"
            if auc is not None
            else "ROC"
        )

        ax.plot(
            fpr,
            tpr,
            color=COLORS.roc,
            label=label,
        )

        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            color=COLORS.diagonal,
        )

        configure_axes(
            ax,
            title="ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            legend=True,
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax

    # --------------------------------------------------------
    # Precision Recall
    # --------------------------------------------------------
    @staticmethod
    def precision_recall_curve(
        recall,
        precision,
        filename: str | None = None,
        show: bool = True,
    ):

        fig, ax = create_figure()

        ax.plot(
            recall,
            precision,
            color=COLORS.pr,
        )

        configure_axes(
            ax,
            title="Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
        )

        export_figure(
            fig,
            filename,
            show,
        )

        return fig, ax