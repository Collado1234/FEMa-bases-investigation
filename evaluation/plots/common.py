"""
evaluation/plots/common.py
--------------------------

Funções utilitárias compartilhadas por todos os gráficos do framework FEMa.

Este módulo centraliza a criação, configuração, exibição e salvamento de
figuras, evitando duplicação de código entre gráficos de classificação
e regressão.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .styles import FIGURE, FONT, GRID, SAVE


def create_figure(
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
) -> tuple[Figure, Axes]:
    """
    Cria uma nova figura e seus eixos.

    Parameters
    ----------
    figsize
        Tamanho da figura (largura, altura).

    dpi
        Resolução da figura.

    Returns
    -------
    Figure
        Figura criada.

    Axes
        Eixo principal.
    """

    fig, ax = plt.subplots(
        figsize=figsize or FIGURE.figsize,
        dpi=dpi or FIGURE.dpi,
    )

    return fig, ax


def configure_axes(
    ax: Axes,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = False,
    grid: bool = True,
) -> None:
    """
    Configura os elementos básicos de um gráfico.
    """

    if title:
        ax.set_title(title, fontsize=FONT.title_size)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT.label_size)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT.label_size)

    ax.tick_params(labelsize=FONT.tick_size)

    if grid and GRID.enabled:
        ax.grid(
            True,
            alpha=GRID.alpha,
            linestyle=GRID.linestyle,
            linewidth=GRID.linewidth,
        )

    if legend:
        ax.legend(fontsize=FONT.legend_size)


def finalize_figure(
    fig: Figure,
) -> None:
    """
    Aplica configurações finais à figura.
    """

    if FIGURE.tight_layout:
        fig.tight_layout()


def save_figure(
    fig: Figure,
    filename: str | Path,
) -> None:
    """
    Salva uma figura em disco.

    Parameters
    ----------
    fig
        Figura a ser salva.

    filename
        Caminho do arquivo de saída.
    """

    path = Path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        bbox_inches=SAVE.bbox_inches,
        transparent=SAVE.transparent,
    )


def show_figure(
    fig: Figure,
) -> None:
    """
    Exibe a figura.
    """

    finalize_figure(fig)
    plt.show()


def close_figure(
    fig: Figure,
) -> None:
    """
    Fecha a figura liberando memória.
    """

    plt.close(fig)


def export_figure(
    fig: Figure,
    filename: str | Path | None = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Finaliza, salva, exibe e/ou fecha uma figura.

    Esta função concentra todo o ciclo de vida de uma figura.
    """

    finalize_figure(fig)

    if filename is not None:
        save_figure(fig, filename)

    if show:
        plt.show()

    if close:
        plt.close(fig)