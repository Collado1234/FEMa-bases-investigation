"""
evaluation/plots/styles.py
--------------------------

Configurações visuais padrão utilizadas pelos gráficos do framework FEMa.

Este módulo centraliza parâmetros de aparência para manter consistência
entre todas as figuras geradas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureStyle:
    """
    Configurações gerais de uma figura.
    """

    figsize: tuple[float, float] = (8.0, 6.0)
    dpi: int = 150
    tight_layout: bool = True


@dataclass(frozen=True)
class FontStyle:
    """
    Configurações de fontes.
    """

    title_size: int = 16
    label_size: int = 13
    tick_size: int = 11
    legend_size: int = 11


@dataclass(frozen=True)
class GridStyle:
    """
    Configurações da grade.
    """

    enabled: bool = True
    alpha: float = 0.30
    linestyle: str = "--"
    linewidth: float = 0.8


@dataclass(frozen=True)
class ColorStyle:
    """
    Paleta de cores padrão.
    """

    primary: str = "tab:blue"
    secondary: str = "tab:orange"
    success: str = "tab:green"
    danger: str = "tab:red"

    roc: str = "tab:blue"
    pr: str = "tab:green"

    diagonal: str = "gray"

    confusion_cmap: str = "Blues"


@dataclass(frozen=True)
class SaveStyle:
    """
    Configurações de exportação.
    """

    bbox_inches: str = "tight"
    transparent: bool = False


# ---------------------------------------------------------------------
# Objetos globais
# ---------------------------------------------------------------------

FIGURE = FigureStyle()

FONT = FontStyle()

GRID = GridStyle()

COLORS = ColorStyle()

SAVE = SaveStyle()