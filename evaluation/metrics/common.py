"""
evaluation/metrics/common.py
----------------------------

Funções utilitárias compartilhadas entre classificação e regressão.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize(arr: NDArray) -> NDArray:
    """Converte qualquer entrada em vetor 1D numpy."""
    return np.asarray(arr).ravel()


def _safe_div(numerator: float, denominator: float) -> float:
    """Divisão segura evitando divisão por zero."""
    return numerator / denominator if denominator != 0 else 0.0


def _to_float_array(arr: NDArray) -> NDArray:
    """Garante array float."""
    return np.asarray(arr, dtype=float).ravel()


def _to_int_array(arr: NDArray) -> NDArray:
    """Garante array int."""
    return np.asarray(arr, dtype=int).ravel()