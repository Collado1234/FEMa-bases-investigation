from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize(arr: NDArray) -> NDArray:
    return np.asarray(arr).ravel()


def _check_binary(y: NDArray) -> None:
    unique = np.unique(y)
    if len(unique) > 2:
        raise ValueError("Curvas só suportam classificação binária.")