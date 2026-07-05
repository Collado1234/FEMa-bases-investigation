"""
Controle centralizado de aleatoriedade. Todo run passa por aqui antes de
qualquer fit/predict, garantindo que a mesma seed produza sempre o mesmo
resultado (dentro do que cada biblioteca permite garantir).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Fixa a seed em todas as fontes de aleatoriedade conhecidas do processo.
    Chamar isso imediatamente antes de cada fit() de cada fold/run — não só
    uma vez no início do processo — para que a ordem de execução dos runs
    nunca afete o resultado de um run específico.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except ImportError:
        pass
