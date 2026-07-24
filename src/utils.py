"""Utilitários pequenos e sem estado compartilhados pelo pipeline:
logging, controle de seed e hash determinístico (usado no checkpoint).
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixa a seed do Python `random`, do numpy e do PYTHONHASHSEED. Chamar
    no inicio de cada processo/worker antes de qualquer operacao estocastica.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(master_seed: int, *parts) -> int:
    """Deriva uma seed deterministica e estavel a partir da seed mestre e de
    um conjunto de identificadores (ex: nome do modelo, combo_id, indice do
    fold, indice da repeticao). Usar em vez de seeds soltas, para que o
    MESMO fold/repeticao sempre receba a MESMA seed entre execucoes.

    Exemplo: derive_seed(42, "fema", "fold=2", "repeat=1")
    """
    key = f"{master_seed}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # primeiros 8 hex chars como inteiro de 32 bits (faixa valida p/ numpy/sklearn)
    return int(digest[:8], 16) % (2**31 - 1)
