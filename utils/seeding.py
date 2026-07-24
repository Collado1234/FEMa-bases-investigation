"""
Controle centralizado de reprodutibilidade.

Toda a aleatoriedade do projeto (splits de CV, inicializacao de modelos,
bootstrap, etc.) deve derivar de uma unica seed mestre definida em
configs/base.yaml. Isso garante que dado o mesmo config, o experimento
inteiro seja bit-a-bit reproduzivel.
"""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixa a seed do Python `random` e do numpy. Chamar no inicio de cada
    processo/worker antes de qualquer operacao estocastica."""
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(master_seed: int, *parts) -> int:
    """Deriva uma seed determinística e estável a partir da seed mestre e de
    um conjunto de identificadores (ex: nome do modelo, indice do fold,
    indice da repeticao). Usar em vez de seeds aleatorias soltas, para que
    o MESMO fold/repeticao sempre receba a MESMA seed entre execucoes.

    Exemplo: derive_seed(42, "random_forest", "fold=2", "repeat=1")
    """
    key = f"{master_seed}|" + "|".join(str(p) for p in parts)
    # hash estável (independe do PYTHONHASHSEED, ao contrário de hash())
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # usa os primeiros 8 hex chars como inteiro de 32 bits (faixa valida p/ numpy/sklearn)
    return int(digest[:8], 16) % (2**31 - 1)
