from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)



# DATA STRUCTURE
@dataclass(frozen=True)
class DataSplit:
    """
    Estrutura genérica de dataset para experimentos.

    Pode ser usada tanto para:
    - tuning de hiperparâmetros
    - benchmark final
    - experimentos comparativos

    Attributes
    ----------
    X_train, y_train : dados de treino
    X_val, y_val     : dados de validação (opcional em alguns experimentos)
    X_test, y_test   : dados de teste (opcional, usado em avaliação final)
    name             : identificador do dataset
    """

    X_train: np.ndarray
    y_train: np.ndarray

    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    name: str = ""


# IO UTIL
def _read_csv(path: Path) -> np.ndarray:
    """
    Lê CSV e retorna numpy array float64.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    return np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=np.float64
    )


def _read_if_exists(base: Path, filename: str) -> Optional[np.ndarray]:
    """
    Lê arquivo se existir, senão retorna None.
    """
    path = base / filename
    if not path.exists():
        return None
    return _read_csv(path)


# DATASET LOADER (GENÉRICO)
def load_dataset(dataset_dir: Path | str) -> DataSplit:
    """
    Carrega um dataset estruturado em CSVs.

    Espera arquivos:
        X_train.csv, y_train.csv
        X_val.csv,   y_val.csv      (opcional)
        X_test.csv,  y_test.csv     (opcional)

    Retorna DataSplit pronto para experimentos.
    """
    base = Path(dataset_dir)
    name = base.name

    # -------------------------
    # mandatory split
    # -------------------------
    X_train = _read_csv(base / "X_train.csv")
    y_train = _read_csv(base / "y_train.csv").ravel().astype(int)

    # -------------------------
    # optional validation
    # -------------------------
    X_val = _read_if_exists(base, "X_val.csv")
    y_val = _read_if_exists(base, "y_val.csv")

    if y_val is not None:
        y_val = y_val.ravel().astype(int)

    # -------------------------
    # optional test
    # -------------------------
    X_test = _read_if_exists(base, "X_test.csv")
    y_test = _read_if_exists(base, "y_test.csv")

    if y_test is not None:
        y_test = y_test.ravel().astype(int)

    # -------------------------
    # logging
    # -------------------------
    logger.info(
        "Dataset '%s' carregado | train=%d | val=%s | test=%s | features=%d",
        name,
        len(y_train),
        "yes" if X_val is not None else "no",
        "yes" if X_test is not None else "no",
        X_train.shape[1]
    )

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        name=name,
    )


# MULTI-DATASET LOADER (BENCHMARK READY)
def load_all_datasets(
    data_root: Path | str,
    pattern: str = "dataset_*"
) -> List[DataSplit]:
    """
    Carrega múltiplos datasets para benchmark experimental.
    """
    root = Path(data_root)
    dirs = sorted(root.glob(pattern))

    if not dirs:
        raise FileNotFoundError(
            f"Nenhum dataset encontrado em '{root}' com padrão '{pattern}'"
        )

    return [load_dataset(d) for d in dirs]