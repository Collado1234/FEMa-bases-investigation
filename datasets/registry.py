"""
Registro de datasets.

Mapeia um nome logico de dataset ("fetal_health", "iris",
"classification_data", "synthetic_demo") para os metadados necessarios
para carrega-lo: caminho do CSV, coluna de target, colunas a descartar,
delimitador. O dataset sintetico nao tem arquivo — e gerado em tempo de
execucao (ver datasets/loader.py).

Este modulo NUNCA le nem transforma dados — apenas descreve onde/como
encontra-los. Isso e responsabilidade exclusiva de datasets/loader.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Optional[Path]
    target_column: Optional[str]
    drop_columns: Tuple[str, ...] = field(default_factory=tuple)
    delimiter: str = ","
    synthetic: bool = False


_REGISTRY = {
    "fetal_health": DatasetSpec(
        name="fetal_health",
        csv_path=DATA_RAW / "fetal_health.csv",
        target_column="fetal_health",
    ),
    "iris": DatasetSpec(
        name="iris",
        csv_path=DATA_RAW / "IrisDataset.csv",
        target_column="Species",
        drop_columns=("Id",),
    ),
    "classification_data": DatasetSpec(
        name="classification_data",
        csv_path=DATA_RAW / "classificationData.csv",
        target_column="class",
        delimiter=";",
    ),
    "synthetic_demo": DatasetSpec(
        name="synthetic_demo",
        csv_path=None,
        target_column=None,
        synthetic=True,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise ValueError(f"Dataset '{name}' desconhecido. Opcoes: {available_datasets()}")
    spec = _REGISTRY[name]
    if not spec.synthetic and not spec.csv_path.exists():
        raise FileNotFoundError(f"Arquivo do dataset '{name}' nao encontrado em {spec.csv_path}.")
    return spec


def available_datasets():
    return list(_REGISTRY.keys())
