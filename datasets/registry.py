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
    # Dataset embutido do sklearn (ex: "digits"), carregado via
    # datasets/loader.py::_load_sklearn_builtin em vez de CSV. Usa sempre a
    # base COMPLETA (todas as amostras) do sklearn; class_filter, se
    # informado, apenas SELECIONA um subconjunto de classes dessa base
    # completa (nao reduz amostras dentro das classes mantidas).
    sklearn_loader: Optional[str] = None
    class_filter: Optional[Tuple[int, ...]] = None


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
    # Dataset de 5 classes do protocolo experimental (IC): base COMPLETA de
    # digitos do sklearn (load_digits, 1797 amostras, 64 features/pixels,
    # sem missing values, sem pre-processamento necessario alem do scale
    # padrao ja feito por datasets/loader.py), filtrada para os digitos
    # 0-4 (5 classes) - mantem TODAS as amostras dessas 5 classes na base
    # completa, nao um subconjunto reduzido artificialmente.
    "digits_5class": DatasetSpec(
        name="digits_5class",
        csv_path=None,
        target_column=None,
        sklearn_loader="digits",
        class_filter=(0, 1, 2, 3, 4),
    ),
    # Base completa de digitos (10 classes), sem filtro - disponivel caso
    # seja util para outras analises alem do protocolo de 3 datasets.
    "digits": DatasetSpec(
        name="digits",
        csv_path=None,
        target_column=None,
        sklearn_loader="digits",
    ),
    # Breast Cancer Wisconsin (sklearn.load_breast_cancer): binario (2
    # classes), 30 features, 569 amostras, sem missing values.
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        csv_path=None,
        target_column=None,
        sklearn_loader="breast_cancer",
    ),
    # Wine (sklearn.load_wine): 3 classes, 13 features, 178 amostras, sem
    # missing values.
    "wine": DatasetSpec(
        name="wine",
        csv_path=None,
        target_column=None,
        sklearn_loader="wine",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise ValueError(f"Dataset '{name}' desconhecido. Opcoes: {available_datasets()}")
    spec = _REGISTRY[name]
    if not spec.synthetic and not spec.sklearn_loader and not spec.csv_path.exists():
        raise FileNotFoundError(f"Arquivo do dataset '{name}' nao encontrado em {spec.csv_path}.")
    return spec


def available_datasets():
    return list(_REGISTRY.keys())
