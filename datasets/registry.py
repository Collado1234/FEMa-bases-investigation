"""
Registro de datasets.

Mapeia um nome lógico de dataset para os metadados necessários
para carregá-lo: caminho do CSV, coluna de target, colunas a descartar,
delimitador.

Este módulo NUNCA lê nem transforma dados — apenas descreve onde/como
encontrá-los. Isso é responsabilidade exclusiva de datasets/loader.py.
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

    # Dataset embutido do sklearn.
    sklearn_loader: Optional[str] = None

    # Seleção opcional de classes para datasets sklearn.
    class_filter: Optional[Tuple[int, ...]] = None


_REGISTRY = {

    # ============================================================
    # DATASETS DO PROTOCOLO EXPERIMENTAL
    # ============================================================

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

    # "classification_data": DatasetSpec(
    #     name="classification_data",
    #     csv_path=DATA_RAW / "classificationData.csv",
    #     target_column="class",
    #     delimiter=";",
    # ),


    # 9 features | 7 classes | 137.781 amostras
    # BNG / dataset gerado | desbalanceado
    "bng_glass": DatasetSpec(
        name="bng_glass",
        csv_path=DATA_RAW / "bng_glass.csv",
        target_column="Type",
    ),

    # ============================================================
    # DATASETS OPENML — PROTOCOLO FEMa
    # ============================================================

    # 8 features | 2 classes | 768 amostras
    # Real | desbalanceado
    "diabetes": DatasetSpec(
        name="diabetes",
        csv_path=DATA_RAW / "diabetes.csv",
        target_column="class",
    ),

    # 57 features | 2 classes | 4601 amostras
    # Real | desbalanceado
    "spambase": DatasetSpec(
        name="spambase",
        csv_path=DATA_RAW / "spambase.csv",
        target_column="class",
    ),

    # 10 features | 2 classes | 19020 amostras
    # Real | desbalanceado
    "magic_telescope": DatasetSpec(
        name="magic_telescope",
        csv_path=DATA_RAW / "magictelescope.csv",
        target_column="class:",
    ),

    # 13 features | 3 classes | 178 amostras
    # Real | balanceado
    "wine": DatasetSpec(
        name="wine",
        csv_path=DATA_RAW / "wine.csv",
        target_column="class",
    ),

    # 7 features | 3 classes | 210 amostras
    # Real | balanceado
    "seeds": DatasetSpec(
        name="seeds",
        csv_path=DATA_RAW / "seeds.csv",
        target_column="Class",
    ),

    # 6 features | 3 classes | 310 amostras
    # Real | desbalanceado
    "vertebra_column": DatasetSpec(
        name="vertebra_column",
        csv_path=DATA_RAW / "vertebracolumn.csv",
        target_column="Class",
    ),

    # 10 features | 5 classes | 5473 amostras
    # Real | extremamente desbalanceado
    "page_blocks": DatasetSpec(
        name="page_blocks",
        csv_path=DATA_RAW / "pageblocks.csv",
        target_column="class",
    ),

    # 19 features | 7 classes | 2310 amostras
    # Real | balanceado
    "segment": DatasetSpec(
        name="segment",
        csv_path=DATA_RAW / "segment.csv",
        target_column="class",
    ),

    # 16 features | 26 classes | 20000 amostras
    # Real | balanceado
    "letter": DatasetSpec(
        name="letter",
        csv_path=DATA_RAW / "letter.csv",
        target_column="class",
    ),

    # 617 features | 26 classes | 7797 amostras
    # Real | balanceado
    "isolet": DatasetSpec(
        name="isolet",
        csv_path=DATA_RAW / "isolet.csv",
        target_column="class",
    ),
    # ============================================================
    # DATASETS SKLEARN
    # ============================================================

    # Base completa: 1797 amostras | 64 features | 10 classes.
    "digits": DatasetSpec(
        name="digits",
        csv_path=None,
        target_column=None,
        sklearn_loader="digits",
    ),

    # 1797 amostras | 64 features | 5 classes.
    # Filtra as classes 0-4 da base completa.
    # "digits_5class": DatasetSpec(
    #     name="digits_5class",
    #     csv_path=None,
    #     target_column=None,
    #     sklearn_loader="digits",
    #     class_filter=(0, 1, 2, 3, 4),
    # ),

    # 569 amostras | 30 features | 2 classes.
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        csv_path=None,
        target_column=None,
        sklearn_loader="breast_cancer",
    ),

    # 178 amostras | 13 features | 3 classes.
    # "wine": DatasetSpec(
    #     name="wine",
    #     csv_path=None,
    #     target_column=None,
    #     sklearn_loader="wine",
    # ),


    # ============================================================
    # DATASET SINTÉTICO
    # ============================================================

    # "synthetic_demo": DatasetSpec(
    #     name="synthetic_demo",
    #     csv_path=None,
    #     target_column=None,
    #     synthetic=True,
    # ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """
    Retorna a especificação de um dataset registrado.

    Para datasets baseados em CSV, verifica se o arquivo existe.
    """

    if name not in _REGISTRY:
        raise ValueError(
            f"Dataset '{name}' desconhecido. "
            f"Opções: {available_datasets()}"
        )

    spec = _REGISTRY[name]

    if (
        not spec.synthetic
        and not spec.sklearn_loader
        and not spec.csv_path.exists()
    ):
        raise FileNotFoundError(
            f"Arquivo do dataset '{name}' não encontrado "
            f"em {spec.csv_path}."
        )

    return spec


def available_datasets():
    """
    Retorna os nomes dos datasets disponíveis no registry.
    """

    return list(_REGISTRY.keys())