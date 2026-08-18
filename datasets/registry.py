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

    "classification_data": DatasetSpec(
        name="classification_data",
        csv_path=DATA_RAW / "classificationData.csv",
        target_column="class",
        delimiter=";",
    ),


    # ============================================================
    # DATASETS OPENML — BAIXA DIMENSIONALIDADE
    # ============================================================

    # 3 features | 2 classes | 573.891 amostras
    # Real | desbalanceado
    "avida_hil6": DatasetSpec(
        name="avida_hil6",
        csv_path=DATA_RAW / "avida-hil6.csv",
        target_column="label",
    ),

    # 3 features | 2 classes | 1.000.000 amostras
    # Sintético | levemente desbalanceado
    "sea_50000": DatasetSpec(
        name="sea_50000",
        csv_path=DATA_RAW / "sea_50000.csv",
        target_column="class",
    ),

    # 3 features | 5 classes | 73.503 amostras
    # Real | balanceado
    "products": DatasetSpec(
        name="products",
        csv_path=DATA_RAW / "products.csv",
        target_column="Product",
    ),

    # 4 features | 22 classes | 149.332 amostras
    # Real | desbalanceado
    "walking_activity": DatasetSpec(
        name="walking_activity",
        csv_path=DATA_RAW / "walking-activity.csv",
        target_column="Class",
    ),

    # 2 features | 54 classes | 500.000 amostras
    # Real | extremamente desbalanceado
    "us_inter_state_migration": DatasetSpec(
        name="us_inter_state_migration",
        csv_path=DATA_RAW / "us_inter_state_migration.csv",
        target_column="last_state",
    ),


    # ============================================================
    # DATASETS OPENML — BAIXA/MÉDIA DIMENSIONALIDADE
    # ============================================================

    # 9 features | 2 classes | 1.000.000 amostras
    # Sintético | desbalanceado
    "agrawal1": DatasetSpec(
        name="agrawal1",
        csv_path=DATA_RAW / "agrawal1.csv",
        target_column="class",
    ),

    # 9 features | 29 classes | 2.916.697 amostras
    # Real | extremamente desbalanceado
    "bitcoinheist_ransomware": DatasetSpec(
        name="bitcoinheist_ransomware",
        csv_path=DATA_RAW / "bitcoinheist_ransomware.csv",
        target_column="label",
    ),

    # 9 features | 7 classes | 137.781 amostras
    # BNG / dataset gerado | desbalanceado
    "bng_glass": DatasetSpec(
        name="bng_glass",
        csv_path=DATA_RAW / "bng_glass.csv",
        target_column="Type",
    ),


    # ============================================================
    # DATASETS OPENML — MÉDIA DIMENSIONALIDADE
    # ============================================================

    # 11 features | 3 classes | 78.053 amostras
    # Real | desbalanceado | 2 features categóricas
    "sdss17": DatasetSpec(
        name="sdss17",
        csv_path=DATA_RAW / "sdss17.csv",
        target_column="ObjectType",
    ),

    # 18 features | 3 classes | 78.600 amostras
    # Real | desbalanceado
    "risk_level_classification": DatasetSpec(
        name="risk_level_classification",
        csv_path=DATA_RAW / "risk_level_classification.csv",
        target_column="anomaly",
    ),

    # 7 features | 2 classes | 539.383 amostras
    # Real | balanceado | 4 features categóricas
    "airlines": DatasetSpec(
        name="airlines",
        csv_path=DATA_RAW / "airlines.csv",
        target_column="Delay",
    ),

    # 5 features | 49 classes | 1.292.579 amostras
    # Real | extremamente desbalanceado
    "ddxplus": DatasetSpec(
        name="ddxplus",
        csv_path=DATA_RAW / "ddxplus.csv",
        target_column="PATHOLOGY",
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
    "digits_5class": DatasetSpec(
        name="digits_5class",
        csv_path=None,
        target_column=None,
        sklearn_loader="digits",
        class_filter=(0, 1, 2, 3, 4),
    ),

    # 569 amostras | 30 features | 2 classes.
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        csv_path=None,
        target_column=None,
        sklearn_loader="breast_cancer",
    ),

    # 178 amostras | 13 features | 3 classes.
    "wine": DatasetSpec(
        name="wine",
        csv_path=None,
        target_column=None,
        sklearn_loader="wine",
    ),


    # ============================================================
    # DATASET SINTÉTICO
    # ============================================================

    "synthetic_demo": DatasetSpec(
        name="synthetic_demo",
        csv_path=None,
        target_column=None,
        synthetic=True,
    ),
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