"""
Adapta a camada de dados já existente (io.loaders.data_loader) para o
resto do framework, sem alterá-la. Um "dataset" aqui é identificado por
nome (ex.: "fetal_health") e resolvido para um diretório em data/raw ou
data/processed contendo os splits esperados por io.loaders.data_loader.

Para datasets que ainda são um único CSV (como os arquivos em data/raw/
deste projeto), fornecemos um loader alternativo que aplica splitter.py
para gerar train/val/test antes de devolver o mesmo formato DataSplit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np

from data_io.loaders.data_loader import DataSplit, load_dataset
from preprocessing.splitter import train_test_split as _two_way_split

# NOTA: preprocessing.splitter.train_val_test_split tem um bug pré-existente
# (desempacota a fatia de TESTE do split auxiliar de `stratify` onde deveria
# pegar a fatia de TREINO, quebrando sempre que val_ratio > 0 — ver
# README.md, seção "Bugs conhecidos"). Como a etapa de preprocessing não
# deve ser modificada, este módulo usa train_test_split (que está correto)
# duas vezes em sequência para obter train/val/test, em vez do
# train_val_test_split quebrado.


def _three_way_split(X, y, val_ratio, test_ratio, seed):
    X_temp, X_test, y_temp, y_test = _two_way_split(
        X, y, test_ratio=test_ratio, random_state=seed, stratify=y
    )
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = _two_way_split(
        X_temp, y_temp, test_ratio=val_ratio_adjusted, random_state=seed, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

_REGISTRY: Dict[str, Callable[[], DataSplit]] = {}


def register_dataset(name: str, loader_fn: Callable[[], DataSplit]) -> None:
    _REGISTRY[name] = loader_fn


def get_dataset(name: str) -> DataSplit:
    if name in _REGISTRY:
        return _REGISTRY[name]()
    raise KeyError(f"Dataset '{name}' não registrado. Disponíveis: {sorted(_REGISTRY)}")


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


def register_prepared_dir(name: str, directory: str) -> None:
    """Dataset já dividido em X_train.csv/y_train.csv/... (formato nativo do loader)."""
    register_dataset(name, lambda: load_dataset(Path(directory)))


def register_single_csv(
    name: str,
    csv_path: str,
    target_column: str,
    drop_columns=(),
    delimiter: str = ",",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Dataset em um único CSV (features + rótulo), split feito uma vez e
    reaproveitando a mesma seed sempre — determinístico entre execuções.
    Usa pandas para tolerar colunas categóricas (ex.: espécie do Iris) e
    delimitadores diferentes (ex.: ';' em classificationData.csv)."""

    def _loader() -> DataSplit:
        import pandas as pd

        df = pd.read_csv(csv_path, sep=delimiter, engine="python")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        y_raw = df[target_column]
        X_df = df.drop(columns=[target_column])

        if not np.issubdtype(y_raw.dtype, np.number):
            y = y_raw.astype("category").cat.codes.to_numpy()
        else:
            y = y_raw.to_numpy().astype(int)

        X = X_df.to_numpy(dtype=np.float64)

        X_train, y_train, X_val, y_val, X_test, y_test = _three_way_split(
            X, y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            name=name,
        )

    register_dataset(name, _loader)


def register_synthetic(name: str, n_samples: int = 600, n_features: int = 12,
                        n_classes: int = 3, seed: int = 42,
                        val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
    """Dataset sintético (sklearn.make_classification) — útil para smoke
    tests do pipeline sem depender de nenhum CSV específico."""

    def _loader() -> DataSplit:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=max(4, n_features // 2),
            n_classes=n_classes, n_clusters_per_class=1, random_state=seed,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = _three_way_split(
            X, y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            name=name,
        )

    register_dataset(name, _loader)


def _register_builtin_datasets() -> None:
    """Registra os CSVs já presentes em data/raw/ deste repositório."""
    base = Path(__file__).resolve().parent.parent / "data" / "raw"

    fetal = base / "fetal_health.csv"
    if fetal.exists():
        register_single_csv(name="fetal_health", csv_path=str(fetal), target_column="fetal_health")

    iris = base / "IrisDataset.csv"
    if iris.exists():
        register_single_csv(
            name="iris", csv_path=str(iris), target_column="Species", drop_columns=("Id",)
        )

    classif = base / "classificationData.csv"
    if classif.exists():
        register_single_csv(
            name="classification_data", csv_path=str(classif), target_column="class", delimiter=";"
        )

    register_synthetic("synthetic_demo")


_register_builtin_datasets()
