"""Carregamento de datasets. Um dataset é identificado por nome (usado na
config YAML) e resolvido para um DataSplit (train/val/test). Datasets novos
se registram com uma linha em DATASETS (dict nome -> função sem argumentos).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    name: str = ""


def _standardize(X_train, *others):
    """Padroniza (x - média) / desvio usando estatísticas do treino — essencial
    para modelos baseados em distância (FEMa) e ajuda os demais também."""
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    scaled = [(X_train - mean) / std]
    for X in others:
        scaled.append((X - mean) / std if X is not None and len(X) else X)
    return scaled


def load_csv_dataset(csv_path: str, target_column: str, drop_columns=(), delimiter: str = ",",
                      val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42,
                      scale: bool = True, name: str = "") -> DataSplit:
    """Carrega um CSV único (features + rótulo) e faz o split train/val/test
    uma única vez, de forma determinística (mesma seed sempre)."""
    df = pd.read_csv(csv_path, sep=delimiter, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    y_raw = df[target_column]
    X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)
    y = (y_raw.to_numpy().astype(int) if pd.api.types.is_numeric_dtype(y_raw)
         else y_raw.astype("category").cat.codes.to_numpy())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y)
    val_adj = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj, random_state=seed, stratify=y_temp)

    if scale:
        X_train, X_val, X_test = _standardize(X_train, X_val, X_test)

    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, name=name)


def load_synthetic(n_samples=600, n_features=12, n_classes=3, seed=42,
                    val_ratio=0.15, test_ratio=0.15, name="synthetic_demo") -> DataSplit:
    """Dataset sintético — smoke test do pipeline sem depender de nenhum CSV."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=max(4, n_features // 2),
        n_classes=n_classes, n_clusters_per_class=1, random_state=seed)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)
    val_adj = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_adj, random_state=seed, stratify=y_temp)
    X_train, X_val, X_test = _standardize(X_train, X_val, X_test)
    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, name=name)


# nome (usado na config) -> função sem argumentos que devolve DataSplit
DATASETS: Dict[str, Callable[[], DataSplit]] = {
    "fetal_health": lambda: load_csv_dataset(
        DATA_RAW / "fetal_health.csv", target_column="fetal_health", name="fetal_health"),
    "iris": lambda: load_csv_dataset(
        DATA_RAW / "IrisDataset.csv", target_column="Species", drop_columns=("Id",), name="iris"),
    "classification_data": lambda: load_csv_dataset(
        DATA_RAW / "classificationData.csv", target_column="class", delimiter=";", name="classification_data"),
    "synthetic_demo": load_synthetic,
}


def get_dataset(name: str) -> DataSplit:
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' não encontrado. Disponíveis: {sorted(DATASETS)}")
    return DATASETS[name]()
