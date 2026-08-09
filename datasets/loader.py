"""
Camada de carregamento de dados.

Responsabilidade UNICA: ler o CSV (ou gerar o dataset sintetico) descrito
em datasets/registry.py, fazer o split train/val/test de forma
deterministica e, opcionalmente, padronizar as features (necessario para
o FEMa, que e um metodo baseado em distancia). Nao conhece nenhum modelo
nem faz parte do pipeline de tuning/avaliacao.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.registry import DatasetSpec, get_dataset_spec


@dataclass(frozen=True)
class LoadedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    dataset_name: str


def _standardize(X_train, *others):
    """Padroniza (x - media) / desvio usando estatisticas do treino -
    essencial para modelos baseados em distancia (FEMa) e ajuda os demais
    tambem."""
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    scaled = [(X_train - mean) / std]
    for X in others:
        scaled.append((X - mean) / std if X is not None and len(X) else X)
    return scaled


def _load_csv(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.csv_path, sep=spec.delimiter, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.drop(columns=[c for c in spec.drop_columns if c in df.columns])
    return df


def _split_xy_from_csv(spec: DatasetSpec):
    df = _load_csv(spec)
    y_raw = df[spec.target_column]
    X = df.drop(columns=[spec.target_column]).to_numpy(dtype=np.float64)
    y = (
        y_raw.to_numpy().astype(int)
        if pd.api.types.is_numeric_dtype(y_raw)
        else y_raw.astype("category").cat.codes.to_numpy()
    )
    return X, y


def _make_synthetic(seed: int, n_samples=600, n_features=12, n_classes=3):
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )


_SKLEARN_LOADERS = {
    "digits": "load_digits",
}


def _load_sklearn_builtin(spec: DatasetSpec):
    """Carrega um dataset embutido do sklearn (sempre a base COMPLETA,
    todas as amostras) e, se `spec.class_filter` estiver definido, filtra
    as LINHAS cujo rotulo esta nesse conjunto de classes - nao reduz o
    numero de amostras dentro das classes mantidas."""
    import sklearn.datasets as skds

    loader_name = _SKLEARN_LOADERS.get(spec.sklearn_loader)
    if loader_name is None:
        raise ValueError(f"sklearn_loader '{spec.sklearn_loader}' nao suportado. Opcoes: {list(_SKLEARN_LOADERS)}")

    bunch = getattr(skds, loader_name)()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target).astype(int)

    if spec.class_filter is not None:
        mask = np.isin(y, spec.class_filter)
        X, y = X[mask], y[mask]
        # remapeia os rotulos para 0..n_classes-1 contiguos (ex: digitos
        # 0-4 ja sao 0..4, mas o remapeamento generico evita problemas
        # caso o class_filter usado nao comece em 0 ou tenha buracos).
        remap = {old: new for new, old in enumerate(sorted(spec.class_filter))}
        y = np.array([remap[label] for label in y])

    return X, y


def load_dataset(
    dataset_name: str,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scale: bool = True,
) -> LoadedData:
    """Carrega o dataset logico (nome definido em datasets/registry.py),
    faz o split train/val/test de forma deterministica (mesma seed sempre)
    e, se `scale=True`, padroniza as features com estatisticas do treino.
    """
    spec = get_dataset_spec(dataset_name)

    if spec.synthetic:
        X, y = _make_synthetic(seed)
    elif spec.sklearn_loader:
        X, y = _load_sklearn_builtin(spec)
    else:
        X, y = _split_xy_from_csv(spec)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    val_adj = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj, random_state=seed, stratify=y_temp
    )

    if scale:
        X_train, X_val, X_test = _standardize(X_train, X_val, X_test)

    return LoadedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
    )
