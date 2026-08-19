"""
Camada de carregamento de dados.

Responsabilidade UNICA: ler o CSV (ou gerar o dataset sintetico) descrito
em datasets/registry.py, fazer o split train/val/test de forma
deterministica e, opcionalmente, padronizar as features (necessario para
o FEMa, que e um metodo baseado em distancia). Nao conhece nenhum modelo
nem faz parte do pipeline de tuning/avaliacao.

Features categoricas sao convertidas para representacao numerica por
One-Hot Encoding. O encoder e ajustado SOMENTE no conjunto de treino e
depois aplicado aos conjuntos de validacao e teste, evitando vazamento
de informacao.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
    """Padroniza usando somente estatisticas do treino.

    Essencial para modelos baseados em distancia (FEMa) e util para
    os demais modelos.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    scaled = [(X_train - mean) / std]

    for X in others:
        if X is not None and len(X):
            scaled.append((X - mean) / std)
        else:
            scaled.append(X)

    return scaled


def _load_csv(spec: DatasetSpec) -> pd.DataFrame:
    """Le o CSV e remove colunas explicitamente descartadas."""
    df = pd.read_csv(
        spec.csv_path,
        sep=spec.delimiter,
        engine="python",
    )

    df.columns = [
        c.strip().lstrip("\ufeff")
        for c in df.columns
    ]

    df = df.drop(
        columns=[
            c
            for c in spec.drop_columns
            if c in df.columns
        ]
    )

    return df


def _split_xy_from_csv(spec: DatasetSpec):
    """Carrega um CSV e retorna X como DataFrame e y como ndarray.

    Diferentemente da implementacao anterior, X permanece como DataFrame
    para que features categoricas possam ser identificadas e codificadas
    depois do split.
    """
    df = _load_csv(spec)

    y_raw = df[spec.target_column]
    X = df.drop(columns=[spec.target_column]).copy()

    # Target numerico
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy().astype(int)

    # Target categorico
    else:
        y = y_raw.astype("category").cat.codes.to_numpy()

    return X, y


def _encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame],
):
    """Converte features categoricas para representacao numerica.

    O OneHotEncoder e ajustado exclusivamente em X_train.

    Features numericas sao mantidas como estao.
    Features categoricas sao convertidas para colunas binarias.

    handle_unknown='ignore' garante que uma categoria encontrada em
    validacao/teste, mas ausente no treino, nao cause erro.
    """

    if not isinstance(X_train, pd.DataFrame):
        return (
            np.asarray(X_train, dtype=np.float64),
            None if X_val is None else np.asarray(X_val, dtype=np.float64),
            None if X_test is None else np.asarray(X_test, dtype=np.float64),
        )

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col
        for col in X_train.columns
        if col not in categorical_columns
    ]

    # Nenhuma feature categorica.
    if not categorical_columns:
        return (
            X_train.to_numpy(dtype=np.float64),
            None if X_val is None else X_val.to_numpy(dtype=np.float64),
            None if X_test is None else X_test.to_numpy(dtype=np.float64),
        )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    # Ajusta SOMENTE no treino.
    X_train_cat = encoder.fit_transform(
        X_train[categorical_columns]
    )

    # Converte as partes numericas.
    if numeric_columns:
        X_train_num = X_train[numeric_columns].to_numpy(
            dtype=np.float64
        )

        X_train_encoded = np.hstack(
            [X_train_num, X_train_cat]
        )
    else:
        X_train_encoded = X_train_cat

    def transform(df):
        if df is None:
            return None

        X_cat = encoder.transform(
            df[categorical_columns]
        )

        if numeric_columns:
            X_num = df[numeric_columns].to_numpy(
                dtype=np.float64
            )

            return np.hstack(
                [X_num, X_cat]
            )

        return X_cat

    X_val_encoded = transform(X_val)
    X_test_encoded = transform(X_test)

    return (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
    )


def _make_synthetic(
    seed: int,
    n_samples=600,
    n_features=12,
    n_classes=3,
):
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
    "breast_cancer": "load_breast_cancer",
    "wine": "load_wine",
}


def _load_sklearn_builtin(spec: DatasetSpec):
    """Carrega um dataset embutido do sklearn."""
    import sklearn.datasets as skds

    loader_name = _SKLEARN_LOADERS.get(spec.sklearn_loader)

    if loader_name is None:
        raise ValueError(
            f"sklearn_loader '{spec.sklearn_loader}' nao suportado. "
            f"Opcoes: {list(_SKLEARN_LOADERS)}"
        )

    bunch = getattr(skds, loader_name)()

    X = np.asarray(
        bunch.data,
        dtype=np.float64,
    )

    y = np.asarray(
        bunch.target
    ).astype(int)

    if spec.class_filter is not None:
        mask = np.isin(
            y,
            spec.class_filter,
        )

        X = X[mask]
        y = y[mask]

        remap = {
            old: new
            for new, old in enumerate(
                sorted(spec.class_filter)
            )
        }

        y = np.array(
            [
                remap[label]
                for label in y
            ]
        )

    return X, y


def load_dataset(
    dataset_name: str,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scale: bool = True,
) -> LoadedData:
    """Carrega o dataset, faz split deterministico e prepara as features.

    Para datasets CSV:

    1. carrega X como DataFrame;
    2. separa treino, validacao e teste;
    3. identifica features categoricas;
    4. aplica One-Hot Encoding ajustado somente no treino;
    5. padroniza usando estatisticas do treino.

    Para datasets sklearn e sinteticos, X ja e numerico.
    """

    spec = get_dataset_spec(dataset_name)

    # ============================================================
    # CARREGAMENTO
    # ============================================================

    if spec.synthetic:
        X, y = _make_synthetic(seed)

    elif spec.sklearn_loader:
        X, y = _load_sklearn_builtin(spec)

    else:
        X, y = _split_xy_from_csv(spec)

    # ============================================================
    # SPLIT
    # ============================================================

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    val_adj = val_ratio / (1 - test_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_adj,
        random_state=seed,
        stratify=y_temp,
    )

    # ============================================================
    # FEATURES CATEGORICAS
    # ============================================================

    X_train, X_val, X_test = _encode_categorical_features(
        X_train,
        X_val,
        X_test,
    )

    # ============================================================
    # PADRONIZACAO
    # ============================================================

    if scale:
        X_train, X_val, X_test = _standardize(
            X_train,
            X_val,
            X_test,
        )

    # ============================================================
    # RETORNO
    # ============================================================

    return LoadedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
    )