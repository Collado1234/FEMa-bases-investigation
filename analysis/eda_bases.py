"""
EDA automática de todos os datasets registrados no FEMa.

Uso:
    python analysis/eda_all.py

Saídas:
    reports/eda/dataset_overview.csv
    reports/eda/dataset_overview.json

Objetivo:
    Fazer uma triagem estrutural das bases antes da execução
    experimental do FEMa.

A análise NÃO altera os datasets e NÃO executa o pipeline de
treinamento/tuning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.registry import available_datasets, get_dataset_spec


# ============================================================
# CONFIGURAÇÃO
# ============================================================

OUTPUT_DIR = Path("reports/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILIDADES
# ============================================================

def safe_int(value):
    """Converte valores numpy para int nativo."""
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def safe_float(value):
    """Converte valores numpy para float nativo."""
    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except Exception:
        return None


def format_memory(bytes_value):
    """Converte bytes para uma representação legível."""
    if bytes_value is None:
        return "N/A"

    if bytes_value < 1024:
        return f"{bytes_value:.0f} B"

    if bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"

    if bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.2f} MB"

    return f"{bytes_value / (1024 ** 3):.2f} GB"


def classify_balance(class_counts):
    """
    Classifica aproximadamente o balanceamento.

    A razão é:
        menor classe / maior classe

    Interpretação:
        >= 0.80 -> balanceada
        >= 0.50 -> moderadamente desbalanceada
        <  0.50 -> desbalanceada
    """

    if not class_counts:
        return "N/A", None

    values = list(class_counts.values())

    if not values or max(values) == 0:
        return "N/A", None

    ratio = min(values) / max(values)

    if ratio >= 0.80:
        label = "balanced"
    elif ratio >= 0.50:
        label = "moderately_imbalanced"
    else:
        label = "imbalanced"

    return label, ratio


def estimate_one_hot_dimension(
    df: pd.DataFrame,
    target_column: str,
):
    """
    Estima a dimensionalidade após One-Hot Encoding.

    Features numéricas:
        permanecem 1 coluna.

    Features categóricas:
        viram uma coluna por categoria observada.
    """

    X = df.drop(columns=[target_column])

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col
        for col in X.columns
        if col not in categorical_columns
    ]

    categorical_cardinalities = {
        col: int(X[col].nunique(dropna=True))
        for col in categorical_columns
    }

    one_hot_features = len(numeric_columns) + sum(
        categorical_cardinalities.values()
    )

    return (
        len(numeric_columns),
        len(categorical_columns),
        categorical_cardinalities,
        one_hot_features,
    )


# ============================================================
# DATASETS SKLEARN
# ============================================================

SKLEARN_LOADERS = {
    "digits": "load_digits",
    "breast_cancer": "load_breast_cancer",
    "wine": "load_wine",
}


def analyze_sklearn_dataset(spec):
    """Analisa dataset embutido do sklearn."""

    import sklearn.datasets as skds

    loader_name = SKLEARN_LOADERS.get(spec.sklearn_loader)

    if loader_name is None:
        raise ValueError(
            f"Loader sklearn não suportado: {spec.sklearn_loader}"
        )

    bunch = getattr(skds, loader_name)()

    X = np.asarray(bunch.data)
    y = np.asarray(bunch.target)

    if spec.class_filter is not None:
        mask = np.isin(y, spec.class_filter)
        X = X[mask]
        y = y[mask]

    n_samples, n_features = X.shape

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    class_counts = {
        str(cls): int(count)
        for cls, count in zip(classes, counts)
    }

    balance, balance_ratio = classify_balance(
        class_counts
    )

    memory_bytes = X.nbytes + y.nbytes

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_features_numeric": n_features,
        "n_features_categorical": 0,
        "n_classes": len(classes),
        "class_distribution": class_counts,
        "balance": balance,
        "balance_ratio": safe_float(balance_ratio),
        "missing_values": 0,
        "missing_percentage": 0.0,
        "infinite_values": int(
            np.isinf(X).sum()
        ),
        "constant_features": int(
            np.sum(np.nanstd(X, axis=0) == 0)
        ),
        "duplicate_rows": None,
        "categorical_cardinalities": {},
        "one_hot_features": n_features,
        "memory_bytes": memory_bytes,
        "memory": format_memory(memory_bytes),
    }


# ============================================================
# DATASET SINTÉTICO
# ============================================================

def analyze_synthetic(spec):
    """Analisa o dataset sintético do registry."""

    from datasets.loader import _make_synthetic

    X, y = _make_synthetic(seed=42)

    n_samples, n_features = X.shape

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    class_counts = {
        str(cls): int(count)
        for cls, count in zip(classes, counts)
    }

    balance, balance_ratio = classify_balance(
        class_counts
    )

    memory_bytes = X.nbytes + y.nbytes

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_features_numeric": n_features,
        "n_features_categorical": 0,
        "n_classes": len(classes),
        "class_distribution": class_counts,
        "balance": balance,
        "balance_ratio": safe_float(balance_ratio),
        "missing_values": 0,
        "missing_percentage": 0.0,
        "infinite_values": int(np.isinf(X).sum()),
        "constant_features": int(
            np.sum(np.nanstd(X, axis=0) == 0)
        ),
        "duplicate_rows": None,
        "categorical_cardinalities": {},
        "one_hot_features": n_features,
        "memory_bytes": memory_bytes,
        "memory": format_memory(memory_bytes),
    }


# ============================================================
# DATASETS CSV
# ============================================================

def analyze_csv_dataset(spec):
    """Analisa uma base CSV registrada no registry."""

    df = pd.read_csv(
        spec.csv_path,
        sep=spec.delimiter,
        engine="python",
    )

    df.columns = [
        c.strip().lstrip("\ufeff")
        for c in df.columns
    ]

    # Remove colunas explicitamente descartadas pelo registry.
    drop_columns = [
        c
        for c in spec.drop_columns
        if c in df.columns
    ]

    if drop_columns:
        df = df.drop(columns=drop_columns)

    if spec.target_column not in df.columns:
        raise ValueError(
            f"Target '{spec.target_column}' não encontrado. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    y = df[spec.target_column]

    X = df.drop(
        columns=[spec.target_column]
    )

    # --------------------------------------------------------
    # Tipos de features
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "string",
            "bool",
        ]
    ).columns.tolist()

    numeric_columns = [
        col
        for col in X.columns
        if col not in categorical_columns
    ]

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes, counts = np.unique(
        y.astype(str),
        return_counts=True,
    )

    class_counts = {
        str(cls): int(count)
        for cls, count in zip(classes, counts)
    }

    balance, balance_ratio = classify_balance(
        class_counts
    )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    missing_values = int(
        df.isna().sum().sum()
    )

    total_cells = df.shape[0] * df.shape[1]

    missing_percentage = (
        100.0 * missing_values / total_cells
        if total_cells
        else 0.0
    )

    # --------------------------------------------------------
    # Infinitos
    # --------------------------------------------------------

    infinite_values = 0

    for col in numeric_columns:
        values = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        infinite_values += int(
            np.isinf(values.to_numpy()).sum()
        )

    # --------------------------------------------------------
    # Features constantes
    # --------------------------------------------------------

    constant_features = 0

    for col in X.columns:
        if X[col].nunique(dropna=False) <= 1:
            constant_features += 1

    # --------------------------------------------------------
    # Duplicatas
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # --------------------------------------------------------
    # Cardinalidade categórica
    # --------------------------------------------------------

    categorical_cardinalities = {
        col: int(
            X[col].nunique(dropna=True)
        )
        for col in categorical_columns
    }

    # --------------------------------------------------------
    # Dimensão após One-Hot
    # --------------------------------------------------------

    one_hot_features = (
        len(numeric_columns)
        + sum(
            categorical_cardinalities.values()
        )
    )

    # --------------------------------------------------------
    # Memória
    # --------------------------------------------------------

    memory_bytes = int(
        df.memory_usage(
            deep=True
        ).sum()
    )

    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_numeric": len(numeric_columns),
        "n_features_categorical": len(
            categorical_columns
        ),
        "categorical_columns": categorical_columns,
        "n_classes": len(classes),
        "class_distribution": class_counts,
        "balance": balance,
        "balance_ratio": safe_float(balance_ratio),
        "missing_values": missing_values,
        "missing_percentage": safe_float(
            missing_percentage
        ),
        "infinite_values": infinite_values,
        "constant_features": constant_features,
        "duplicate_rows": duplicate_rows,
        "categorical_cardinalities": categorical_cardinalities,
        "one_hot_features": one_hot_features,
        "memory_bytes": memory_bytes,
        "memory": format_memory(memory_bytes),
    }


# ============================================================
# STATUS
# ============================================================

def determine_status(info):
    """Determina um status simples para triagem."""

    problems = []

    if info["missing_values"] > 0:
        problems.append("missing")

    if info["infinite_values"] > 0:
        problems.append("infinite")

    if info["constant_features"] > 0:
        problems.append("constant_features")

    if info["n_features_categorical"] > 0:
        problems.append("categorical")

    # One-hot pode aumentar muito a dimensionalidade.
    if (
        info["n_features"] > 0
        and info["one_hot_features"]
        > info["n_features"] * 5
    ):
        problems.append("one_hot_expansion")

    # Classes muito pequenas podem prejudicar CV estratificado.
    class_distribution = info.get(
        "class_distribution",
        {},
    )

    if class_distribution:
        min_class = min(
            class_distribution.values()
        )

        if min_class < 10:
            problems.append("small_class")

    if not problems:
        return "OK"

    return ";".join(problems)


# ============================================================
# ANÁLISE DE UM DATASET
# ============================================================

def analyze_dataset(dataset_name):
    """Analisa um dataset individual."""

    print(
        f"\n{'=' * 80}\n"
        f"DATASET: {dataset_name}\n"
        f"{'=' * 80}"
    )

    spec = get_dataset_spec(dataset_name)

    if spec.synthetic:
        info = analyze_synthetic(spec)

    elif spec.sklearn_loader:
        info = analyze_sklearn_dataset(spec)

    else:
        info = analyze_csv_dataset(spec)

    info["dataset"] = dataset_name
    info["status"] = determine_status(info)

    print(
        f"  Amostras:             {info['n_samples']}"
    )
    print(
        f"  Features:             {info['n_features']}"
    )
    print(
        f"  Numéricas:            {info['n_features_numeric']}"
    )
    print(
        f"  Categóricas:          {info['n_features_categorical']}"
    )
    print(
        f"  Após One-Hot:         {info['one_hot_features']}"
    )
    print(
        f"  Classes:              {info['n_classes']}"
    )
    print(
        f"  Balanceamento:        {info['balance']}"
    )
    print(
        f"  Missing:              {info['missing_values']}"
    )
    print(
        f"  Infinitos:            {info['infinite_values']}"
    )
    print(
        f"  Constantes:           {info['constant_features']}"
    )
    print(
        f"  Duplicatas:           {info['duplicate_rows']}"
    )
    print(
        f"  Memória:              {info['memory']}"
    )
    print(
        f"  STATUS:               {info['status']}"
    )

    if info.get("categorical_cardinalities"):
        print("\n  Cardinalidade categórica:")

        for column, cardinality in info[
            "categorical_cardinalities"
        ].items():
            print(
                f"    - {column}: {cardinality}"
            )

    return info


# ============================================================
# MAIN
# ============================================================

def main():

    datasets = available_datasets()

    print(
        f"\nEncontrados {len(datasets)} datasets "
        f"no registry."
    )

    results = []

    for dataset_name in datasets:

        try:
            result = analyze_dataset(
                dataset_name
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\nERRO ao analisar "
                f"'{dataset_name}': {exc}"
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "status": "ERROR",
                    "error": str(exc),
                }
            )

    # ========================================================
    # JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "dataset_overview.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # CSV
    # ========================================================

    csv_rows = []

    for result in results:

        row = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "class_distribution",
                "categorical_cardinalities",
                "categorical_columns",
                "error",
            }
        }

        row["class_distribution"] = json.dumps(
            result.get(
                "class_distribution",
                {},
            ),
            ensure_ascii=False,
        )

        row[
            "categorical_cardinalities"
        ] = json.dumps(
            result.get(
                "categorical_cardinalities",
                {},
            ),
            ensure_ascii=False,
        )

        row["categorical_columns"] = json.dumps(
            result.get(
                "categorical_columns",
                [],
            ),
            ensure_ascii=False,
        )

        if "error" in result:
            row["error"] = result["error"]

        csv_rows.append(row)

    df_results = pd.DataFrame(
        csv_rows
    )

    # Ordena colunas para facilitar leitura.
    preferred_columns = [
        "dataset",
        "status",
        "n_samples",
        "n_features",
        "n_features_numeric",
        "n_features_categorical",
        "one_hot_features",
        "n_classes",
        "balance",
        "balance_ratio",
        "missing_values",
        "missing_percentage",
        "infinite_values",
        "constant_features",
        "duplicate_rows",
        "memory",
        "memory_bytes",
        "class_distribution",
        "categorical_columns",
        "categorical_cardinalities",
        "error",
    ]

    columns = [
        col
        for col in preferred_columns
        if col in df_results.columns
    ]

    remaining = [
        col
        for col in df_results.columns
        if col not in columns
    ]

    df_results = df_results[
        columns + remaining
    ]

    csv_path = (
        OUTPUT_DIR
        / "dataset_overview.csv"
    )

    df_results.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print("EDA CONCLUÍDA")

    print("=" * 80)

    print(
        f"Datasets analisados: {len(results)}"
    )

    ok_count = sum(
        r.get("status") == "OK"
        for r in results
    )

    error_count = sum(
        r.get("status") == "ERROR"
        for r in results
    )

    print(
        f"OK:                   {ok_count}"
    )

    print(
        f"Com atenção:           "
        f"{len(results) - ok_count - error_count}"
    )

    print(
        f"Erros:                {error_count}"
    )

    print(
        f"\nCSV:  {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )


if __name__ == "__main__":
    main()