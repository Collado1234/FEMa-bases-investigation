#!/usr/bin/env python3
"""
Caracteriza os datasets do protocolo FEMa diretamente pelos dados brutos.

Fonte de verdade:
- datasets/registry.py: identifica os datasets e seus metadados;
- datasets/loader.py: realiza a leitura e separação X/y.

Não utiliza reports/eda/dataset_overview.csv.

Execução na raiz do repositório:
    python scripts/characterize_datasets.py

Dependências:
    python -m pip install pandas tabulate scikit-learn

Saídas:
    reports/eda/dataset_characterization.csv
    reports/eda/dataset_characterization.md
    reports/eda/class_distributions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT.name == "scripts":
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from datasets.registry import get_dataset_spec  # noqa: E402
from datasets.loader import (  # noqa: E402
    _load_sklearn_builtin,
    _split_xy_from_csv,
)

# Os 28 datasets presentes na matriz experimental V3.
# bng_glass consta no registry, mas não integra esta lista V3.
V3_DATASETS = [
    "blood_transfusion_service_center",
    "breast_cancer",
    "climate_model_simulation_crashes",
    "diabetes",
    "digits",
    "fetal_health",
    "gina_agnostic",
    "haberman",
    "iris",
    "isolet",
    "letter",
    "magic_telescope",
    "mfeat_factors",
    "mfeat_fourier",
    "mfeat_karhunen",
    "mfeat_pixel",
    "mfeat_zernike",
    "monks_problems_1",
    "monks_problems_2",
    "musk",
    "page_blocks",
    "seeds",
    "segment",
    "spambase",
    "tic_tac_toe",
    "vertebra_column",
    "wine",
    "zoo",
]


def load_raw_xy(dataset_name: str):
    """
    Usa as rotinas de leitura já existentes em datasets/loader.py.

    Importante: não chama load_dataset(), pois essa função faz split,
    One-Hot Encoding e padronização. Para a tabela de caracterização,
    queremos contar as features originais do dataset, antes do One-Hot.
    """
    spec = get_dataset_spec(dataset_name)

    if spec.synthetic:
        raise ValueError(
            f"{dataset_name}: datasets sintéticos não fazem parte "
            "da caracterização dos datasets reais da V3."
        )

    if spec.sklearn_loader:
        return _load_sklearn_builtin(spec)

    return _split_xy_from_csv(spec)


def characterize_dataset(dataset_name: str) -> dict[str, Any]:
    X, y = load_raw_xy(dataset_name)

    # X pode ser DataFrame (CSV) ou ndarray (sklearn).
    n_samples = int(len(y))
    n_features = int(X.shape[1])

    labels, counts = pd.Series(y).value_counts().index, pd.Series(y).value_counts().values
    if len(counts) == 0 or min(counts) <= 0:
        raise ValueError(f"{dataset_name}: distribuição de classes inválida.")

    n_classes = int(len(counts))
    imbalance = float(max(counts) / min(counts))

    distribution = {
        str(label): int(count)
        for label, count in zip(labels, counts)
    }

    return {
        "dataset": dataset_name,
        "n_features": n_features,
        "n_amostras": n_samples,
        "n_classes": n_classes,
        "indice_desbalanceamento": imbalance,
        "distribuicao_classes": distribution,
    }


def classify_quartile_dimension(
    values: pd.Series, q1: float, q3: float
) -> tuple[pd.Series, pd.Series]:
    labels: list[str] = []
    points: list[int] = []

    for value in values:
        if value <= q1:
            labels.append("Pequena")
            points.append(1)
        elif value <= q3:
            labels.append("Média")
            points.append(2)
        else:
            labels.append("Grande")
            points.append(3)

    return (
        pd.Series(labels, index=values.index),
        pd.Series(points, index=values.index, dtype="int64"),
    )


def classify_imbalance(value: float) -> tuple[str, int]:
    if value <= 1.5:
        return "Balanceado", 0
    if value <= 3:
        return "Leve", 1
    if value <= 10:
        return "Moderado", 2
    return "Severo", 3


def classify_ihdd(value: int) -> str:
    # Faixas inferidas da tabela de referência fornecida.
    if value <= 5:
        return "Moderada"
    if value <= 7:
        return "Alta"
    return "Muito Alta"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caracteriza datasets diretamente via registry e loader."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=V3_DATASETS,
        help="Datasets a caracterizar. Padrão: os 28 datasets da V3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "eda",
        help="Diretório de saída.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for name in args.datasets:
        try:
            row = characterize_dataset(name)
            rows.append(row)
            print(
                f"[OK] {name}: {row['n_amostras']} amostras, "
                f"{row['n_features']} features, {row['n_classes']} classes"
            )
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError(
            "Não foi possível caracterizar todos os datasets. "
            "Nenhuma tabela parcial será salva.\n- "
            + "\n- ".join(failures)
        )

    result = pd.DataFrame(rows)

    if result["dataset"].duplicated().any():
        raise ValueError("A lista de datasets contém nomes duplicados.")

    # Quartis calculados sobre os datasets selecionados nesta execução.
    quartiles = {
        "features": (
            float(result["n_features"].quantile(0.25)),
            float(result["n_features"].quantile(0.75)),
        ),
        "amostras": (
            float(result["n_amostras"].quantile(0.25)),
            float(result["n_amostras"].quantile(0.75)),
        ),
        "classes": (
            float(result["n_classes"].quantile(0.25)),
            float(result["n_classes"].quantile(0.75)),
        ),
    }

    result["cat_features"], points_features = classify_quartile_dimension(
        result["n_features"], *quartiles["features"]
    )
    result["cat_amostras"], points_samples = classify_quartile_dimension(
        result["n_amostras"], *quartiles["amostras"]
    )
    result["cat_classes"], points_classes = classify_quartile_dimension(
        result["n_classes"], *quartiles["classes"]
    )

    imbalance_result = result["indice_desbalanceamento"].apply(
        classify_imbalance
    )
    result["cat_desbalanceamento"] = imbalance_result.apply(lambda x: x[0])
    points_imbalance = imbalance_result.apply(lambda x: x[1])

    result["IHDD"] = (
        points_features + points_samples + points_classes + points_imbalance
    )
    result["cat_IHDD"] = result["IHDD"].apply(classify_ihdd)

    result = result.sort_values("dataset").reset_index(drop=True)

    # Tabela principal: mantém também as dimensões originais para auditoria.
    table_columns = [
        "dataset",
        "n_features",
        "n_amostras",
        "n_classes",
        "indice_desbalanceamento",
        "cat_features",
        "cat_amostras",
        "cat_classes",
        "cat_desbalanceamento",
        "IHDD",
        "cat_IHDD",
    ]
    table = result[table_columns].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "dataset_characterization.csv"
    md_path = args.output_dir / "dataset_characterization.md"
    json_path = args.output_dir / "class_distributions.json"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    distributions = {
        row["dataset"]: row["distribuicao_classes"]
        for row in rows
    }
    json_path.write_text(
        json.dumps(distributions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Caracterização dos datasets",
        "",
        f"Quantidade de datasets: {len(table)}.",
        "",
        "Os metadados foram calculados diretamente a partir dos dados brutos, "
        "usando as especificações de `datasets/registry.py` e as rotinas de "
        "leitura de `datasets/loader.py`.",
        "",
        "A contagem de features considera as colunas originais de entrada, "
        "após `drop_columns` e a remoção do target, mas antes de One-Hot Encoding.",
        "",
        "## Quartis calculados nesta execução",
        "",
        "| Dimensão | Q1 | Q3 |",
        "|---|---:|---:|",
        f"| Features | {quartiles['features'][0]:g} | {quartiles['features'][1]:g} |",
        f"| Amostras | {quartiles['amostras'][0]:g} | {quartiles['amostras'][1]:g} |",
        f"| Classes | {quartiles['classes'][0]:g} | {quartiles['classes'][1]:g} |",
        "",
        "## Critérios",
        "",
        "- Pequena: valor menor ou igual a Q1 (1 ponto).",
        "- Média: valor acima de Q1 e menor ou igual a Q3 (2 pontos).",
        "- Grande: valor acima de Q3 (3 pontos).",
        "- Índice de desbalanceamento = contagem da classe majoritária / "
        "contagem da classe minoritária.",
        "- Balanceado: índice até 1,5 (0 pontos); Leve: até 3 (1 ponto); "
        "Moderado: até 10 (2 pontos); Severo: acima de 10 (3 pontos).",
        "- IHDD = pontos de features + amostras + classes + desbalanceamento.",
        "- Categoria IHDD: até 5 = Moderada; 6–7 = Alta; 8 ou mais = Muito Alta.",
        "",
        "O IHDD é uma heurística descritiva, não uma medida universal de dificuldade.",
        "",
        "## Tabela",
        "",
        table.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nQuartis calculados:")
    for dimension, (q1, q3) in quartiles.items():
        print(f"  {dimension}: Q1={q1:g}; Q3={q3:g}")

    print("\nArquivos gerados:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Distribuições: {json_path}")


if __name__ == "__main__":
    main()
