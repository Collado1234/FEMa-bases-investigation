"""
Busca, valida e categoriza datasets de CLASSIFICAÇÃO do OpenML
para o Experimento_V2 do FEMa.

Grade experimental:

    Classes:
        2
        3
        5
        7-8
        15+

    Amostras:
        500
        2000
        5000
        10000+

    Features:
        3-5
        5-10
        10-20
        40-60
        100+

    Balanceamento:
        S = balanceado
        N = desbalanceado

O script:

1. Baixa os metadados do OpenML.
2. Filtra datasets de classificação.
3. Remove datasets com missing values.
4. Identifica candidatos para cada célula da grade.
5. Baixa cada candidato.
6. Valida os valores REAIS de X e y.
7. Calcula o Imbalance Ratio:
       IR = maior classe / menor classe
8. Classifica como:
       S -> IR <= 1.5
       N -> IR > 1.5
9. Gera CSVs para análise.
10. Mantém cache dos datasets já processados.

Instalação:

    pip install openml pandas numpy

Execução:

    python search_datasets_openml_v2.py

Arquivos gerados:

    openml_candidates_v2.csv
    openml_validated_v2.csv
    openml_experiment_v2.csv
    openml_failed_v2.csv
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import openml


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# ---------------------------------------------------------------------------
# Grade de classes
# ---------------------------------------------------------------------------

CLASS_BINS = {
    "2": (2, 2),
    "3": (3, 3),
    "5": (5, 5),
    "7-8": (7, 8),
    "15+": (15, float("inf")),
}


# ---------------------------------------------------------------------------
# Grade de número de amostras
#
# As faixas são deliberadamente largas para aumentar a quantidade de
# datasets disponíveis.
# ---------------------------------------------------------------------------

SAMPLE_BINS = {
    "500": (300, 999),
    "2000": (1500, 2999),
    "5000": (3500, 7499),
    "10000+": (7500, float("inf")),
}


# ---------------------------------------------------------------------------
# Grade de features
#
# ATENÇÃO:
#
# Existe uma lacuna entre 20 e 40 porque essa é a grade definida no projeto.
# Portanto, datasets com 21-39 features NÃO entram em nenhuma célula.
# ---------------------------------------------------------------------------

FEATURE_BINS = {
    "3-5": (3, 5),
    "5-10": (6, 10),
    "10-20": (11, 20),
    "40-60": (40, 60),
    "100+": (100, float("inf")),
}


# ---------------------------------------------------------------------------
# Balanceamento
# ---------------------------------------------------------------------------

IMBALANCE_RATIO_BALANCED_MAX = 1.5


# ---------------------------------------------------------------------------
# Arquivos de saída
# ---------------------------------------------------------------------------

CANDIDATES_FILE = "openml_candidates_v2.csv"
VALIDATED_FILE = "openml_validated_v2.csv"
EXPERIMENT_FILE = "openml_experiment_v2.csv"
FAILED_FILE = "openml_failed_v2.csv"


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def bin_of(value, bins):
    """
    Retorna o nome da faixa em que 'value' se encontra.
    """

    if pd.isna(value):
        return None

    value = float(value)

    for label, (lo, hi) in bins.items():
        if lo <= value <= hi:
            return label

    return None


def classify_balance(y, threshold=IMBALANCE_RATIO_BALANCED_MAX):
    """
    Calcula o Imbalance Ratio:

        IR = frequência da maior classe /
             frequência da menor classe

    Retorna:
        ir
        label -> S ou N
    """

    y = pd.Series(y)

    counts = y.value_counts()

    if len(counts) < 2:
        return np.nan, None

    min_count = counts.min()
    max_count = counts.max()

    if min_count == 0:
        return np.inf, "N"

    ir = float(max_count / min_count)

    label = "S" if ir <= threshold else "N"

    return ir, label


def make_cell(classes_bin, samples_bin, features_bin, balance_bin=None):
    """
    Monta o identificador da célula experimental.
    """

    cell = (
        f"classes={classes_bin} | "
        f"samples={samples_bin} | "
        f"features={features_bin}"
    )

    if balance_bin is not None:
        cell += f" | balance={balance_bin}"

    return cell


# ============================================================================
# ETAPA 1 — BUSCAR CANDIDATOS
# ============================================================================

def get_openml_candidates():
    """
    Baixa os metadados do OpenML e encontra candidatos.

    Essa etapa NÃO baixa os datasets completos.
    """

    print("=" * 80)
    print("ETAPA 1 — BUSCANDO DATASETS NO OPENML")
    print("=" * 80)

    print("\nBaixando lista de datasets do OpenML...")
    print("Isso pode demorar alguns minutos.\n")

    df = openml.datasets.list_datasets(
        output_format="dataframe"
    )

    print(f"Datasets encontrados no catálogo: {len(df)}")

    # ------------------------------------------------------------------------
    # Filtros básicos
    # ------------------------------------------------------------------------

    required_columns = [
        "NumberOfMissingValues",
        "NumberOfClasses",
        "NumberOfInstances",
        "status",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise RuntimeError(
                f"Coluna esperada não encontrada no OpenML: {col}"
            )

    df = df[
        (df["status"] == "active")
        & (df["NumberOfMissingValues"].fillna(0) == 0)
        & (df["NumberOfClasses"].fillna(0) >= 2)
        & (df["NumberOfInstances"] >= 300)
    ].copy()

    print(
        f"Candidatos após filtros básicos: {len(df)}"
    )

    # ------------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------------

    df["classes_bin"] = df["NumberOfClasses"].apply(
        lambda x: bin_of(x, CLASS_BINS)
    )

    # ------------------------------------------------------------------------
    # Amostras
    # ------------------------------------------------------------------------

    df["samples_bin"] = df["NumberOfInstances"].apply(
        lambda x: bin_of(x, SAMPLE_BINS)
    )

    # ------------------------------------------------------------------------
    # Features
    #
    # NÃO assumimos aqui que NumberOfFeatures - 1 é sempre correto.
    #
    # Portanto usamos NumberOfFeatures como filtro preliminar e a quantidade
    # real será validada posteriormente após o download.
    # ------------------------------------------------------------------------

    df["features_bin_metadata"] = df["NumberOfFeatures"].apply(
        lambda x: bin_of(x, FEATURE_BINS)
    )

    df = df.dropna(
        subset=[
            "classes_bin",
            "samples_bin",
            "features_bin_metadata",
        ]
    ).copy()

    # ------------------------------------------------------------------------
    # Célula preliminar
    # ------------------------------------------------------------------------

    df["cell_metadata"] = [
        make_cell(
            c,
            s,
            f
        )
        for c, s, f in zip(
            df["classes_bin"],
            df["samples_bin"],
            df["features_bin_metadata"],
        )
    ]

    # ------------------------------------------------------------------------
    # Ordenação
    #
    # Priorizamos datasets maiores e mais conhecidos.
    # ------------------------------------------------------------------------

    if "NumberOfDownloads" in df.columns:
        df = df.sort_values(
            "NumberOfDownloads",
            ascending=False,
            na_position="last",
        )

    else:
        df = df.sort_values(
            "NumberOfInstances",
            ascending=False,
        )

    # ------------------------------------------------------------------------
    # Colunas
    # ------------------------------------------------------------------------

    columns = [
        "did",
        "name",
        "NumberOfClasses",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfMissingValues",
        "classes_bin",
        "samples_bin",
        "features_bin_metadata",
        "cell_metadata",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    candidates = df[columns].copy()

    # Evita duplicatas do mesmo dataset
    candidates = candidates.drop_duplicates(
        subset=["did"]
    )

    candidates.to_csv(
        CANDIDATES_FILE,
        index=False,
    )

    print(
        f"\n{len(candidates)} candidatos encontrados."
    )

    print(
        f"Arquivo salvo: {CANDIDATES_FILE}"
    )

    print(
        f"Células preliminares cobertas: "
        f"{candidates['cell_metadata'].nunique()} / 100"
    )

    return candidates


# ============================================================================
# ETAPA 2 — VALIDAR DATASET REAL
# ============================================================================

def validate_dataset(row):
    """
    Baixa e valida um dataset individual.

    Retorna um dicionário com as informações reais.
    """

    did = int(row["did"])
    name = row["name"]

    result = {
        "did": did,
        "name": name,
        "status_validation": "OK",
        "error": "",
    }

    try:

        print(
            f"\n[DOWNLOAD] DID={did} | {name}"
        )

        dataset = openml.datasets.get_dataset(did)

        target = dataset.default_target_attribute

        if target is None:
            raise ValueError(
                "Dataset sem target definido."
            )

        # --------------------------------------------------------------------
        # Baixar dados
        # --------------------------------------------------------------------

        X, y, categorical_indicator, feature_names = (
            dataset.get_data(
                target=target
            )
        )

        X = pd.DataFrame(X)
        y = pd.Series(y)

        # --------------------------------------------------------------------
        # Missing values reais
        # --------------------------------------------------------------------

        missing_x = int(
            X.isna().sum().sum()
        )

        missing_y = int(
            y.isna().sum()
        )

        if missing_x > 0 or missing_y > 0:

            raise ValueError(
                f"Missing values encontrados: "
                f"X={missing_x}, y={missing_y}"
            )

        # --------------------------------------------------------------------
        # Informações reais
        # --------------------------------------------------------------------

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1])
        n_classes = int(y.nunique())

        # --------------------------------------------------------------------
        # Validar classificação
        # --------------------------------------------------------------------

        if n_classes < 2:
            raise ValueError(
                f"Dataset possui apenas {n_classes} classe(s)."
            )

        # --------------------------------------------------------------------
        # Identificar bins reais
        # --------------------------------------------------------------------

        classes_bin = bin_of(
            n_classes,
            CLASS_BINS
        )

        samples_bin = bin_of(
            n_samples,
            SAMPLE_BINS
        )

        features_bin = bin_of(
            n_features,
            FEATURE_BINS
        )

        # Dataset não pertence à grade
        if (
            classes_bin is None
            or samples_bin is None
            or features_bin is None
        ):

            raise ValueError(
                "Dataset não pertence à grade após "
                "validação dos valores reais."
            )

        # --------------------------------------------------------------------
        # Balanceamento
        # --------------------------------------------------------------------

        class_counts = y.value_counts()

        ir, balance = classify_balance(y)

        # --------------------------------------------------------------------
        # Célula
        # --------------------------------------------------------------------

        cell = make_cell(
            classes_bin,
            samples_bin,
            features_bin,
            balance,
        )

        # --------------------------------------------------------------------
        # Salvar resultado
        # --------------------------------------------------------------------

        result.update({

            "target": target,

            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,

            "classes_bin": classes_bin,
            "samples_bin": samples_bin,
            "features_bin": features_bin,

            "imbalance_ratio": ir,
            "balance": balance,

            "cell": cell,

            "min_class_count": int(
                class_counts.min()
            ),

            "max_class_count": int(
                class_counts.max()
            ),

        })

        print(
            f"  OK"
            f" | samples={n_samples}"
            f" | features={n_features}"
            f" | classes={n_classes}"
            f" | IR={ir:.2f}"
            f" | balance={balance}"
        )

        return result

    except Exception as e:

        error_message = str(e)

        print(
            f"  ERRO: {error_message}"
        )

        result["status_validation"] = "ERROR"
        result["error"] = error_message

        return result


# ============================================================================
# CACHE
# ============================================================================

def load_validation_cache():
    """
    Carrega resultados anteriores para permitir continuar uma execução
    interrompida.
    """

    if not os.path.exists(VALIDATED_FILE):
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            VALIDATED_FILE
        )

        print(
            f"\nCache encontrado: "
            f"{len(df)} datasets já processados."
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================================
# ETAPA 3 — VALIDAR TODOS
# ============================================================================

def validate_all(candidates):
    """
    Valida todos os candidatos.

    Se o programa for interrompido, os resultados já processados permanecem
    no CSV e serão reutilizados na próxima execução.
    """

    print("\n")
    print("=" * 80)
    print("ETAPA 2 — VALIDANDO DATASETS")
    print("=" * 80)

    cache = load_validation_cache()

    processed_ids = set()

    if not cache.empty and "did" in cache.columns:

        processed_ids = set(
            cache["did"].astype(int)
        )

    results = []

    # Reaproveitar cache
    if not cache.empty:

        results.extend(
            cache.to_dict("records")
        )

    total = len(candidates)

    for position, (_, row) in enumerate(
        candidates.iterrows(),
        start=1,
    ):

        did = int(row["did"])

        if did in processed_ids:

            print(
                f"\n[{position}/{total}] "
                f"DID={did} já processado. Pulando."
            )

            continue

        print(
            f"\n{'-' * 80}"
        )

        print(
            f"[{position}/{total}] "
            f"Processando {row['name']}"
        )

        result = validate_dataset(row)

        results.append(result)

        # --------------------------------------------------------------------
        # Salvar imediatamente
        # --------------------------------------------------------------------

        temp_df = pd.DataFrame(results)

        temp_df.to_csv(
            VALIDATED_FILE,
            index=False,
        )

        # Pequena pausa para não bombardear o servidor
        time.sleep(0.2)

    validated = pd.DataFrame(results)

    validated.to_csv(
        VALIDATED_FILE,
        index=False,
    )

    return validated


# ============================================================================
# ETAPA 4 — GERAR BASE FINAL DO EXPERIMENTO
# ============================================================================

def generate_experiment_dataset(validated):
    """
    Gera a tabela final de datasets utilizáveis no Experimento_V2.
    """

    print("\n")
    print("=" * 80)
    print("ETAPA 3 — GERANDO DATASETS DO EXPERIMENTO_V2")
    print("=" * 80)

    if validated.empty:
        print("Nenhum dataset validado.")
        return

    # Apenas datasets que passaram
    experiment = validated[
        validated["status_validation"] == "OK"
    ].copy()

    # ------------------------------------------------------------------------
    # Remover duplicatas
    # ------------------------------------------------------------------------

    experiment = experiment.drop_duplicates(
        subset=["did"]
    )

    # ------------------------------------------------------------------------
    # Ordenação
    #
    # Primeiro balanceamento.
    # Depois célula.
    # Depois quantidade de amostras.
    # ------------------------------------------------------------------------

    experiment = experiment.sort_values(
        by=[
            "classes_bin",
            "samples_bin",
            "features_bin",
            "balance",
            "n_samples",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
    )

    experiment.to_csv(
        EXPERIMENT_FILE,
        index=False,
    )

    print(
        f"\nDatasets válidos para o Experimento_V2: "
        f"{len(experiment)}"
    )

    print(
        f"Células completas encontradas: "
        f"{experiment['cell'].nunique()}"
    )

    return experiment


# ============================================================================
# RELATÓRIO DA GRADE
# ============================================================================

def print_grid_report(experiment):
    """
    Mostra quantos datasets existem em cada célula da grade.
    """

    print("\n")
    print("=" * 80)
    print("RELATÓRIO DA GRADE EXPERIMENTAL")
    print("=" * 80)

    if experiment is None or experiment.empty:

        print("Nenhum dataset disponível.")

        return

    counts = (
        experiment
        .groupby(
            [
                "classes_bin",
                "samples_bin",
                "features_bin",
                "balance",
            ]
        )
        .size()
        .reset_index(
            name="n_datasets"
        )
    )

    # ------------------------------------------------------------------------
    # Mostrar apenas células encontradas
    # ------------------------------------------------------------------------

    print("\nCÉLULAS COM DATASETS:\n")

    for _, row in counts.iterrows():

        print(
            f"classes={row['classes_bin']:>4} | "
            f"samples={row['samples_bin']:>7} | "
            f"features={row['features_bin']:>6} | "
            f"balance={row['balance']} | "
            f"datasets={row['n_datasets']}"
        )

    # ------------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------------

    unique_cells = experiment[
        [
            "classes_bin",
            "samples_bin",
            "features_bin",
        ]
    ].drop_duplicates()

    print("\n")
    print(
        f"Células classe/amostras/features: "
        f"{len(unique_cells)} / 100"
    )

    print(
        f"Datasets válidos: "
        f"{len(experiment)}"
    )

    print(
        f"Balanceados (S): "
        f"{(experiment['balance'] == 'S').sum()}"
    )

    print(
        f"Desbalanceados (N): "
        f"{(experiment['balance'] == 'N').sum()}"
    )


# ============================================================================
# RELATÓRIO DE DATASETS COM ERRO
# ============================================================================

def save_failed_datasets(validated):

    if validated is None or validated.empty:
        return

    failed = validated[
        validated["status_validation"] != "OK"
    ].copy()

    if failed.empty:
        return

    failed.to_csv(
        FAILED_FILE,
        index=False,
    )

    print(
        f"\nDatasets que falharam: "
        f"{len(failed)}"
    )

    print(
        f"Arquivo: {FAILED_FILE}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("FEMa — EXPERIMENTO_V2")
    print("Busca e validação de datasets OpenML")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Buscar candidatos
    # ------------------------------------------------------------------------

    candidates = get_openml_candidates()

    # ------------------------------------------------------------------------
    # 2. Validar datasets
    # ------------------------------------------------------------------------

    validated = validate_all(
        candidates
    )

    # ------------------------------------------------------------------------
    # 3. Gerar dataset final
    # ------------------------------------------------------------------------

    experiment = generate_experiment_dataset(
        validated
    )

    # ------------------------------------------------------------------------
    # 4. Relatório
    # ------------------------------------------------------------------------

    print_grid_report(
        experiment
    )

    # ------------------------------------------------------------------------
    # 5. Falhas
    # ------------------------------------------------------------------------

    save_failed_datasets(
        validated
    )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONCLUÍDO")
    print("=" * 80)

    print(
        f"\nCandidatos:"
        f"  {CANDIDATES_FILE}"
    )

    print(
        f"Validados:"
        f"   {VALIDATED_FILE}"
    )

    print(
        f"Experimento:"
        f" {EXPERIMENT_FILE}"
    )

    print(
        f"\nAgora você pode usar "
        f"{EXPERIMENT_FILE} para selecionar as bases do Experimento_V2."
    )


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()