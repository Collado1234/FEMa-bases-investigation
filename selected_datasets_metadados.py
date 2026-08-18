import os
import re

import openml
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAW_DIR = "data/raw"

DATASETS = {
    # ========================================================
    # BAIXA DIMENSIONALIDADE — 2 a 5 features
    # ========================================================

    46799: "AVIDa-hIL6",
    # 3 features | 2 classes | 573k amostras
    # Real | extremamente desbalanceado

    162: "SEA(50000)",
    # 3 features | 2 classes | 1M amostras
    # SINTÉTICO | levemente desbalanceado
    # NÃO ideal para o conjunto final se o objetivo for usar
    # exclusivamente bases reais

    1502: "skin-segmentation",
    # 4 features | 2 classes | 245k amostras
    # Real | desbalanceado

    45927: "Products",
    # 4 features | 5 classes | 73k amostras
    # Real | baixa dimensionalidade | multiclasses

    1509: "walking-activity",
    # 5 features | 22 classes | 149k amostras
    # Real | baixa dimensionalidade | muitas classes

    # ========================================================
    # DIMENSIONALIDADE BAIXA/MÉDIA — 5 a 10 features
    # ========================================================

    1235: "Agrawal1",
    # 9 features | 2 classes | 1M amostras
    # SINTÉTICO | desbalanceado
    # NÃO ideal para o conjunto final se o objetivo for usar
    # exclusivamente bases reais

    42553: "BitcoinHeist_Ransomware",
    # 9 features | 29 classes | 2,9M amostras
    # Real | EXTREMAMENTE desbalanceado
    # Muitas classes | grande volume de dados

    265: "BNG(glass)",
    # 10 features | 7 classes | 138k amostras
    # BNG / gerado | multiclasses

    # ========================================================
    # DIMENSIONALIDADE MÉDIA — 10 a 20 features
    # ========================================================

    46955: "SDSS17",
    # 12 features | 3 classes | 78k amostras
    # Real | multiclasses | dimensão média

    46361: "Risk_Level_Classification",
    # 19 features | 3 classes | 78,6k amostras
    # Real | multiclasses | dimensão média

    # ========================================================
    # DIMENSIONALIDADE MÉDIA/ALTA — 20+ features
    # ========================================================

    # fetal_health:
    # 21 features | 3 classes | 2.126 amostras
    # Real | multiclasses
    # Dataset já existente em data/raw

    # ========================================================
    # BAIXA DIMENSIONALIDADE / MUITAS CLASSES
    # ========================================================

    47264: "US_Inter_State_Migration",
    # 2 features | 54 classes | 500k amostras
    # Real | EXTREMAMENTE desbalanceado
    # Excelente para testar muitas classes com poucas features

    # ========================================================
    # OUTROS / AINDA NÃO CLASSIFICADOS
    # ========================================================

    1169: "airlines",
    # Muitas amostras | baixa/média dimensionalidade
    # Dataset de companhias aéreas
    # Download pode apresentar erro no OpenML

    351: "codrna",
    # Muitas amostras | baixa/média dimensionalidade
    # 2 classes | real
    # Download pode apresentar incompatibilidade no processamento

    46649: "DDXPlus",
    # Dataset grande | múltiplas classes
    # Alta dimensionalidade | real
    # Download bastante pesado
}


# ============================================================
# FUNÇÕES
# ============================================================

def safe_filename(name):
    """
    Converte o nome do dataset em um nome adequado para arquivo.
    """

    name = name.strip()

    name = re.sub(
        r"[^\w\-]+",
        "_",
        name,
        flags=re.UNICODE,
    )

    return name.strip("_").lower()


def calculate_balance(y):
    """
    Calcula:

        IR = maior classe / menor classe

    Balanceado:
        IR <= 1.5

    Desbalanceado:
        IR > 1.5
    """

    counts = pd.Series(y).value_counts()

    if len(counts) < 2:
        raise RuntimeError(
            "Dataset possui menos de duas classes."
        )

    ir = counts.max() / counts.min()

    balance = "S" if ir <= 1.5 else "N"

    return float(ir), balance


def get_metadata_from_csv(filepath, did, expected_name):
    """
    Lê um CSV já existente e reconstrói os metadados.

    Isso evita baixar novamente datasets que já foram
    armazenados em data/raw.
    """

    print()
    print("CSV já existe. Pulando download:")
    print(filepath)

    df = pd.read_csv(filepath)

    if df.shape[1] < 2:
        raise RuntimeError(
            "CSV possui menos de duas colunas."
        )

    # --------------------------------------------------------
    # Target
    #
    # O script original sempre salva o target como última
    # coluna. Portanto, podemos recuperá-lo dessa forma.
    # --------------------------------------------------------

    target = df.columns[-1]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # --------------------------------------------------------
    # Verificações
    # --------------------------------------------------------

    print(f"Amostras    : {X.shape[0]}")
    print(f"Features    : {X.shape[1]}")
    print(f"Classes     : {y.nunique()}")

    missing_x = X.isna().sum().sum()
    missing_y = y.isna().sum()

    if missing_x > 0 or missing_y > 0:
        raise RuntimeError(
            f"Dataset possui missing values: "
            f"X={missing_x}, y={missing_y}"
        )

    # --------------------------------------------------------
    # Features categóricas
    #
    # Como estamos lendo apenas o CSV, não temos mais o
    # categorical_indicator original do OpenML.
    #
    # Fazemos uma estimativa baseada nos tipos das colunas.
    # --------------------------------------------------------

    categorical_count = 0

    for column in X.columns:

        if (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_categorical_dtype(X[column])
            or pd.api.types.is_bool_dtype(X[column])
        ):
            categorical_count += 1

    print(
        f"Features categóricas: {categorical_count}"
    )

    # --------------------------------------------------------
    # Balanceamento
    # --------------------------------------------------------

    ir, balance = calculate_balance(y)

    print(
        f"Imbalance Ratio: {ir:.4f}"
    )

    print(
        f"Balanceamento: {balance}"
    )

    # --------------------------------------------------------
    # Metadados
    # --------------------------------------------------------

    return {
        "did": did,
        "name": expected_name,
        "file": filepath,
        "target": target,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": y.nunique(),
        "imbalance_ratio": ir,
        "balance": balance,
        "categorical_features": categorical_count,
        "status": "EXISTING",
    }


def download_dataset(did, expected_name):

    print()
    print("=" * 80)
    print(f"DID: {did}")
    print(f"Dataset: {expected_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Nome do arquivo
    # --------------------------------------------------------

    filename = safe_filename(
        expected_name
    ) + ".csv"

    filepath = os.path.join(
        RAW_DIR,
        filename,
    )

    # ========================================================
    # VERIFICAR SE O CSV JÁ EXISTE
    # ========================================================

    if os.path.exists(filepath):

        return get_metadata_from_csv(
            filepath,
            did,
            expected_name,
        )

    # ========================================================
    # DOWNLOAD OPENML
    # ========================================================

    print("CSV não encontrado.")
    print("Baixando dataset do OpenML...")

    dataset = openml.datasets.get_dataset(did)

    target = dataset.default_target_attribute

    if target is None:
        raise RuntimeError(
            "Dataset não possui target definido."
        )

    print(f"Nome OpenML : {dataset.name}")
    print(f"Target      : {target}")

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    X, y, categorical_indicator, feature_names = dataset.get_data(
        target=target
    )

    # --------------------------------------------------------
    # Conversão robusta
    # --------------------------------------------------------

    X = pd.DataFrame(X)

    # Alguns datasets/OpenML podem retornar y como lista.
    # A conversão para Series evita problemas com factorize.
    y = pd.Series(
        y,
        name=target,
    )

    # --------------------------------------------------------
    # Verificações
    # --------------------------------------------------------

    print(f"Amostras    : {X.shape[0]}")
    print(f"Features    : {X.shape[1]}")
    print(f"Classes     : {y.nunique()}")

    missing_x = X.isna().sum().sum()
    missing_y = y.isna().sum()

    if missing_x > 0 or missing_y > 0:
        raise RuntimeError(
            f"Dataset possui missing values: "
            f"X={missing_x}, y={missing_y}"
        )

    # --------------------------------------------------------
    # Verificar features categóricas
    # --------------------------------------------------------

    categorical_count = 0

    if categorical_indicator is not None:

        categorical_count = sum(
            bool(x)
            for x in categorical_indicator
        )

    print(
        f"Features categóricas: {categorical_count}"
    )

    # --------------------------------------------------------
    # Balanceamento
    # --------------------------------------------------------

    ir, balance = calculate_balance(y)

    print(
        f"Imbalance Ratio: {ir:.4f}"
    )

    print(
        f"Balanceamento: {balance}"
    )

    # --------------------------------------------------------
    # Target como última coluna
    # --------------------------------------------------------

    df = X.copy()

    df[target] = y.values

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    df.to_csv(
        filepath,
        index=False,
    )

    print()
    print("CSV salvo em:")
    print(filepath)

    # --------------------------------------------------------
    # Metadados
    # --------------------------------------------------------

    return {
        "did": did,
        "name": dataset.name,
        "file": filepath,
        "target": target,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": y.nunique(),
        "imbalance_ratio": ir,
        "balance": balance,
        "categorical_features": categorical_count,
        "status": "OK",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Criar diretório
    # --------------------------------------------------------

    os.makedirs(
        RAW_DIR,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Processar datasets
    # --------------------------------------------------------

    for did, name in DATASETS.items():

        try:

            result = download_dataset(
                did,
                name,
            )

            results.append(result)

        except Exception as e:

            print()
            print("ERRO:")
            print(e)

            results.append({
                "did": did,
                "name": name,
                "status": "ERROR",
                "error": str(e),
            })

    # --------------------------------------------------------
    # Metadados
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        results
    )

    metadata.to_csv(
        "selected_datasets_metadata.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)

    print(
        metadata.to_string(
            index=False
        )
    )

    print()
    print(
        "Metadados salvos em:"
    )

    print(
        "selected_datasets_metadata.csv"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()