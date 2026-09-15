"""
Caracterizacao de datasets locais (CSV) que nao vieram do OpenML.

Calcula, para cada CSV:
  - n_samples
  - n_features (excluindo a coluna target)
  - n_classes
  - indice de desbalanceamento (IR = classe majoritaria / classe minoritaria)
  - distribuicao de classes (contagem por classe)

Uso:
  1. Ajuste RAW_DIR se seus CSVs nao estiverem em "./raw".
  2. Para cada arquivo, informe o nome da coluna-target em TARGET_COLUMNS.
     Se deixar None, o script tenta adivinhar (nomes comuns / ultima coluna)
     e AVISA qual coluna escolheu -- confira se fez sentido.
"""

import os
import pandas as pd


# ============================================================
# CONFIGURACAO
# ============================================================

RAW_DIR = "data/raw"

# nome do arquivo -> nome da coluna target (None = tenta adivinhar)
TARGET_COLUMNS = {
    "bng_glass.csv": None,
    "fetal_health.csv": "fetal_health",  # essa e a coluna target no dataset
                                          # original do Kaggle; confirme.
}

# candidatos usados na adivinhacao automatica, em ordem de prioridade
COMMON_TARGET_NAMES = [
    "class", "Class", "target", "Target", "label", "Label",
    "y", "Y", "outcome", "Outcome", "Type",
]


# ============================================================
# CRITERIOS DE CATEGORIZACAO (mesmos do script OpenML)
# ============================================================

def classify_samples(n):
    if n < 1000:
        return "Pequena"
    elif n <= 10000:
        return "Media"
    else:
        return "Grande"


def classify_features(n):
    if n <= 20:
        return "Baixa"
    elif n <= 100:
        return "Media"
    else:
        return "Alta"


def classify_classes(n):
    if n <= 3:
        return "Poucas"
    elif n <= 10:
        return "Medias"
    else:
        return "Muitas"


def classify_imbalance(ir):
    """
    IR = contagem da classe majoritaria / contagem da classe minoritaria.

    ~1.0        : balanceado
    1.0 - 1.5   : leve desbalanceamento
    1.5 - 3.0   : moderado
    > 3.0       : severo
    """
    if ir <= 1.5:
        return "Balanceado"
    elif ir <= 3.0:
        return "Moderado"
    else:
        return "Severo"


# ============================================================
# DETECCAO DA COLUNA TARGET
# ============================================================

def guess_target_column(df, filename):
    for name in COMMON_TARGET_NAMES:
        if name in df.columns:
            print(f"[{filename}] target detectado automaticamente: '{name}'")
            return name

    # fallback: ultima coluna
    last_col = df.columns[-1]
    print(
        f"[{filename}] AVISO: nenhum nome comum de target encontrado. "
        f"Usando a ultima coluna como target: '{last_col}'. "
        f"Confira se isso faz sentido para este dataset!"
    )
    return last_col


# ============================================================
# CARACTERIZACAO DE UM CSV
# ============================================================

def characterize_csv(path, filename, target_col=None):

    print()
    print("=" * 80)
    print(f"Arquivo: {filename}")
    print("=" * 80)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERRO ao ler {filename}] {e}")
        return {"name": filename, "status": "ERROR", "error": str(e)}

    print(f"Colunas encontradas: {list(df.columns)}")

    if target_col is None:
        target_col = guess_target_column(df, filename)

    if target_col not in df.columns:
        msg = f"Coluna target '{target_col}' nao existe no arquivo."
        print(f"[ERRO] {msg}")
        return {"name": filename, "status": "ERROR", "error": msg}

    y = df[target_col]
    X = df.drop(columns=[target_col])

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = int(y.nunique())

    class_counts = y.value_counts()
    maj = int(class_counts.max())
    min_ = int(class_counts.min())
    imbalance_ratio = round(maj / min_, 3) if min_ > 0 else None

    sample_category = classify_samples(n_samples)
    feature_category = classify_features(n_features)
    class_category = classify_classes(n_classes)
    imbalance_category = (
        classify_imbalance(imbalance_ratio) if imbalance_ratio is not None else None
    )

    result = {
        "name": filename,
        "target_column": target_col,
        "n_samples": n_samples,
        "sample_category": sample_category,
        "n_features": n_features,
        "feature_category": feature_category,
        "n_classes": n_classes,
        "class_category": class_category,
        "class_distribution": dict(class_counts),
        "imbalance_ratio": imbalance_ratio,
        "imbalance_category": imbalance_category,
        "status": "OK",
    }

    print()
    print(f"Amostras  : {n_samples}  -> {sample_category}")
    print(f"Features  : {n_features}  -> {feature_category}")
    print(f"Classes   : {n_classes}  -> {class_category}")
    print(f"Distribuicao de classes: {dict(class_counts)}")
    print(f"IR (maj/min): {imbalance_ratio}  -> {imbalance_category}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CARACTERIZACAO DE DATASETS LOCAIS (CSV)")
    print("=" * 80)

    results = []

    for filename, target_col in TARGET_COLUMNS.items():
        path = os.path.join(RAW_DIR, filename)

        if not os.path.exists(path):
            print(f"\n[PULADO] Arquivo nao encontrado: {path}")
            results.append({"name": filename, "status": "FILE_NOT_FOUND"})
            continue

        result = characterize_csv(path, filename, target_col)
        results.append(result)

    metadata = pd.DataFrame(results)

    output_file = "local_datasets_characterization.csv"
    metadata.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("TABELA FINAL")
    print("=" * 80)
    cols_to_show = [
        c for c in [
            "name", "n_samples", "sample_category",
            "n_features", "feature_category",
            "n_classes", "class_category",
            "imbalance_ratio", "imbalance_category",
            "status",
        ] if c in metadata.columns
    ]
    print(metadata[cols_to_show].to_string(index=False))

    print()
    print("ARQUIVO GERADO:", output_file)


if __name__ == "__main__":
    main()