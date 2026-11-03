
"""
Download dos datasets ausentes do protocolo FEMa.

- Obtém os dados do OpenML.
- Preserva os nomes de arquivos e targets do registry.py.
- Valida as dimensões originais.
- Não sobrescreve arquivos existentes.
"""

from pathlib import Path
import pandas as pd
import openml


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

# nome lógico: (OpenML ID, arquivo CSV, target esperado,
#               features, amostras, classes)
DATASETS = {
    "haberman": (
        43, "haberman.csv", "Survival_status",
        3, 306, 2,
    ),
    "blood_transfusion_service_center": (
        1464, "bloodtransfusionservicecenter.csv", "Class",
        4, 748, 2,
    ),
    "monks_problems_1": (
        333, "monksproblems1.csv", "class",
        6, 556, 2,
    ),
    "monks_problems_2": (
        334, "monksproblems2.csv", "class",
        6, 601, 2,
    ),
    "climate_model_simulation_crashes": (
        1467, "climatemodelsimulationcrashes.csv", "Class",
        20, 540, 2,
    ),
    "tic_tac_toe": (
        50, "tictactoe.csv", "Class",
        9, 958, 2,
    ),
    "zoo": (
        62, "zoo.csv", "type",
        16, 101, 7,
    ),
    "mfeat_zernike": (
        22, "mfeatzernike.csv", "class",
        47, 2000, 10,
    ),
    "mfeat_karhunen": (
        16, "mfeatkarhunen.csv", "class",
        64, 2000, 10,
    ),
    "mfeat_fourier": (
        14, "mfeatfourier.csv", "class",
        76, 2000, 10,
    ),
    "mfeat_factors": (
        12, "mfeatfactors.csv", "class",
        216, 2000, 10,
    ),
    "mfeat_pixel": (
        20, "mfeatpixel.csv", "class",
        240, 2000, 10,
    ),
    "gina_agnostic": (
        1038, "ginaagnostic.csv", "label",
        970, 3468, 2,
    ),
    "musk": (
        1116, "muskversion2.csv", "class",
        167, 6598, 2,
    ),
}


def download_one(name, metadata):
    dataset_id, filename, target_column, expected_features, expected_rows, expected_classes = metadata

    destination = DATA_RAW / filename

    # Preserva arquivos já existentes.
    if destination.exists():
        print(f"[PULADO] {name}: arquivo já existe.")
        return "PULADO"

    print(f"\n[DOWNLOAD] {name} | OpenML ID {dataset_id}")

    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,
    )

    source_target = dataset.default_target_attribute

    if not source_target:
        raise ValueError(
            f"{name}: OpenML não informou um target padrão."
        )

    print(f"  Target OpenML: {source_target}")
    print(f"  Target no projeto: {target_column}")

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=source_target,
    )

    if y is None:
        raise ValueError(f"{name}: target não foi carregado.")

    # Verifica as dimensões antes de salvar.
    actual_features = X.shape[1]
    actual_rows = X.shape[0]
    actual_classes = pd.Series(y).nunique()

    actual = (
        actual_features,
        actual_rows,
        actual_classes,
    )

    expected = (
        expected_features,
        expected_rows,
        expected_classes,
    )

    print(f"  Dimensões obtidas: {actual}")
    print(f"  Dimensões esperadas: {expected}")

    if actual != expected:
        raise ValueError(
            f"{name}: dimensões incompatíveis. "
            f"Esperado {expected}, obtido {actual}. "
            "Arquivo não será salvo."
        )

    # Reconstrói o CSV com o nome de target esperado pelo registry.
    df = X.copy()
    df[target_column] = pd.Series(
        y.to_numpy() if hasattr(y, "to_numpy") else y,
        index=df.index,
    )

    if target_column in X.columns:
        raise ValueError(
            f"{name}: target já existe entre as features."
        )

    # Escrita temporária para evitar CSV incompleto.
    temporary = destination.with_suffix(".csv.tmp")

    try:
        df.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
        )

        # Confirma que o CSV salvo pode ser lido.
        check = pd.read_csv(temporary)

        if check.shape != df.shape:
            raise ValueError(
                f"{name}: dimensões do CSV salvo divergiram."
            )

        temporary.replace(destination)

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"  [OK] Salvo em: {destination}")

    return "BAIXADO"


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 65)
    print("DOWNLOAD DOS DATASETS OPENML")
    print("=" * 65)

    for name, metadata in DATASETS.items():
        try:
            status = download_one(name, metadata)

        except Exception as exc:
            status = "FALHOU"
            print(f"\n[ERRO] {name}: {exc}")

        results.append({
            "dataset": name,
            "openml_id": metadata[0],
            "arquivo": metadata[1],
            "status": status,
        })

    report = pd.DataFrame(results)

    report_path = ROOT / "data" / "openml_download_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("RESUMO")
    print("=" * 65)
    print(report.to_string(index=False))

    print(f"\nRelatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
