
"""
Procura IDs de datasets no OpenML pelo nome.

Não baixa datasets nem altera arquivos do projeto.
Gera um CSV com os candidatos encontrados.
"""

from pathlib import Path
import re
import pandas as pd
import openml


DATASETS = {
    "haberman": ["haberman"],
    "blood_transfusion_service_center": [
        "blood-transfusion-service-center",
        "bloodtransfusionservicecenter",
        "blood transfusion service center",
    ],
    "monks_problems_1": ["monks-problems-1", "monks problems 1"],
    "monks_problems_2": ["monks-problems-2", "monks problems 2"],
    "climate_model_simulation_crashes": [
        "climate-model-simulation-crashes",
        "climate model simulation crashes",
    ],
    "tic_tac_toe": ["tic-tac-toe", "tic tac toe"],
    "zoo": ["zoo"],
    "mfeat_zernike": ["mfeat-zernike", "mfeat zernike"],
    "mfeat_karhunen": ["mfeat-karhunen", "mfeat karhunen"],
    "mfeat_fourier": ["mfeat-fourier", "mfeat fourier"],
    "mfeat_factors": ["mfeat-factors", "mfeat factors"],
    "mfeat_pixel": ["mfeat-pixel", "mfeat pixel"],
    "gina_agnostic": ["gina-agnostic", "gina agnostic"],
    "musk": ["musk"],
}


def normalize(value):
    """Normaliza nomes para comparação."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def search_dataset(query):
    """Consulta o catálogo OpenML pelo nome."""
    try:
        result = openml.datasets.list_datasets(
            data_name=query,
            output_format="dataframe",
        )

        if result is None or result.empty:
            return []

        return result.reset_index().to_dict("records")

    except Exception as exc:
        print(f"  Erro na consulta '{query}': {exc}")
        return []


def main():
    results = []

    for project_name, aliases in DATASETS.items():
        print(f"\nProcurando: {project_name}")

        candidates = {}

        for alias in aliases:
            print(f"  Consulta: {alias}")

            for item in search_dataset(alias):
                dataset_id = item.get("did")

                if dataset_id is None:
                    continue

                candidates[str(dataset_id)] = item

        if not candidates:
            print("  Nenhum candidato encontrado.")
            results.append({
                "project_name": project_name,
                "openml_id": None,
                "openml_name": None,
                "status": "NAO ENCONTRADO",
            })
            continue

        for dataset_id, item in candidates.items():
            openml_name = item.get("name", "")
            exact = normalize(openml_name) == normalize(project_name)

            print(
                f"  ID: {dataset_id} | "
                f"Nome: {openml_name} | "
                f"Correspondência exata: {exact}"
            )

            results.append({
                "project_name": project_name,
                "openml_id": dataset_id,
                "openml_name": openml_name,
                "status": "CANDIDATO EXATO" if exact else "REVISAR",
            })

    output = Path("data/openml_candidates.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        output,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Resultado salvo em: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
