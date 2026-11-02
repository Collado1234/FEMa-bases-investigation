
import openml
import pandas as pd
from pathlib import Path

# Dimensões originais: features, amostras, classes.
EXPECTED = {
    "haberman": (3, 306, 2),
    "blood_transfusion_service_center": (4, 748, 2),
    "monks_problems_1": (6, 556, 2),
    "monks_problems_2": (6, 601, 2),
    "climate_model_simulation_crashes": (20, 540, 2),
    "tic_tac_toe": (9, 958, 2),
    "zoo": (16, 101, 7),
    "mfeat_zernike": (47, 2000, 10),
    "mfeat_karhunen": (64, 2000, 10),
    "mfeat_fourier": (76, 2000, 10),
    "mfeat_factors": (216, 2000, 10),
    "mfeat_pixel": (240, 2000, 10),
    "gina_agnostic": (970, 3468, 2),
    "musk": (167, 6598, 2),
}

CANDIDATES = {
    "haberman": [43],
    "blood_transfusion_service_center": [1464, 46913],
    "monks_problems_1": [333],
    "monks_problems_2": [334],
    "climate_model_simulation_crashes": [1467, 40994],
    "tic_tac_toe": [50],
    "zoo": [62, 965, 42667],
    "mfeat_zernike": [22, 995],
    "mfeat_karhunen": [16, 1020],
    "mfeat_fourier": [14, 971],
    "mfeat_factors": [12, 978],
    "mfeat_pixel": [20, 1022, 40979],
    "gina_agnostic": [1038],
    "musk": [1116],
}


def inspect(dataset_id):
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,
    )

    q = ds.qualities or {}

    # OpenML conta o target em NumberOfFeatures.
    total_features = q.get("NumberOfFeatures")
    features = (
        int(total_features) - 1
        if total_features is not None
        else None
    )

    instances = q.get("NumberOfInstances")
    classes = q.get("NumberOfClasses")

    return {
        "openml_id": dataset_id,
        "name": ds.name,
        "version": ds.version,
        "features": features,
        "instances": instances,
        "classes": classes,
        "target": ds.default_target_attribute,
    }


def main():
    rows = []

    for name, ids in CANDIDATES.items():
        expected = EXPECTED[name]

        print(f"\n{name} | esperado: {expected}")

        for dataset_id in ids:
            try:
                info = inspect(dataset_id)

                actual = (
                    info["features"],
                    info["instances"],
                    info["classes"],
                )

                info["project_name"] = name
                info["expected"] = str(expected)
                info["actual"] = str(actual)
                info["compatible"] = actual == expected

                rows.append(info)

                print(
                    f"ID {dataset_id}: {actual} | "
                    f"{'COMPATÍVEL' if actual == expected else 'DIFERENTE'}"
                )

            except Exception as exc:
                print(f"ID {dataset_id}: ERRO: {exc}")

    output = Path("data/openml_validation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output,
        index=False,
        encoding="utf-8",
    )

    print(f"\nResultado salvo em: {output}")


if __name__ == "__main__":
    main()
