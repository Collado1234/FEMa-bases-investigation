"""
Caracterizacao de datasets via OpenML (versao robusta).

Principais mudancas em relacao ao script original:
1. NUNCA usa `dataset.qualities` (endpoint qualities.xml). Esse endpoint faz
   uma chamada de rede extra, separada do download dos dados, e pode
   travar indefinidamente (foi o que aconteceu no DID 1120: o traceback
   mostra um KeyboardInterrupt dentro do handshake SSL -- ou seja, a
   requisicao simplesmente nunca respondeu).
2. Em vez disso, calcula n_samples / n_features / n_classes diretamente
   dos dados baixados (dataset.get_data(...)), que e o que voce
   realmente usa nos seus experimentos.
3. Define um timeout global de socket, para que QUALQUER chamada de rede
   trave no maximo alguns segundos, em vez de travar para sempre.
4. Isola erros por dataset (um dataset com problema nao derruba o loop
   inteiro) e salva o CSV incrementalmente.
5. Permite resolver o DID a partir do nome quando voce nao tiver certeza
   (usa openml.datasets.list_datasets para buscar por nome).
"""

import socket
import time
import openml
import pandas as pd


# ============================================================
# TIMEOUT GLOBAL (evita travar para sempre em qualquer requisicao)
# ============================================================

socket.setdefaulttimeout(30)  # segundos


# ============================================================
# CONFIGURACAO
#
# Nomes convertidos para snake_case a partir da sua lista de datasets
# ja rodados. DIDs confirmados via OpenML; os marcados com None
# precisam ser confirmados manualmente (nomes ambiguos / genericos
# / muitas versoes com o mesmo nome no OpenML).
# ============================================================

DATASETS = {
    "diabetes": 37,
    "spambase": 44,
    "magic_telescope": 1120,
    "wine": 187,
    "seeds": 1499,
    "vertebra_column": 1523,
    "page_blocks": 30,
    "segment": 36,
    "letter": 6,
    "isolet": 300,
    "haberman": 43,
    "blood_transfusion_service_center": 1464,
    "monks_problems_1": 333,
    "monks_problems_2": 334,
    "climate_model_simulation_crashes": 1467,
    "tic_tac_toe": 50,
    "zoo": 62,
    "mfeat_zernike": 22,
    "mfeat_karhunen": 16,
    "mfeat_fourier": 14,
    "mfeat_factors": 12,
    "mfeat_pixel": 20,
    "gina_agnostic": 1038,
    "musk": 1116,          # MuskVersion2 -- confirme se e essa a versao que voce usou
    "iris": 61,            # confirmado (id classico do OpenML)

    # precisam de confirmacao manual (nome ambiguo ou generico/local):
    "bng_glass": None,          # BNG(glass) e um dataset sintetico enorme,
                                # existem varios DIDs "BNG(...)" no OpenML
    "fetal_health": None,       # dataset do Kaggle (cardiotocography); varias
                                # copias foram subidas ao OpenML por usuarios
                                # diferentes, DID nao e unico/oficial
    "classification_data": None,  # nome generico -- provavelmente nao e um
                                   # dataset publicado no OpenML, e sim um
                                   # arquivo local seu
}


# ============================================================
# CRITERIOS DE CATEGORIZACAO
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


# ============================================================
# RESOLVER DID A PARTIR DO NOME (para os "None" acima)
# ============================================================

def find_did_candidates(name, limit=10):
    """
    Busca candidatos de DID no OpenML a partir de um nome (ou parte dele).
    Uso interativo: rode isso separadamente para descobrir o DID certo
    antes de colocar no dicionario DATASETS.
    """
    try:
        df = openml.datasets.list_datasets(
            data_name=name,
            output_format="dataframe",
        )
        return df.head(limit)
    except Exception as e:
        print(f"[ERRO ao buscar '{name}'] {e}")
        return None


# ============================================================
# OBTEM METADADOS -- SEM DEPENDER DE qualities.xml
# ============================================================

def get_metadata(did, expected_name, max_retries=2):

    print()
    print("=" * 80)
    print(f"DID: {did}")
    print(f"Dataset: {expected_name}")
    print("=" * 80)

    if did is None:
        print("[PULADO] DID nao definido -- confirme manualmente no OpenML.")
        return {
            "did": None,
            "name": expected_name,
            "status": "SKIPPED_NO_DID",
        }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            dataset = openml.datasets.get_dataset(
                did,
                download_data=True,       # baixa os dados de verdade
                download_qualities=False,  # <- evita o endpoint problematico
                download_features_meta_data=False,
            )

            target = dataset.default_target_attribute

            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=target
            )

            n_samples = X.shape[0]
            n_features = X.shape[1]

            if y is not None:
                n_classes = int(y.nunique())
            else:
                n_classes = None

            sample_category = classify_samples(n_samples)
            feature_category = classify_features(n_features)
            class_category = (
                classify_classes(n_classes) if n_classes is not None else None
            )

            result = {
                "did": did,
                "name": expected_name,
                "openml_name": dataset.name,
                "n_samples": n_samples,
                "sample_category": sample_category,
                "n_features": n_features,
                "feature_category": feature_category,
                "n_classes": n_classes,
                "class_category": class_category,
                "target": target,
                "status": "OK",
            }

            print()
            print(f"Amostras : {n_samples}  -> {sample_category}")
            print(f"Features : {n_features}  -> {feature_category}")
            print(f"Classes  : {n_classes}  -> {class_category}")

            return result

        except Exception as e:
            last_error = e
            print(f"[tentativa {attempt}/{max_retries} falhou] {e}")
            time.sleep(2)

    print(f"[ERRO DEFINITIVO] {last_error}")
    return {
        "did": did,
        "name": expected_name,
        "status": "ERROR",
        "error": str(last_error),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CARACTERIZACAO DOS DATASETS VIA OPENML")
    print("=" * 80)

    results = []
    output_file = "datasets_characterization.csv"
    total = len(DATASETS)

    for i, (name, did) in enumerate(DATASETS.items(), start=1):
        print()
        print(f"[{i}/{total}] {name}")

        result = get_metadata(did, name)
        results.append(result)

        # salva incrementalmente -- se travar/crashar no meio,
        # voce nao perde o que ja processou
        pd.DataFrame(results).to_csv(output_file, index=False)

    metadata = pd.DataFrame(results)
    if "n_samples" in metadata.columns:
        metadata = metadata.sort_values(by="n_samples", na_position="last")
    metadata.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("TABELA FINAL")
    print("=" * 80)
    print(metadata.to_string(index=False))

    print()
    print("ARQUIVO GERADO:", output_file)

    pendentes = metadata[metadata["status"] != "OK"]
    if not pendentes.empty:
        print()
        print("=" * 80)
        print("ATENCAO -- precisam de revisao manual:")
        print("=" * 80)
        print(pendentes[["name", "did", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()