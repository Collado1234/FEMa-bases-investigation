from typing import Any, Dict

from sklearn.neighbors import KNeighborsClassifier

from models.base import ModelPlugin


class KNNPlugin(ModelPlugin):
    """Baseline externo (k-Vizinhos Mais Proximos).

    NAO faz parte do objeto de estudo do projeto (comparacao de bases de
    interpolacao do FEMa) - serve apenas como referencia metodologica
    simples e rapida, mantida isolada em
    results/external_baselines/knn/<dataset>/<experiment_name>/.
    """

    name = "knn"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        # KNeighborsClassifier nao aceita random_state (e determinístico
        # dado o dataset, exceto empates em weights='distance', que o
        # sklearn resolve de forma estavel).
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            p=params.get("p", 2),
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "n_neighbors": [3, 5, 7, 10, 15, 20],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }

    def random_search_space(self) -> Dict[str, Any]:
        return {
            "n_neighbors": ("randint", 3, 30),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }