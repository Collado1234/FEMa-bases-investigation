from typing import Any, Dict

from sklearn.neural_network import MLPClassifier

from models.base import ModelPlugin


class MLPPlugin(ModelPlugin):
    """Baseline de rede neural (stand-in para uma CNN real, mesma interface)."""

    name = "mlp"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        return MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64,))),
            learning_rate_init=params.get("learning_rate", 1e-3),
            alpha=params.get("dropout", 1e-4),
            max_iter=params.get("epochs", 200),
            random_state=random_state,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "dropout": [1e-6, 1e-4, 1e-2],
            "epochs": [200],
        }

    def random_search_space(self) -> Dict[str, Any]:
        return {
            "learning_rate": ("loguniform", 1e-4, 1e-1),
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "dropout": ("loguniform", 1e-6, 1e-2),
            "epochs": ("randint", 50, 300),
        }
