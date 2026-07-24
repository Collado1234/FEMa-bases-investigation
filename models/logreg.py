from typing import Any, Dict

from sklearn.linear_model import LogisticRegression

from models.base import ModelPlugin


class LogRegPlugin(ModelPlugin):
    """Baseline simples e rapido para validar o pipeline."""

    name = "logreg"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        penalty = params.get("penalty", "l2")
        kwargs = dict(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 500),
            random_state=random_state,
        )
        # so passa `penalty`/`solver` explicitamente quando != default do
        # sklearn, para nao disparar o FutureWarning de deprecacao do
        # parametro `penalty` em versoes recentes do sklearn.
        if penalty != "l2":
            kwargs["penalty"] = penalty
            kwargs["solver"] = "saga"
        return LogisticRegression(**kwargs)

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "max_iter": [500],
        }

    def random_search_space(self) -> Dict[str, Any]:
        return {
            "C": ("loguniform", 1e-3, 1e2),
            "penalty": ["l2"],
            "max_iter": [500],
        }
