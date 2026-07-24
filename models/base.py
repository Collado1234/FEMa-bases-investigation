"""
Contrato de plugin de modelo.

Cada arquivo em models/ (fema.py, logreg.py, mlp.py, ...) deve definir uma
classe que herda de ModelPlugin e implementa os metodos abaixo. O pipeline
NUNCA importa um algoritmo diretamente - ele so fala com essa interface.
Isso e o que garante que trocar de modelo seja so uma mudanca de config
(model_name), igual ao padrao usado no icd-project.

Diferenca em relacao ao icd-project: alem de parameter_grid() (grade
discreta, para grid_search), um plugin pode opcionalmente definir
random_search_space() com distribuicoes continuas (ver tuning/random_search.py),
pois o FEMa e os baselines aqui tem hiperparametros continuos (ex: z do
FEMa, C da regressao logistica) que se beneficiam de random_search.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelPlugin(ABC):
    #: nome usado em configs e no registry (ex: "fema", "logreg", "mlp")
    name: str = "base"

    #: se este modelo suporta predict_proba de forma nativa
    supports_proba: bool = True

    @abstractmethod
    def create_model(self, params: Dict[str, Any], random_state: int):
        """Instancia e retorna um estimator (nao treinado) com os
        hiperparametros fornecidos. `random_state` deve SEMPRE ser repassado
        ao estimator quando ele suportar, para reprodutibilidade."""
        raise NotImplementedError

    @abstractmethod
    def parameter_grid(self) -> Dict[str, list]:
        """Espaco de busca DISCRETO deste modelo, para tuning_strategy=
        'grid_search'. Formato: {nome_do_param: [lista de candidatos]}."""
        raise NotImplementedError

    def random_search_space(self) -> Optional[Dict[str, Any]]:
        """Espaco de busca CONTINUO deste modelo, para tuning_strategy=
        'random_search'. Formato: {nome_do_param: lista_de_escolhas |
        ("uniform"|"loguniform", low, high) | ("randint", low, high)}.

        Por padrao, reaproveita parameter_grid() (so escolhas discretas).
        Plugins com parametros continuos devem sobrescrever este metodo.
        """
        return self.parameter_grid()

    def fit(self, estimator, X, y, X_val=None, y_val=None):
        """Treina o estimator. Comportamento padrao: chama .fit(X, y).
        Plugins podem sobrescrever se precisarem de X_val/y_val (ex: early
        stopping)."""
        estimator.fit(X, y)
        return estimator

    def predict(self, estimator, X):
        return estimator.predict(X)

    def predict_proba(self, estimator, X):
        """Retorna a matriz de probabilidades (n_amostras, n_classes) ou
        None se o modelo nao suportar. O pipeline de metricas trata None
        graciosamente (pulando metricas que dependem de proba, como
        roc_auc)."""
        if not self.supports_proba or not hasattr(estimator, "predict_proba"):
            return None
        return estimator.predict_proba(X)

    def save(self, estimator, path: str) -> None:
        import joblib

        joblib.dump(estimator, path)

    def load(self, path: str):
        import joblib

        return joblib.load(path)
