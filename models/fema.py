"""Plugin do FEMa (Finite Element Machine).

MUDANCA ARQUITETURAL: cada instancia de FEMaPlugin agora representa o FEMa
rodando com UM contexto (classifier|regressor) e UMA base de interpolacao
FIXOS - ex.: FEMaPlugin(context="classifier", basis="shepard"). Antes,
existia um unico FEMaPlugin "generico" e a base era só mais um valor
dentro do espaco de hiperparametros buscado (basis_function), misturada
com k, z, epsilon etc. Isso fazia com que o ranking final do summary
misturasse combinacoes de bases DIFERENTES (ver
persistence/summary_builder.py) - o que é conceitualmente errado para um
projeto cujo objeto de estudo e' comparar bases entre si, não encontrar a
"melhor combinação global" entre todas.

Com a base fixada na instancia:
  - cada execucao do pipeline (pipeline/run_model.py::run_basis_experiment)
    já e' inerentemente escopada a uma unica base;
  - parameter_grid()/random_search_space() voltam a ser um unico
    Dict[str, list] "achatado" (k + os parametros PROPRIOS daquela base),
    sem precisar da ramificacao condicional List[Dict[str, list]] que
    existia so' para evitar produto cartesiano entre bases dentro de um
    unico plugin "fema" (tuning/grid_search.py e tuning/random_search.py
    continuam suportando o formato de lista por compatibilidade com
    outros plugins que um dia possam precisar, mas o FEMa não usa mais).

PARAM_GRIDS/RANDOM_SEARCH_SPACES continuam centralizados aqui (nao dentro
de core/math/basis/) por decisao deliberada: são faixas de busca de
EXPERIMENTO (decisão de quem desenha o experimento), não uma propriedade
matemática da base (que pertence a core/). A validação cruzada com
basis.PARAMS/OPTIONAL_PARAMS agora acontece na CONSTRUÇÃO de cada
FEMaPlugin (_validate_basis_entry), não mais de uma vez para todas as
bases no import do modulo.
"""
from typing import Any, Dict, List

import numpy as np

from core import (
    Basis,
    BasisParameters,
    EuclideanDistance,
    BruteForceSearch,
    FEMaClassifier,
    FEMaRegressor,
)
from models.base import ModelPlugin

# Grade para grid_search: basis_name -> {campo: [valores]}. Faixas
# calibradas a partir da varredura de leakage em
# tests/linear_algebra/math_validation.py (SWEEP_CONFIG lá resolve o
# mesmo tipo de pergunta: "que faixa faz sentido para este parâmetro
# nesta fórmula"). Ajuste livremente — é decisão de experimento, não é
# propriedade de core/.
PARAM_GRIDS: Dict[str, Dict[str, List[float]]] = {
    "shepard": {"z": [1.0, 2.0, 3.0, 4.0]},
    "radial": {"z": [0.05, 0.25, 1.0, 2.0, 4.0]},
    "rbf_gaussian": {"epsilon": [0.1, 0.5, 1.0, 5.0, 20.0]},
    "multiquadratic": {"c": [0.1, 0.5, 1.0, 2.0, 5.0]},
    "inverse_multiquadratic": {"c": [0.01, 0.1, 0.5, 1.0, 2.0]},
    "wendland_c2": {"h": [0.5, 1.0, 2.0, 3.0, 5.0]},
    "cubic_spline": {"h": [0.5, 1.0, 2.0, 3.0, 5.0]},
    "quartic_spline": {"h": [0.5, 1.0, 2.0, 3.0, 5.0]},
    "gen_exponential": {"epsilon": [0.1, 1.0, 5.0, 20.0], "p": [1.0, 2.0, 3.0]},
    "softmax_radial": {"beta": [0.1, 1.0, 5.0, 20.0]},
    "attention": {},
    "logarithmic": {"c": [0.01, 0.1, 0.5, 1.0]},
    "harmonic": {"nu": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "laplacian": {"epsilon": [0.1, 1.0, 5.0, 20.0]},
    "cauchy": {"epsilon": [0.1, 1.0, 5.0, 20.0]},
    "student_t": {"nu": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "cosine": {"h": [0.5, 1.0, 2.0, 3.0, 5.0]},
    "sigmoidal": {"alpha": [0.5, 1.0, 5.0, 20.0], "c": [0.5, 1.0, 2.0]},
    "lorentzian": {},
    "entropic": {"beta": [0.1, 1.0, 5.0, 20.0]},
    "rational_quadratic": {"alpha": [0.5, 1.0, 2.0], "l": [0.01, 0.1, 1.0, 2.0]},
}

RANDOM_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "shepard": {"z": ("uniform", 0.5, 5.0)},
    "radial": {"z": ("loguniform", 0.01, 5.0)},
    "rbf_gaussian": {"epsilon": ("loguniform", 0.1, 50.0)},
    "multiquadratic": {"c": ("uniform", 0.1, 5.0)},
    "inverse_multiquadratic": {"c": ("loguniform", 0.01, 2.0)},
    "wendland_c2": {"h": ("uniform", 0.5, 5.0)},
    "cubic_spline": {"h": ("uniform", 0.5, 5.0)},
    "quartic_spline": {"h": ("uniform", 0.5, 5.0)},
    "gen_exponential": {"epsilon": ("loguniform", 0.1, 50.0), "p": ("uniform", 1.0, 3.0)},
    "softmax_radial": {"beta": ("loguniform", 0.1, 50.0)},
    "attention": {},
    "logarithmic": {"c": ("loguniform", 0.01, 1.0)},
    "harmonic": {"nu": ("loguniform", 0.0001, 1.0)},
    "laplacian": {"epsilon": ("loguniform", 0.1, 50.0)},
    "cauchy": {"epsilon": ("loguniform", 0.1, 50.0)},
    "student_t": {"nu": ("loguniform", 0.0001, 1.0)},
    "cosine": {"h": ("uniform", 0.5, 5.0)},
    "sigmoidal": {"alpha": ("loguniform", 0.5, 20.0), "c": ("uniform", 0.5, 2.0)},
    "lorentzian": {},
    "entropic": {"beta": ("loguniform", 0.1, 50.0)},
    "rational_quadratic": {"alpha": ("uniform", 0.5, 2.0), "l": ("loguniform", 0.01, 2.0)},
}

_K_GRID = [5, 10, 15, 20]
_K_SPACE = ("randint", 3, 30)

_CONTEXT_CLASSES = {
    "classifier": FEMaClassifier,
    "regressor": FEMaRegressor,
}


def _validate_basis_entry(basis_name: str) -> None:
    """Confere que PARAM_GRIDS/RANDOM_SEARCH_SPACES tem uma entrada para
    `basis_name` cobrindo exatamente os campos que ela declara
    (basis.PARAMS | basis.OPTIONAL_PARAMS)."""
    search = BruteForceSearch(metric=EuclideanDistance())
    basis = Basis.get(basis_name, search=search)
    required = set(basis.PARAMS) | set(basis.OPTIONAL_PARAMS)

    for grid_name, grids in (("PARAM_GRIDS", PARAM_GRIDS), ("RANDOM_SEARCH_SPACES", RANDOM_SEARCH_SPACES)):
        if basis_name not in grids:
            raise ValueError(
                f"Base '{basis_name}' esta registrada em core mas nao tem entrada em "
                f"{grid_name} (models/fema.py). Adicione uma entrada, mesmo que "
                f"vazia ({{}}) para bases sem parametro de escala."
            )
        provided = set(grids[basis_name].keys())
        if provided != required:
            raise ValueError(
                f"{grid_name}['{basis_name}'] declara os campos {sorted(provided)}, mas "
                f"{type(basis).__name__}.PARAMS|OPTIONAL_PARAMS = {sorted(required)} — "
                f"eles precisam bater exatamente. Confira core/math/basis/{basis_name}.py "
                f"e models/fema.py."
            )


class _FEMaEstimator:
    """Estimator "sklearn-like" que encapsula o ciclo fit/predict do FEMa,
    para caber no contrato generico ModelPlugin.create_model/fit/predict."""

    def __init__(self, context: str, basis_function: str, k: int, random_state: int = 42, **basis_kwargs: float):
        self.context = context
        self.basis_function = basis_function
        self.k = k
        self.random_state = random_state
        self.basis_kwargs = basis_kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        search = BruteForceSearch(metric=EuclideanDistance())
        basis = Basis.get(self.basis_function, search=search)
        model_cls = _CONTEXT_CLASSES[self.context]
        self._model = model_cls(basis=basis, search=search)
        self._model.fit(X, y)
        return self

    def _params(self) -> BasisParameters:
        return BasisParameters(**self.basis_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = self._model.predict(X, k=self.k, params=self._params())
        if self.context == "classifier":
            labels, _ = result
            return labels
        return result  # FEMaRegressor.predict ja' retorna so' o vetor de previsoes

    def predict_proba(self, X: np.ndarray):
        if self.context != "classifier":
            return None
        _, probs = self._model.predict(X, k=self.k, params=self._params())
        return probs


class FEMaPlugin(ModelPlugin):
    """Uma instancia = um (context, basis) fixo.

    Uso:
        plugin = FEMaPlugin(context="classifier", basis="shepard")

    k e os parametros proprios da base (z, epsilon, c, h, nu, beta, ...)
    continuam sendo hiperparametros tunados normalmente via
    parameter_grid()/random_search_space() - a unica mudanca e' que agora
    esses metodos devolvem um Dict[str, list] simples (so' desta base),
    nao mais uma lista de ramificacoes cobrindo todas as bases.
    """

    def __init__(self, context: str, basis: str):
        if context not in _CONTEXT_CLASSES:
            raise ValueError(f"context invalido: '{context}'. Use 'classifier' ou 'regressor'.")
        if basis not in Basis.available():
            raise ValueError(f"Base '{basis}' nao registrada. Disponiveis: {Basis.available()}")
        _validate_basis_entry(basis)

        self.context = context
        self.basis = basis
        self.name = f"fema:{context}:{basis}"
        self.supports_proba = context == "classifier"

    def create_model(self, params: Dict[str, Any], random_state: int):
        k = params.get("k", 5)
        basis_kwargs = {key: value for key, value in params.items() if key != "k"}
        return _FEMaEstimator(
            context=self.context, basis_function=self.basis, k=k, random_state=random_state, **basis_kwargs
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {"k": _K_GRID, **PARAM_GRIDS[self.basis]}

    def random_search_space(self) -> Dict[str, Any]:
        return {"k": _K_SPACE, **RANDOM_SEARCH_SPACES[self.basis]}


def available_bases() -> List[str]:
    """Lista as bases registradas em core (core.Basis.available()) - usado
    por pipeline.run_model.run_all_bases e por reporting/compare_bases.py
    para nao duplicar essa lista em outro lugar do projeto."""
    return Basis.available()