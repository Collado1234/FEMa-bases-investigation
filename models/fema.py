"""Plugin do FEMa (Finite Element Machine) - modelo principal do projeto.

Wrapper fino sobre core.FEMaClassifier / core.Basis. Nenhum codigo dentro
de core/ e alterado aqui - este plugin so orquestra o que ja existe la
(search + basis + classifier) atras do contrato ModelPlugin.

basis_function/k/<parametros da base> sao hiperparametros de INFERENCIA do
FEMa (nao existe "treino" no sentido classico: fit() so indexa X_train).
Aqui eles viram hiperparametros normais do modelo, para poderem ser
tunados como qualquer outro (grid_search ou random_search).

ESPAÇO DE BUSCA CONDICIONAL: cada base usa um subconjunto diferente de
campos de BasisParameters (basis.PARAMS/OPTIONAL_PARAMS — ver
core/math/basis/base_basis.py), então parameter_grid()/random_search_space()
não retornam um único Dict[str, list] "achatado" (isso faria produto
cartesiano de TODOS os campos de TODAS as bases, gerando combinações
inválidas/redundantes — ex.: 'epsilon' combinado com basis_function=
'shepard', que nunca usa epsilon). Em vez disso retornam List[Dict[str,
list]], uma ramificação por base — suportado por tuning/grid_search.py e
tuning/random_search.py (expandem/amostram cada ramificação separadamente,
sem produto cruzado entre elas).

PARAM_GRIDS/RANDOM_SEARCH_SPACES são validados contra basis.PARAMS |
basis.OPTIONAL_PARAMS no import deste módulo (_validate_param_grids): se
uma base nova for registrada em core/math/basis/factory_basis.py e
esquecerem de cadastrar a faixa de busca aqui, o erro aparece na hora de
importar o plugin, não no meio de um tuning de horas.
"""
from typing import Any, Dict, List

import numpy as np

from core import Basis, BasisParameters, EuclideanDistance, BruteForceSearch, FEMaClassifier
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


def _validate_param_grids() -> None:
    """Confere, para toda base registrada em core (Basis.available()),
    que PARAM_GRIDS e RANDOM_SEARCH_SPACES cobrem exatamente os campos
    que ela declara (basis.PARAMS | basis.OPTIONAL_PARAMS)."""
    search = BruteForceSearch(metric=EuclideanDistance())

    for name in Basis.available():
        basis = Basis.get(name, search=search)
        required = set(basis.PARAMS) | set(basis.OPTIONAL_PARAMS)

        for grid_name, grids in (("PARAM_GRIDS", PARAM_GRIDS), ("RANDOM_SEARCH_SPACES", RANDOM_SEARCH_SPACES)):
            if name not in grids:
                raise ValueError(
                    f"Base '{name}' está registrada em core mas não tem entrada em "
                    f"{grid_name} (models/fema.py). Adicione uma entrada, mesmo que "
                    f"vazia ({{}}) para bases sem parâmetro de escala."
                )
            provided = set(grids[name].keys())
            if provided != required:
                raise ValueError(
                    f"{grid_name}['{name}'] declara os campos {sorted(provided)}, mas "
                    f"{type(basis).__name__}.PARAMS|OPTIONAL_PARAMS = {sorted(required)} — "
                    f"eles precisam bater exatamente. Confira core/math/basis/{name}.py "
                    f"e models/fema.py."
                )


_validate_param_grids()


class _FEMaEstimator:
    """Estimator "sklearn-like" que encapsula o ciclo fit/predict do FEMa,
    para caber no contrato genérico ModelPlugin.create_model/fit/predict."""

    def __init__(self, basis_function: str, k: int, random_state: int = 42, **basis_kwargs: float):
        """basis_kwargs: hiperparâmetros da base escolhida (mesmos nomes
        de campo de BasisParameters: z, epsilon, c, alpha, l, h, nu,
        beta, p). O que não for passado fica None; se a base precisar de
        um campo ausente, BaseBasis._require levanta erro claro no
        predict (ver core/math/basis/base_basis.py)."""
        self.basis_function = basis_function
        self.k = k
        self.random_state = random_state
        self.basis_kwargs = basis_kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        search = BruteForceSearch(metric=EuclideanDistance())
        basis = Basis.get(self.basis_function, search=search)
        self._model = FEMaClassifier(basis=basis, search=search)
        self._model.fit(X, y)
        return self

    def _params(self) -> BasisParameters:
        return BasisParameters(**self.basis_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels, _ = self._model.predict(X, k=self.k, params=self._params())
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = self._model.predict(X, k=self.k, params=self._params())
        return probs


class FEMaPlugin(ModelPlugin):
    name = "fema"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        basis_function = params.get("basis_function", "shepard")
        k = params.get("k", 5)
        basis_kwargs = {key: value for key, value in params.items() if key not in ("basis_function", "k")}

        return _FEMaEstimator(basis_function=basis_function, k=k, random_state=random_state, **basis_kwargs)

    def parameter_grid(self) -> List[Dict[str, list]]:
        """Uma ramificação por base — ver docstring do módulo sobre por
        que isso não é um único Dict[str, list] achatado."""
        return [
            {"basis_function": [name], "k": _K_GRID, **PARAM_GRIDS[name]}
            for name in Basis.available()
        ]

    def random_search_space(self) -> List[Dict[str, Any]]:
        """Mesma ideia do parameter_grid, para random_search."""
        return [
            {"basis_function": [name], "k": _K_SPACE, **RANDOM_SEARCH_SPACES[name]}
            for name in Basis.available()
        ]
