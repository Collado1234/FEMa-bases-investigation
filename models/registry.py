"""
Registro de plugins de BASELINE EXTERNO.

Mapeia nome (string, usado em config) -> instancia de ModelPlugin. Se a
biblioteca de um modelo nao estiver instalada, o plugin simplesmente nao
aparece no registry em vez de quebrar o import de todo o projeto.

IMPORTANTE: o FEMa NAO esta neste registry. Ele nao e' identificado por um
unico nome (e' identificado por um par (contexto, base) - ver
models/fema.py::FEMaPlugin), entao nao cabe no formato "nome -> instancia
fixa" usado aqui. Este registry existe apenas para os baselines externos
(logreg, knn), que servem de referencia metodologica e NAO fazem parte da
comparacao de bases que e' o objeto de estudo do projeto.
"""

from models.knn import KNNPlugin
from models.logreg import LogRegPlugin

_PLUGINS = {
    "logreg": LogRegPlugin(),
    "knn": KNNPlugin(),
}


def get_model_plugin(model_name: str):
    if model_name not in _PLUGINS:
        raise ValueError(
            f"Baseline '{model_name}' desconhecido ou biblioteca nao instalada. "
            f"Disponiveis: {list(_PLUGINS.keys())}"
        )
    return _PLUGINS[model_name]


def available_models():
    return list(_PLUGINS.keys())