"""
Registro de plugins de modelo.

Mapeia nome (string, usado em config) -> instancia de ModelPlugin. Se a biblioteca de um modelo nao estiver
instalada, o plugin simplesmente nao aparece no registry em vez de quebrar
o import de todo o projeto.
"""

from models.fema import FEMaPlugin
from models.logreg import LogRegPlugin
from models.mlp import MLPPlugin

_PLUGINS = {
    "fema": FEMaPlugin(),
    "logreg": LogRegPlugin(),
    "mlp": MLPPlugin(),
}


def get_model_plugin(model_name: str):
    if model_name not in _PLUGINS:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido ou biblioteca nao instalada. "
            f"Disponiveis: {list(_PLUGINS.keys())}"
        )
    return _PLUGINS[model_name]


def available_models():
    return list(_PLUGINS.keys())
