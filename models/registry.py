"""
Registry pattern: mapeia um nome string (usado nas configs YAML) para a
classe de modelo correspondente. Adicionar um novo modelo ao framework é
1) criar o pacote em models/<novo_modelo>/ implementando BaseModel
2) chamar register_model("<nome>", MinhaClasse) uma vez (normalmente no
   __init__.py do próprio pacote do modelo)

Nenhum outro módulo do pipeline precisa ser editado.
"""
from __future__ import annotations

from typing import Dict, Type

from models.base import BaseModel

_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, model_cls: Type[BaseModel]) -> None:
    if not issubclass(model_cls, BaseModel):
        raise TypeError(f"{model_cls} precisa herdar de BaseModel")
    if name in _REGISTRY and _REGISTRY[name] is not model_cls:
        raise ValueError(f"Modelo '{name}' já registrado com outra classe")
    model_cls.name = name
    _REGISTRY[name] = model_cls


def get_model(name: str) -> Type[BaseModel]:
    _ensure_builtin_models_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Modelo '{name}' não registrado. Disponíveis: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    _ensure_builtin_models_loaded()
    return sorted(_REGISTRY)


_loaded = False


def _ensure_builtin_models_loaded() -> None:
    """Importa os pacotes de plugin conhecidos para que se auto-registrem.

    Isso é a ÚNICA lista que precisa ganhar uma linha quando um novo plugin
    é criado. Se um plugin não puder ser importado (ex.: dependência
    ausente, ou bug em código externo do qual ele depende), o erro é
    registrado mas não derruba o restante do framework.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    import importlib
    import logging

    logger = logging.getLogger("models.registry")

    known_plugins = [
        "models.fema",
        "models.sklearn_mlp",
        "models.sklearn_logreg",
    ]
    for module_name in known_plugins:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha de plugin
            logger.warning("Plugin '%s' não pôde ser carregado: %s", module_name, exc)
