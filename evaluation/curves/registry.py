"""
evaluation/curves/registry.py
"""

from typing import Callable, Dict

CurveFunction = Callable[..., dict]

_CURVE_REGISTRY: Dict[str, CurveFunction] = {}


def register_curve(name: str):
    def decorator(func: CurveFunction):
        _CURVE_REGISTRY[name] = func
        return func
    return decorator


def get_curve(name: str) -> CurveFunction:
    return _CURVE_REGISTRY[name]


def list_curves():
    return list(_CURVE_REGISTRY.keys())