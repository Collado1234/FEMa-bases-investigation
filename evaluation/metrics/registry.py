"""
evaluation/metrics/registry.py
------------------------------

Registry global de métricas (plugin system).
Permite adicionar métricas sem alterar o evaluator.
"""

from __future__ import annotations

from typing import Callable, Dict, Any

MetricFunction = Callable[..., float | dict]


_METRIC_REGISTRY: Dict[str, MetricFunction] = {}


def register_metric(name: str):
    """
    Decorator para registrar uma métrica no sistema.
    """

    def decorator(func: MetricFunction):
        _METRIC_REGISTRY[name] = func
        return func

    return decorator


def get_metric(name: str) -> MetricFunction:
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Métrica '{name}' não registrada.")
    return _METRIC_REGISTRY[name]


def list_metrics() -> list[str]:
    return list(_METRIC_REGISTRY.keys())


def has_metric(name: str) -> bool:
    return name in _METRIC_REGISTRY


def resolve_metrics(names: list[str]) -> Dict[str, MetricFunction]:
    return {n: get_metric(n) for n in names}