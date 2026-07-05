"""
Public API for evaluation.plots.
"""

from . import classification, common, regression, styles
from .common import _check_binary, _normalize

__all__ = [
    "classification",
    "common",
    "regression",
    "styles",
    "_check_binary",
    "_normalize",
]