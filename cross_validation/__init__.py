"""
Pacote de validação cruzada
"""

from .strategies import iterate_folds, get_cv_splitter

__all__ = [
    "iterate_folds",
    "get_cv_splitter",
]