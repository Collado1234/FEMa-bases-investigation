"""Utilitario simples de medicao de tempo de execucao."""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager que mede tempo decorrido em segundos.

    Uso:
        with timer() as t:
            fazer_algo()
        print(t.elapsed_seconds)
    """

    class _Result:
        elapsed_seconds: float = 0.0

    result = _Result()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
