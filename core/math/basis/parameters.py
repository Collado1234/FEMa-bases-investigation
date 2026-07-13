from dataclasses import dataclass


@dataclass
class BasisParameters:

    p: float = 2
    epsilon: float = 1
    c: float = 1
    alpha: float = 1
    beta: float = 1
    nu: float = 1
    h: float | None = None