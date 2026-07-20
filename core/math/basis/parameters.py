from dataclasses import dataclass
from typing import Optional


@dataclass
class BasisParameters:
    """
    Contêiner único de hiperparâmetros passado a compute_weights de
    qualquer base ("classe coringa").

    Motivação: antes, cada base tinha uma assinatura própria de
    compute_weights (z, epsilon, c, alpha+c, alpha+l, nu, h...), o que
    obrigava quem chama a base (FEMaClassifier/FEMaRegressor, harness de
    validação, camada de tuning) a saber a fórmula de cada base para
    montar os argumentos certos. Com BasisParameters, a assinatura é
    from dataclasses import dataclass
    sempre a mesma: compute_weights(dists, params). Cada base lê, via
    self._require(params) (ver BaseBasis), só os campos que sua fórmula
    usa; os demais ficam ignorados para aquela base — isso é esperado,
    não é um erro.

    Todos os campos são Optional com default None. None significa
    "não configurado". BaseBasis._require() usa isso para detectar cedo,
    com mensagem clara, quando falta um parâmetro que a base realmente
    precisa — em vez de deixar a conta estourar mais adiante com um erro
    numérico difícil de rastrear (ex.: TypeError de NoneType numa
    operação aritmética, ou um resultado silenciosamente errado).

    Reaproveitamento de nomes entre bases é intencional (ex.: `c` é usado
    por Multiquadratic, Inverse Multiquadratic, Logarithmic e Sigmoidal,
    cada uma com um papel matemático diferente na própria fórmula) — o
    nome é só o "slot" de configuração, não implica que as bases
    compartilhem semântica.
    """

    z: Optional[float] = None        # Shepard (expoente), Radial (raio efetivo r0)
    epsilon: Optional[float] = None  # RBF Gaussian, Generalized Exponential, Laplacian, Cauchy
    c: Optional[float] = None        # Multiquadratic, Inverse Multiquadratic, Logarithmic, Sigmoidal
    alpha: Optional[float] = None    # Sigmoidal, Rational Quadratic
    beta: Optional[float] = None     # Softmax Radial, Entropic
    nu: Optional[float] = None       # Harmonic, Student-t
    h: Optional[float] = None        # Wendland C2, Cubic/Quartic Spline, Cosine (raio de suporte compacto)
    l: Optional[float] = None        # Rational Quadratic (lengthscale)
    p: Optional[float] = None        # Generalized Exponential (expoente)
