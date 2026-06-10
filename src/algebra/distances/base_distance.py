# ...existing code...
import numpy as np
from abc import ABC, abstractmethod


class BaseDistance(ABC):
    """
    Classe base abstrata para implementações de métricas de distância.

    Esta interface define o contrato que todas as classes de distância
    devem seguir. Cada métrica concreta deve implementar o método
    `compute`, que recebe dois vetores NumPy e retorna um valor escalar
    representando a distância entre eles.

    Essa abordagem permite trocar facilmente a métrica usada em algoritmos
    de aprendizado de máquina, clustering e similaridade, mantendo uma
    interface consistente.
    """

    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcula a distância entre dois vetores.

        Args:
            x1 (np.ndarray): Primeiro vetor de entrada.
            x2 (np.ndarray): Segundo vetor de entrada.

        Returns:
            float: Valor da distância entre `x1` e `x2`.

        Notes:
            Este método deve ser implementado por subclasses concretas,
            como uma métrica euclidiana, Manhattan, etc.
        """
        pass
# ...existing code...