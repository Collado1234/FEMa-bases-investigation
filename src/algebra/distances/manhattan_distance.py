import numpy as np
from .base_distance import BaseDistance

class ManhattanDistance(BaseDistance):
    """
    Implementa a distância Manhattan (ou distância de blocos) entre dois pontos representados por vetores NumPy.

    A distância Manhattan é a soma das diferenças absolutas das coordenadas correspondentes
    e é frequentemente usada em algoritmos de aprendizado de máquina, clustering e análise de similaridade.

    Herda da classe BaseDistance e fornece a implementação concreta da métrica.
    """

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcula a distância Manhattan entre dois vetores.

        Args:
            x1 (np.ndarray): Primeiro vetor de entrada.
            x2 (np.ndarray): Segundo vetor de entrada.

        Returns:
            float: O valor da distância Manhattan entre x1 e x2.

        Example:
            >>> ManhattanDistance().compute(np.array([0, 0]), np.array([3, 4]))
            7.0
        """
        return np.sum(np.abs(x1 - x2))