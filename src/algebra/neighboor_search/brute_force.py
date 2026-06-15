from abc import ABC
import numpy as np
from .base_search import BaseSearch

class BruteForceSearch(BaseSearch):
    def search(self, dists, k):
        """
        Realiza a busca de vizinhos mais próximos usando uma abordagem de força bruta.

        Args:
            dists: O vetor de distâncias do ponto de consulta aos vizinhos.
            k: O número de vizinhos mais próximos a serem retornados.

        Returns:
            Uma lista dos índices dos k vizinhos mais próximos ao ponto de consulta.

        Notes:
            Esta implementação percorre todos os pontos do conjunto de dados para encontrar os vizinhos mais próximos,
            o que pode ser ineficiente para grandes conjuntos de dados. É recomendado usar esta abordagem apenas para
            conjuntos de dados pequenos ou para fins de teste.
        """
        if k < 0:
            raise ValueError("O número de vizinhos (k) deve ser um inteiro não negativo.")
        elif k == 0:
            return np.arange(len(dists))  # Retorna todos os índices se k for 0 / talvez bugue por não ser len(train_x) mas sim len(dists)
        else:
            return np.argpartition(dists, k)[:k]  # Retorna os índices dos k menores valores de distância