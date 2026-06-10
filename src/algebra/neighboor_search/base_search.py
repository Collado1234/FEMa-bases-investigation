from abc import ABC, abstractmethod

class BaseSearch(ABC):
    @abstractmethod
    def search(self, dists, k):
        """
        Realiza a busca de vizinhos mais próximos.

        Args:
            dists: O vetor de distâncias do ponto de consulta aos vizinhos.
            k: O número de vizinhos mais próximos a serem retornados.

        Returns:
            Uma lista dos índices dos k vizinhos mais próximos ao ponto de consulta.

        Notes:
            Este método deve ser implementado por subclasses concretas, como brute force search, KD-tree, etc.
        """
        pass