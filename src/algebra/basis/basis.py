from algebra.distances.base_distance import BaseDistance
from algebra.neighboor_search.base_search import BaseSearch
from algebra.neighboor_search.brute_force import BruteForceSearch
from algebra.distances.euclidean_distance import EuclideanDistance
import numpy as np


class Basis:
    """
    Fábrica de bases de interpolação do FEMa.

    Uso:
        basis = Basis.get('shepard')
        basis.fit(X_train, y_train)
        prediction = basis.predict(sample, k=5, z=2)

        # Ou com configuração customizada:
        basis = Basis.get('radial', distance=ManhattanDistance())
    """

    AVAILABLE = ['shepard', 'radial']

    def __init__(
        self,
        distance: BaseDistance = None,
        search: BaseSearch = None
    ):
        self.distance = distance or EuclideanDistance()
        self.search = search or BruteForceSearch()
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Armazena os dados de treinamento para uso na interpolação.

        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor de targets (n_samples,)
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, sample: np.ndarray, k: int, z: float) -> float:
        """
        Método de interpolação a ser implementado pelas subclasses.

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)
            z:      Parâmetro específico da base (ex: expoente para Shepard)

        Returns:
            Valor interpolado.
        """
        raise NotImplementedError("Subclasses devem implementar o método predict.")

    @staticmethod
    def get(
        name: str,
        distance: BaseDistance = None,
        search: BaseSearch = None
    ) -> 'Basis':
        """
        Retorna a base de interpolação pelo nome.

        Args:
            name:     Nome da base. Opções: 'shepard', 'radial'
            distance: Métrica de distância (padrão: EuclideanDistance)
            search:   Método de busca de vizinhos (padrão: BruteForceSearch)

        Returns:
            Instância da base de interpolação solicitada.

        Raises:
            ValueError: Se o nome da base não for reconhecido.

        Example:
            basis = Basis.get('shepard')
            basis = Basis.get('radial', distance=ManhattanDistance())
        """
        # Import local para evitar dependência circular
        from algebra.basis.shepard import SheppardBasis
        from algebra.basis.radial import RadialBasis

        bases = {
            'sheppard': SheppardBasis,
            'radial':  RadialBasis,
        }

        if name not in bases:
            raise ValueError(
                f"Base '{name}' não reconhecida. "
                f"Opções disponíveis: {Basis.AVAILABLE}"
            )

        distance = distance or EuclideanDistance()
        search = search or BruteForceSearch()

        return bases[name](distance=distance, search=search)