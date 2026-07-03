from ..neighboor_search import BaseSearch, BruteForceSearch

class Basis:
    """
    Fábrica de bases de interpolação do FEMa.

    Uso:
        basis = Basis.get('shepard')
        basis = Basis.get('radial', search=KDTreeSearch())

    Bases disponíveis: 'shepard', 'radial'
    """

    AVAILABLE = ['shepard', 'radial']

    @staticmethod
    def get(name: str, search: BaseSearch = None):
        """
        Retorna a base de interpolação pelo nome.

        Args:
            name:   Nome da base. Opções: 'shepard', 'radial'
            search: Método de busca (padrão: BruteForceSearch)

        Returns:
            Instância da base solicitada.

        Raises:
            ValueError: Se o nome não for reconhecido.

        Example:
            basis = Basis.get('shepard')
            basis = Basis.get('radial', search=KDTreeSearch())
        """