from algebra.neighboor_search.base_search import BaseSearch
from algebra.neighboor_search.brute_force import BruteForceSearch


class Basis:
    """
    Fábrica de bases de interpolação do FEMa.

    Uso:
        basis = Basis.get('shepard')
        basis = Basis.get('radial', search=KDTreeSearch())
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
        """
        from algebra.basis.shepard import ShepardBasis
        from algebra.basis.radial import RadialBasis

        bases = {
            'shepard': ShepardBasis,
            'radial':  RadialBasis,
        }

        if name not in bases:
            raise ValueError(
                f"Base '{name}' não reconhecida. "
                f"Opções disponíveis: {Basis.AVAILABLE}"
            )

        search = search or BruteForceSearch()
        return bases[name](search=search)