from ..neighboor_search import BaseSearch, BruteForceSearch

class Basis:
    """
    Fábrica de bases de interpolação do FEMa.

    Uso:
        basis = Basis.get('shepard')
        basis = Basis.get('radial', search=KDTreeSearch())

    Bases disponíveis: 'shepard', 'radial', 'rbf_gaussian'
    """

    AVAILABLE = ['shepard', 'radial', 'rbf_gaussian']

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
        if name == 'shepard':
            from .shepard import ShepardBasis
            return ShepardBasis(search=search)
        elif name == 'radial':
            from .radial import RadialBasis
            return RadialBasis(search=search)
        elif name == 'rbf_gaussian':
            from .rbf_gaussian import RbfGaussianBasis
            return RbfGaussianBasis(search=search)
        elif name == "multiquadratic":
            from .multiquadratic import MultiquadraticBasis
            return MultiquadraticBasis(search=search)
        elif name == "inverse_multiquadratic":
            from .inverse_multiquadratic import InverseMultiquadraticBasis
            return InverseMultiquadraticBasis(search=search)
        
        