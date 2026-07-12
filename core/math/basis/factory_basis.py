from .shepard import ShepardBasis
from .radial import RadialBasis
from .rbf_gaussian import RbfGaussianBasis
from .multiquadratic import MultiquadraticBasis
from .inverse_multiquadratic import InverseMultiquadraticBasis

_CLASSES = {
    "shepard": ShepardBasis,
    "radial": RadialBasis,
    "rbf_gaussian": RbfGaussianBasis,
    "multiquadratic": MultiquadraticBasis,
    "inverse_multiquadratic": InverseMultiquadraticBasis,
}

class Basis:
    @staticmethod
    def get(name, search=None):
        try:
            return _CLASSES[name](search=search)
        except KeyError:
            raise ValueError(f"Base '{name}' não reconhecida.")