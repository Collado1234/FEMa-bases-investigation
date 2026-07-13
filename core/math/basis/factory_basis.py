from .shepard import ShepardBasis
from .radial import RadialBasis
from .rbf_gaussian import RbfGaussianBasis
from .multiquadratic import MultiquadraticBasis
from .inverse_multiquadratic import InverseMultiquadraticBasis
from .wendland_c2 import WendlandC2Basis
from .cubic_spline import CubicSplineBasis
from .quartic_spline import QuarticSplineBasis
from .exp_gen import ExponentialGenBasis

_CLASSES = {
    "shepard": ShepardBasis,
    "radial": RadialBasis,
    "rbf_gaussian": RbfGaussianBasis,
    "multiquadratic": MultiquadraticBasis,
    "inverse_multiquadratic": InverseMultiquadraticBasis,
    "wendland_c2": WendlandC2Basis,
    "cubic_spline": CubicSplineBasis,
    "quartic_spline": QuarticSplineBasis,
    "gen_exponential": ExponentialGenBasis,

}

class Basis:
    @staticmethod
    def get(name, search=None):
        try:
            return _CLASSES[name](search=search)
        except KeyError:
            raise ValueError(f"Base '{name}' não reconhecida.")