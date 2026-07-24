from .shepard                   import ShepardBasis
from .radial                    import RadialBasis
from .rbf_gaussian              import RbfGaussianBasis
from .multiquadratic            import MultiquadraticBasis
from .inverse_multiquadratic    import InverseMultiquadraticBasis
from .wendland_c2               import WendlandC2Basis
from .cubic_spline              import CubicSplineBasis
from .quartic_spline            import QuarticSplineBasis
from .exp_gen                   import ExponentialGenBasis
from .softmax_radial            import SoftmaxRadialBasis
from .attention_quadratica      import AttentionQuadraticBasis
from .logarithmic               import LogarithmicBasis
from .harmonic                  import HarmonicBasis
from .laplacian_kernel          import LaplacianKernelBasis
from .cauchy_kernel             import CauchyKernelBasis
from .student_t                 import StudentTBasis
from .cosine                    import CosineBasis
from .sigmoidal_basis           import SigmoidalBasis
from .lorentzian_basis          import LorentzianBasis
from .entropic_basis            import EntropicBasis
from .rational_quadratic        import RationalQuadraticBasis


_CLASSES = {
    "shepard":                  ShepardBasis,
    "radial":                   RadialBasis,
    "rbf_gaussian":             RbfGaussianBasis,
    "multiquadratic":           MultiquadraticBasis,
    "inverse_multiquadratic":   InverseMultiquadraticBasis,
    "wendland_c2":              WendlandC2Basis,
    "cubic_spline":             CubicSplineBasis,
    "quartic_spline":           QuarticSplineBasis,
    "gen_exponential":          ExponentialGenBasis,
    "softmax_radial":           SoftmaxRadialBasis,
    "attention":                AttentionQuadraticBasis,
    "logarithmic":              LogarithmicBasis,
    "harmonic":                 HarmonicBasis,
    "laplacian":                LaplacianKernelBasis,
    "cauchy":                   CauchyKernelBasis,
    "student_t":                StudentTBasis,
    "cosine":                   CosineBasis,
    "sigmoidal":                SigmoidalBasis,
    "lorentzian":               LorentzianBasis,
    "entropic":                 EntropicBasis,
    "rational_quadratic":       RationalQuadraticBasis,
}

class Basis:
    @staticmethod
    def get(name, search=None):
        try:
            return _CLASSES[name](search=search)
        except KeyError:
            raise ValueError(f"Base '{name}' não reconhecida. Disponíveis: {Basis.available()}")

    @staticmethod
    def available() -> list:
        """
        Retorna os nomes de todas as bases registradas na fábrica.

        Usado, entre outras coisas, pelo harness de validação matemática
        (tests/linear_algebra/math_validation.py) para descobrir
        automaticamente quais bases testar, sem precisar manter uma
        lista duplicada e sujeita a ficar desatualizada.
        """
        return sorted(_CLASSES.keys())