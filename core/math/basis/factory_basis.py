from .attention_quadratica      import AttentionQuadraticBasis
from .cauchy_kernel             import CauchyKernelBasis 
from .cosine                    import CosineBasis
from .cubic_spline              import CubicSplineBasis
from .entropic_basis            import EntropicBasis
from .exp_gen                   import ExponentialGenBasis
from .harmonic                  import HarmonicBasis
from .inverse_multiquadratic    import InverseMultiquadraticBasis
from .laplacian_kernel          import LaplacianKernelBasis
from .logarithmic               import LogarithimicBasis
from .lorentzian_basis          import LorentzianBasis
from .multiquadratic            import MultiquadraticBasis
from .quartic_spline            import QuarticSplineBasis
from .radial                    import RadialBasis
from .rational_quadratic        import RationalQuadraticBasis
from .rbf_gaussian              import RbfGaussianBasis
from .shepard                   import ShepardBasis
from .sigmoidal_basis           import SigmoidalBasis 
from .softmax_radial            import SoftmaxRadialBasis
from .student_t                 import StudentTBasis
from .wendland_c2               import WendlandC2Basis 


_CLASSES = {
"attention":                    AttentionQuadraticBasis,
    "cauchy":                   CauchyKernelBasis,
    "cosine":                   CosineBasis,
    "cubic_spline":             CubicSplineBasis,
    "entropic":                 EntropicBasis,
    "gen_exponential":          ExponentialGenBasis,
    "harmonic":                 HarmonicBasis,
    "inverse_multiquadratic":   InverseMultiquadraticBasis,
    "laplacian":                LaplacianKernelBasis,
    "logarithmic":              LogarithimicBasis,
    "lorentzian":               LorentzianBasis,
    "multiquadratic":           MultiquadraticBasis,
    "quartic_spline":           QuarticSplineBasis,
    "radial":                   RadialBasis,
    "rational_quadratic":       RationalQuadraticBasis,
    "rbf_gaussian":             RbfGaussianBasis,    
    "shepard":                  ShepardBasis,
    "sigmoidal":                SigmoidalBasis,
    "softmax_radial":           SoftmaxRadialBasis,            
    "student_t":                StudentTBasis,
    "wendland_c2":              WendlandC2Basis,    
}

class Basis:
    @staticmethod
    def get(name, search=None):
        try:
            return _CLASSES[name](search=search)
        except KeyError:
            raise ValueError(f"Base '{name}' não reconhecida.")