"""
plot_basis_families.py
=======================

Reproduz o estilo da Fig. 2 do artigo do seu professor (interpolação
com a base de Shepard variando o hiperparâmetro z/"k") só que para
TODAS as bases do seu pacote `basis/`.

Por que o script não importa direto o seu pacote `basis/`?
------------------------------------------------------------
`base_basis.py` importa `from ..neighboor_search import BaseSearch`,
que não estava no zip que você me passou (e além disso a classe base
espera receber um objeto `search` de verdade). Em vez de montar um
mock de `BaseSearch` só pra gerar um gráfico, replico aqui a MESMA
matemática de cada `evaluate()` (copiada linha a linha dos seus
arquivos) e aplico a MESMA normalização feita em
`BaseBasis.compute_weights` (w = phi(d) / sum(phi(d)), com fallback
pra pesos uniformes se soma == 0). Ou seja: o número que sai daqui é
idêntico ao que sairia do seu `compute_weights`, mas sem precisar da
infraestrutura de busca de vizinhos (com só 4 pontos de amostra, todos
os pontos SÃO os "vizinhos", então isso não muda o resultado).

Se depois você quiser plugar o pacote de verdade (com uma
`BaseSearch` real), basta trocar a função `compute_curve()` por uma
chamada a `basis_obj.compute_weights(dists, params)`.

Como usar
---------
$ python plot_basis_families.py
    -> gera um PNG por base em ./figs/<nome_da_base>.png

Num notebook / REPL:
    from plot_basis_families import plot_basis
    plot_basis("shepard")            # gera e MOSTRA (plt.show)
    plot_basis("rbf_gaussian", save_dir="outras_figs")

Para ajustar os valores do hiperparâmetro de cada base (os "k1, k2,
k3" que aparecem nos painéis (a)(b)(c)), edite o dicionário
BASIS_CONFIG lá embaixo — são só listas de números.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DELTA = 1e-10

# ------------------------------------------------------------------
# 1. Dados de amostra (iguais à Fig. 2 do artigo)
# ------------------------------------------------------------------
X_SAMPLES = np.array([-2.0, -1.0, 0.0, 1.0])
Y_SAMPLES = np.array([0.75, 0.0, 1.0, 0.5])

X_MIN, X_MAX = -3.0, 3.0
N_POINTS = 600


def _safe(d: np.ndarray) -> np.ndarray:
    """Evita divisão por zero / log(0), igual ao DELTA do seu pacote."""
    return np.where(d == 0, DELTA, d)


# ------------------------------------------------------------------
# 2. phi(d) cru de cada base — copiado de evaluate() em basis/*.py
#    (NUNCA normalizado aqui; normalização é feita por compute_weights)
# ------------------------------------------------------------------
def phi_shepard(d, z):
    return 1.0 / (_safe(d) ** z)


def phi_radial(d, z):
    r0 = z
    return np.exp(-0.5 * (d / r0) ** 2)


def phi_rbf_gaussian(d, epsilon):
    return np.exp(-(epsilon * _safe(d)) ** 2)


def phi_multiquadratic(d, c):
    return np.sqrt(d ** 2 + c ** 2)


def phi_inverse_multiquadratic(d, c):
    return 1.0 / (np.sqrt(d ** 2 + c ** 2) + DELTA)


def phi_wendland_c2(d, h):
    r = d / h
    return np.where(r <= 1, (1 - r) ** 4 * (4 * r + 1), 0.0)


def phi_cubic_spline(d, h):
    r = d / h
    return np.where(r <= 1, (1 - r) ** 3, 0.0)


def phi_quartic_spline(d, h):
    r = d / h
    return np.where(r <= 1, (1 - r) ** 4, 0.0)


def phi_gen_exponential(d, epsilon, p=2.0):
    return np.exp(-epsilon * (_safe(d) ** p))


def phi_softmax_radial(d, beta):
    return np.exp(-beta * d)


def phi_attention(d):
    return 1.0 / (1.0 + d ** 2)


def phi_logarithmic(d, c):
    denom = np.log(1 + d + c)
    denom = np.where(np.abs(denom) < DELTA, DELTA, denom)
    return 1.0 / denom


def phi_harmonic(d):
    return 1.0 / (1.0 + d)


def phi_laplacian(d, epsilon):
    return np.exp(-epsilon * d)


def phi_cauchy(d, epsilon):
    return 1.0 / (1.0 + (epsilon * d) ** 2)


def phi_student_t(d, nu):
    return (1 + (d ** 2 / nu)) ** (-(nu + 1) / 2)


def phi_cosine(d, h):
    r = d / h
    return np.where(r <= 1, np.cos(np.pi * r / 2) ** 2, 0.0)


def phi_sigmoidal(d, alpha, c=0.0):
    return 1.0 / (1.0 + np.exp(alpha * (d - c)))


def phi_lorentzian(d):
    return 1.0 / (1.0 + _safe(d) ** 4)


def phi_entropic(d, beta):
    # evaluate() do EntropicBasis já normaliza sozinho (softmax de
    # -beta*d^2); aqui devolvo só o numerador cru, pois a normalização
    # final é feita de qualquer forma por compute_curve() abaixo,
    # com o mesmo resultado.
    return np.exp(-beta * d ** 2)


def phi_rational_quadratic(d, alpha, l=1.0):
    return (1 + (d ** 2) / (2 * alpha * l ** 2)) ** (-alpha)


BASIS_FORMULAS = {
    "shepard": phi_shepard,
    "radial": phi_radial,
    "rbf_gaussian": phi_rbf_gaussian,
    "multiquadratic": phi_multiquadratic,
    "inverse_multiquadratic": phi_inverse_multiquadratic,
    "wendland_c2": phi_wendland_c2,
    "cubic_spline": phi_cubic_spline,
    "quartic_spline": phi_quartic_spline,
    "gen_exponential": phi_gen_exponential,
    "softmax_radial": phi_softmax_radial,
    "attention": phi_attention,
    "logarithmic": phi_logarithmic,
    "harmonic": phi_harmonic,
    "laplacian": phi_laplacian,
    "cauchy": phi_cauchy,
    "student_t": phi_student_t,
    "cosine": phi_cosine,
    "sigmoidal": phi_sigmoidal,
    "lorentzian": phi_lorentzian,
    "entropic": phi_entropic,
    "rational_quadratic": phi_rational_quadratic,
}

# ------------------------------------------------------------------
# 3. Configuração de hiperparâmetro por base.
#
#    "param"  -> nome do hiperparâmetro que aparece no eixo/legenda
#                (o análogo ao "k" do Shepard no artigo)
#    "values" -> os 3 valores usados nos painéis (a) (b) (c). Coloquei
#                em ordem "menos localizada -> mais localizada" (igual
#                à progressão do artigo: quanto maior o k, mais a
#                interpolação vira um platô perto de cada amostra e
#                cai bruscamente entre elas). Ajuste à vontade.
#    "fixed"  -> outros parâmetros da fórmula que ficam fixos durante
#                o sweep (ex.: gen_exponential tem epsilon E p; aqui
#                varia-se epsilon e fixa-se p).
#    None     -> base sem hiperparâmetro tunável (PARAMS = () no seu
#                pacote): plota-se 1 painel só, sem variação.
# ------------------------------------------------------------------
BASIS_CONFIG = {
    "shepard": {"param": "z", "values": [1, 3, 5]},
    "radial": {"param": "z (r0)", "values": [1.5, 0.8, 0.4]},
    "rbf_gaussian": {"param": "epsilon", "values": [0.4, 1.0, 2.5]},
    "multiquadratic": {"param": "c", "values": [3.0, 1.0, 0.2]},
    "inverse_multiquadratic": {"param": "c", "values": [2.0, 0.8, 0.2]},
    "wendland_c2": {"param": "h", "values": [3.0, 1.5, 0.7]},
    "cubic_spline": {"param": "h", "values": [3.0, 1.5, 0.7]},
    "quartic_spline": {"param": "h", "values": [3.0, 1.5, 0.7]},
    "gen_exponential": {
        "param": "epsilon",
        "values": [0.3, 1.0, 3.0],
        "fixed": {"p": 2.0},
    },
    "softmax_radial": {"param": "beta", "values": [0.5, 1.5, 4.0]},
    "attention": None,
    "logarithmic": {"param": "c", "values": [2.0, 0.5, 0.05]},
    "harmonic": None,
    "laplacian": {"param": "epsilon", "values": [0.5, 1.5, 4.0]},
    "cauchy": {"param": "epsilon", "values": [0.5, 1.5, 4.0]},
    "student_t": {"param": "nu", "values": [30, 5, 1]},
    "cosine": {"param": "h", "values": [3.0, 1.5, 0.7]},
    "sigmoidal": {
        "param": "alpha",
        "values": [1.0, 3.0, 8.0],
        "fixed": {"c": 0.0},
    },
    "lorentzian": None,
    "entropic": {"param": "beta", "values": [0.5, 1.5, 4.0]},
    "rational_quadratic": {
        "param": "alpha",
        "values": [8.0, 2.0, 0.5],
        "fixed": {"l": 1.0},
    },
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


# ------------------------------------------------------------------
# 4. Núcleo: replica BaseBasis.compute_weights (normalização +
#    fallback p/ pesos uniformes) e faz a interpolação ponto a ponto.
# ------------------------------------------------------------------
def compute_curve(basis_name: str, param_value=None, fixed: dict | None = None):
    formula = BASIS_FORMULAS[basis_name]
    config = BASIS_CONFIG[basis_name]

    kwargs = dict(fixed or {})
    if param_value is not None:
        param_key = config["param"].split(" ")[0]  # "z (r0)" -> "z"
        kwargs[param_key] = param_value

    x_grid = np.linspace(X_MIN, X_MAX, N_POINTS)
    y_grid = np.empty_like(x_grid)

    for i, x in enumerate(x_grid):
        d = np.abs(x - X_SAMPLES)
        phi = np.asarray(formula(d, **kwargs), dtype=float)
        total = phi.sum()
        w = np.full_like(phi, 1.0 / phi.size) if total == 0 else phi / total
        y_grid[i] = np.dot(w, Y_SAMPLES)

    return x_grid, y_grid


# ------------------------------------------------------------------
# 5. Plot no estilo da Fig. 2 (retângulos azuis nos pontos amostrados,
#    curva pontilhada azul, um painel por valor do hiperparâmetro).
# ------------------------------------------------------------------
def plot_basis(basis_name: str, save_dir: str = "figs", show: bool = False):
    if basis_name not in BASIS_FORMULAS:
        raise ValueError(f"Base desconhecida: {basis_name}. Disponíveis: {sorted(BASIS_FORMULAS)}")

    config = BASIS_CONFIG[basis_name]
    fixed = (config or {}).get("fixed")
    values = (config or {}).get("values", [None])
    n_panels = len(values)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6), squeeze=False)
    axes = axes[0]

    for ax, val, label in zip(axes, values, PANEL_LABELS):
        x_grid, y_grid = compute_curve(basis_name, val, fixed)

        ax.plot(x_grid, y_grid, linestyle=":", color="blue", linewidth=1.3)
        ax.plot(
            X_SAMPLES, Y_SAMPLES,
            linestyle="none", marker="s", markersize=5,
            markerfacecolor="blue", markeredgecolor="blue",
        )

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

        subtitle = label if config is None else f"{label} {config['param']} = {val}"
        ax.set_xlabel(subtitle)

    fig.suptitle(f"Base: {basis_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{basis_name}.png"
    fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_all(save_dir: str = "figs"):
    paths = []
    for name in BASIS_FORMULAS:
        paths.append(plot_basis(name, save_dir=save_dir))
    return paths


if __name__ == "__main__":
    saved = plot_all()
    print(f"{len(saved)} figuras salvas em ./figs/:")
    for p in saved:
        print(" -", p)