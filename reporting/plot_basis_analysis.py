"""
plot_basis_analysis.py
========================

Ponto ÚNICO de análise visual das funções de base do FEMa. SUBSTITUI
plot_basis_families.py e plot_fema_mechanism.py -- não importa nada
deles (não depende que eles existam). Toda a matemática (fórmulas
phi(d), normalização, configuração de hiperparâmetro por base) mora
AQUI, numa fonte única de verdade: BASIS_CONFIG e BASIS_FORMULAS.

Por que isso importa na prática: se você adicionar um novo valor de
hiperparâmetro (ou um novo intervalo) em BASIS_CONFIG, ou mudar
DEFAULT_K_VALUES, isso se propaga AUTOMATICAMENTE para os dois tipos de
imagem abaixo -- não tem mais um script cuidando da curva 1D e outro
cuidando do mapa 2D com configurações que podem ficar dessincronizadas.

Duas famílias de imagem, as DUAS em formato de MATRIZ (linhas = k,
colunas = hiperparâmetro), para que sejam diretamente comparáveis:

  1. curve_grid.png        -- interpolação 1D (4 amostras, estilo
                               Fig. 2 do artigo), variando quantos
                               vizinhos k entram na interpolação E o
                               hiperparâmetro próprio da base.
  2. probability_grid.png  -- classificação 2D (mapa de probabilidade/
                               certeza, estilo Fig. 4+5 combinadas),
                               variando k e o hiperparâmetro.

Lendo uma LINHA da matriz: k fixo, hiperparâmetro variando.
Lendo uma COLUNA da matriz: hiperparâmetro fixo, k variando.

Como usar
---------
$ python -m reporting.plot_basis_analysis
    -> gera figs/<basis>/curve_grid.png e figs/<basis>/probability_grid.png
       para TODAS as bases de BASIS_FORMULAS.

Num notebook:
    from reporting.plot_basis_analysis import plot_curve_grid, plot_probability_grid
    plot_curve_grid("shepard", k_values=(1, 2, 3, 4))
    plot_probability_grid("rbf_gaussian", k_values=(1, 3, 5, 9))
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

DELTA = 1e-10


# ==========================================================================
# 1. FÓRMULAS phi(d) cruas de cada base — fonte única (não normalizadas
#    aqui; a normalização é feita por _normalize_rows, sempre a mesma
#    para o caso 1D e o caso 2D).
# ==========================================================================
def _safe(d):
    return np.where(d == 0, DELTA, d)


def phi_shepard(d, z):
    return 1.0 / (_safe(d) ** z)


def phi_radial(d, z):
    return np.exp(-0.5 * (d / z) ** 2)


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


# ==========================================================================
# 2. CONFIGURAÇÃO DE HIPERPARÂMETRO POR BASE — fonte única. Editar aqui
#    é o ÚNICO lugar necessário; curve_grid E probability_grid usam,
#    por padrão, EXATAMENTE os valores da lista "values" como colunas
#    da matriz (na ordem em que você escreveu -- não são regenerados
#    nem reordenados). Se quiser um leque mais denso/exploratório em
#    vez dos valores curados manualmente, use param_range() e passe o
#    resultado explicitamente em param_values=.
#
#    "param"  -> nome do hiperparâmetro (aparece no eixo/legenda)
#    "values" -> os valores EXATOS usados como colunas da matriz
#    "fixed"  -> outros parâmetros da fórmula que ficam fixos
#    None     -> base sem hiperparâmetro tunável
#
#    Valores curados a partir de tests.linear_algebra.validation (busca
#    do leakage minimo por base -- "leakage" = 1 - peso no vizinho a
#    distancia 0; leakage=0 => interpolacao exata). Onde a validação
#    encontrou um ótimo que ainda não estava na lista, foi adicionado
#    explicitamente (comentado abaixo, por base).
# ==========================================================================
BASIS_CONFIG = {
    "attention": None,
    "cauchy": {"param": "epsilon", "values": [0.5, 1.5, 4.0, 7.0, 15.0, 50.0]},
    "cosine": {"param": "h", "values": [3.0, 1.5, 0.7, 0.5, 0.1, 0.05]},
    "cubic_spline": {"param": "h", "values": [3.0, 1.5, 0.7, 0.5, 0.1]},
    "entropic": {"param": "beta", "values": [0.5, 1.5, 4.0, 20, 30, 38.59, 50]},
    "gen_exponential": {"param": "epsilon", "values": [0.3, 1.0, 3.0, 5.0, 20.0, 38.59, 50], "fixed": {"p": 2.0}},
    "harmonic": None,
    "inverse_multiquadratic": {"param": "c", "values": [2.0, 0.8, 0.2, 0.1, 0.01, 0.001]},
    "laplacian": {"param": "epsilon", "values": [0.5, 1.5, 4.0, 20, 30, 38.59, 50]},
    "logarithmic": {"param": "c", "values": [2.0, 0.5, 0.05, 0.01, 0.001]},
    "lorentzian": None,
    # validacao achou o melhor leakage (0.7654, o menor possivel pra essa
    # base) em c=5 -- fora do range original [3.0, 1.0, 0.2]. Adicionado.
    "multiquadratic": {"param": "c", "values": [5.0, 3.0, 1.0, 0.2]},
    "quartic_spline": {"param": "h", "values": [3.0, 1.5, 0.7, 0.5, 0.1]},
    "radial": {"param": "z", "values": [1.5, 0.8, 0.4, 0.1, 0.05, 0.01]},
    # CORRIGIDO: a validacao mostrou que quem leva essa base perto da
    # interpolacao exata (leakage=2.7e-6) e' o parametro "l", nao
    # "alpha" -- estava variando o eixo errado. Agora "l" e' o
    # hiperparametro exposto na matriz; "alpha" fica fixo.
    # ATENCAO: o valor de alpha abaixo (1.0) e' uma suposicao -- a
    # validacao nao registrou qual alpha foi usado quando achou o
    # melhor l. Ajuste se souber o valor exato usado no sweep.
    "rational_quadratic": {"param": "l", "values": [1.0, 0.5, 0.1, 0.05, 0.01, 0.001], "fixed": {"alpha": 1.0}},
    "rbf_gaussian": {"param": "epsilon", "values": [0.4, 1.0, 2.5, 5, 6, 6.3, 7]},
    "shepard": {"param": "z", "values": [1, 1.782, 3, 5, 8]},
    # validacao achou o melhor leakage (0.3333) em alpha=38.59 -- fora
    # do range original [1.0, 3.0, 8.0]. Adicionado.
    "sigmoidal": {"param": "alpha", "values": [1.0, 3.0, 8.0, 38.59], "fixed": {"c": 0.0}},
    # validacao achou leakage=0 (EXATA) em beta=38.59 -- fora do range
    # original [0.5, 1.5, 4.0]. Adicionado (achado mais importante:
    # essa base consegue ser exata, e o range antigo nao mostrava isso).
    "softmax_radial": {"param": "beta", "values": [0.5, 1.5, 4.0, 38.59]},
    # validacao achou leakage=0.00183 em nu=1e-06 -- adicionado ao lado
    # do 0.0001 que ja' estava na lista.
    "student_t": {"param": "nu", "values": [30, 5, 1, 0.0001, 0.000001]},
    "wendland_c2": {"param": "h", "values": [3.0, 1.5, 0.7, 0.3, 0.1]},
}

DEFAULT_K_VALUES_1D = (1, 2, 3, 4)     # so' ha' 4 amostras no caso 1D
DEFAULT_K_VALUES_2D = (1, 2, 3, 4)
DEFAULT_N_PARAM_VALUES = 5


def param_range(basis_name: str, n_values: int = DEFAULT_N_PARAM_VALUES):
    """Gera `n_values` valores do hiperparametro da base, por
    espacamento GEOMETRICO entre o menor e o maior valor configurados
    em BASIS_CONFIG[basis_name]['values']. Preserva a direcao original
    (crescente/decrescente). Bases sem hiperparametro retornam [None].
    Unica funcao que decide "quais valores testar" -- muda aqui (ou no
    BASIS_CONFIG) e todas as imagens (curva e mapa) se atualizam."""
    config = BASIS_CONFIG[basis_name]
    if config is None:
        return [None]

    values = config["values"]
    if len(values) < 2:
        return list(values)

    lo, hi = min(values), max(values)
    if lo <= 0:
        result = np.linspace(min(values), max(values), n_values)
    else:
        result = np.geomspace(lo, hi, n_values)

    if values[0] > values[-1]:
        result = result[::-1]

    return [round(float(v), 6) for v in result]


def _basis_kwargs(basis_name: str, param_value, fixed: Optional[dict]):
    config = BASIS_CONFIG[basis_name]
    kwargs = dict(fixed or (config or {}).get("fixed") or {})
    if param_value is not None and config is not None:
        kwargs[config["param"]] = param_value
    return kwargs


def _normalize_rows(phi: np.ndarray) -> np.ndarray:
    """Partição da unidade (w_i = phi_i / sum(phi)), vetorizada por
    LINHA -- uma normalização por ponto de consulta. Fallback para
    pesos uniformes se a soma da linha for zero. Usada tanto pela
    curva 1D quanto pelo mapa 2D -- MESMA normalização nos dois casos."""
    totals = phi.sum(axis=1, keepdims=True)
    uniform = np.full_like(phi, 1.0 / phi.shape[1])
    zero_mask = np.isclose(totals, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = phi / totals
    return np.where(zero_mask, uniform, normalized)


def _format_param_label(param_name: str, value) -> str:
    if value is None:
        return "todos"
    if isinstance(value, (int, float)):
        return f"{param_name}={value:g}"
    return f"{param_name}={value}"


def get_basis_dir(basis_name: str, root_dir: str = "figs") -> Path:
    out_dir = Path(root_dir) / basis_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ==========================================================================
# 3. CASO 1D — interpolação com 4 amostras (estilo Fig. 2 do artigo),
#    agora generalizada para also variar k (quantos dos 4 vizinhos
#    entram na interpolação), igual ao caso 2D.
# ==========================================================================
X_SAMPLES = np.array([-2.0, -1.0, 0.0, 1.0])
Y_SAMPLES = np.array([0.75, 0.0, 1.0, 0.5])
X_MIN, X_MAX = -3.0, 3.0
N_POINTS_1D = 600


def interpolate_1d_grid(basis_name: str, k, param_value=None, fixed: Optional[dict] = None):
    """Interpolação 1D vetorizada, usando so' os k vizinhos mais
    proximos (dentre as 4 amostras) de cada ponto da grade -- k<=0 ou
    k>=4 usa todas as amostras (equivalente ao comportamento original
    de plot_basis_families.py, que sempre usava as 4)."""
    formula = BASIS_FORMULAS[basis_name]
    kwargs = _basis_kwargs(basis_name, param_value, fixed)

    n_samples = len(X_SAMPLES)
    k_eff = n_samples if (k is None or k <= 0 or k > n_samples) else k

    x_grid = np.linspace(X_MIN, X_MAX, N_POINTS_1D)
    d = np.abs(x_grid[:, None] - X_SAMPLES[None, :])  # (N, n_samples)

    idx = np.argpartition(d, k_eff - 1, axis=1)[:, :k_eff]
    dk = np.take_along_axis(d, idx, axis=1)

    phi = np.asarray(formula(dk, **kwargs), dtype=float)
    w = _normalize_rows(phi)

    y_vals = Y_SAMPLES[idx]
    y_grid = np.sum(w * y_vals, axis=1)
    return x_grid, y_grid


def plot_curve_grid(
    basis_name: str,
    k_values: Sequence[Optional[int]] = DEFAULT_K_VALUES_1D,
    param_values: Optional[Sequence] = None,
    fixed: Optional[dict] = None,
    save_dir: str = "figs",
    cell_size: float = 2.6,
    filename: str = "curve_grid.png",
) -> Path:
    """Matriz de curvas de interpolação 1D -- linhas = k, colunas =
    hiperparametro. Por padrão usa EXATAMENTE BASIS_CONFIG[basis]['values']
    (a curadoria manual, na ordem escrita) como colunas -- passe
    param_values=param_range(basis_name) explicitamente se quiser o
    leque geométrico automático em vez dos valores curados."""
    if basis_name not in BASIS_FORMULAS:
        raise ValueError(f"Base desconhecida: {basis_name}. Disponiveis: {sorted(BASIS_FORMULAS)}")

    config = BASIS_CONFIG[basis_name]
    if param_values is None:
        param_values = (config or {}).get("values", [None])
    param_name = (config or {}).get("param", "")

    n_rows, n_cols = len(k_values), len(param_values)
    fig_width = max(cell_size * n_cols, 8.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, cell_size * n_rows), squeeze=False)

    for i, k in enumerate(k_values):
        for j, param_value in enumerate(param_values):
            ax = axes[i][j]
            x_grid, y_grid = interpolate_1d_grid(basis_name, k, param_value, fixed)

            ax.plot(x_grid, y_grid, linestyle=":", color="blue", linewidth=1.2)
            ax.plot(X_SAMPLES, Y_SAMPLES, linestyle="none", marker="s", markersize=4,
                     markerfacecolor="blue", markeredgecolor="blue")
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(_format_param_label(param_name, param_value), fontsize=9)
            if j == 0:
                k_label = "todos" if (k is None or k <= 0) else str(k)
                ax.set_ylabel(f"k={k_label}", fontsize=9)

    col_desc = param_name if param_name else "sem hiperparametro"
    fig.suptitle(f"Interpolação 1D — base: {basis_name}  (linhas=k, colunas={col_desc})",
                 fontsize=11, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = get_basis_dir(basis_name, save_dir)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==========================================================================
# 4. CASO 2D — classificação (mapa de probabilidade/certeza).
# ==========================================================================
CLASS_COLORS = ["red", "green", "blue"]
CLASS_RGB = np.array([[1.0, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 0.0, 1.0]])


def make_training_set(n_per_class: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.07, 0.93, size=(n_per_class * 3, 2))
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


def fema_predict_grid(
    query_pts: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k,
    basis_name: str, param_value=None, fixed: Optional[dict] = None, n_classes: int = 3,
):
    """Classificacao FEMa vetorizada para MUITOS pontos de consulta de
    uma vez (uma grade inteira). Mesma formula, mesma normalizacao
    (_normalize_rows) que a curva 1D -- so' a dimensionalidade muda."""
    formula = BASIS_FORMULAS[basis_name]
    kwargs = _basis_kwargs(basis_name, param_value, fixed)

    n_train = len(X_train)
    k_eff = n_train if (k is None or k <= 0 or k > n_train) else k

    diffs = query_pts[:, None, :] - X_train[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]
    dk = np.take_along_axis(dists, idx, axis=1)

    phi = np.asarray(formula(dk, **kwargs), dtype=float)
    w = _normalize_rows(phi)

    onehot = np.zeros((n_train, n_classes))
    onehot[np.arange(n_train), y_train] = 1.0
    neighbor_onehot = onehot[idx]

    scores = np.einsum("qk,qkc->qc", w, neighbor_onehot)
    pred = np.argmax(scores, axis=1)
    return pred, scores


def plot_probability_grid(
    basis_name: str,
    k_values: Sequence[Optional[int]] = DEFAULT_K_VALUES_2D,
    param_values: Optional[Sequence] = None,
    fixed: Optional[dict] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    grid_res: int = 180,
    save_dir: str = "figs",
    cell_size: float = 2.5,
    filename: str = "probability_grid.png",
) -> Path:
    """Matriz de mapas de probabilidade 2D -- linhas = k, colunas =
    hiperparametro. Por padrão usa EXATAMENTE BASIS_CONFIG[basis]['values']
    (a curadoria manual) como colunas -- passe
    param_values=param_range(basis_name) explicitamente se quiser o
    leque geométrico automático em vez dos valores curados."""
    if basis_name not in BASIS_FORMULAS:
        raise ValueError(f"Base desconhecida: {basis_name}. Disponiveis: {sorted(BASIS_FORMULAS)}")

    config = BASIS_CONFIG[basis_name]
    if param_values is None:
        param_values = (config or {}).get("values", [None])
    param_name = (config or {}).get("param", "")

    if X_train is None or y_train is None:
        X_train, y_train = make_training_set()

    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    n_rows, n_cols = len(k_values), len(param_values)
    fig_width = max(cell_size * n_cols, 8.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, cell_size * n_rows), squeeze=False)

    for i, k in enumerate(k_values):
        for j, param_value in enumerate(param_values):
            ax = axes[i][j]
            pred, scores = fema_predict_grid(
                grid_pts, X_train, y_train, k, basis_name, param_value=param_value, fixed=fixed,
            )
            certainty = scores.max(axis=1).reshape(grid_res, grid_res)
            pred_img = pred.reshape(grid_res, grid_res)
            rgb = np.clip(CLASS_RGB[pred_img] * certainty[..., None], 0.0, 1.0)

            ax.imshow(rgb, extent=(0, 1, 0, 1), origin="lower")
            ax.scatter(X_train[:, 0], X_train[:, 1], c=[CLASS_COLORS[c] for c in y_train],
                       edgecolor="white", linewidth=0.4, s=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(_format_param_label(param_name, param_value), fontsize=9)
            if j == 0:
                k_label = "todos" if (k is None or k <= 0) else str(k)
                ax.set_ylabel(f"k={k_label}", fontsize=9)

    col_desc = param_name if param_name else "sem hiperparametro"
    fig.suptitle(f"Mapa de probabilidade — base: {basis_name}  (linhas=k, colunas={col_desc})",
                 fontsize=11, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = get_basis_dir(basis_name, save_dir)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==========================================================================
# 5. ENTRY POINT ÚNICO — roda as duas matrizes para TODAS as bases.
# ==========================================================================
def generate_all_figures(
    k_values_1d: Sequence[Optional[int]] = DEFAULT_K_VALUES_1D,
    k_values_2d: Sequence[Optional[int]] = DEFAULT_K_VALUES_2D,
    use_param_range: bool = False,
    n_param_values: int = DEFAULT_N_PARAM_VALUES,
    grid_res: int = 180,
    save_dir: str = "figs",
):
    """Gera curve_grid.png e probability_grid.png para TODAS as bases
    de BASIS_FORMULAS, sempre no MESMO conjunto de treino 2D (pra ficar
    comparavel entre bases).

    Por padrao (use_param_range=False), usa EXATAMENTE os valores
    curados manualmente em BASIS_CONFIG[basis]['values'] como colunas
    -- e' o modo recomendado quando voce ja' sabe quais valores testar
    (ex: valores achados por tests.linear_algebra.validation). Passe
    use_param_range=True para gerar um leque geometrico automatico via
    param_range() em vez dos valores curados (util para exploracao
    inicial, antes de saber quais valores valem a pena)."""
    X_train, y_train = make_training_set()

    summary = []
    for name in BASIS_FORMULAS:
        if use_param_range:
            param_values = param_range(name, n_param_values)
        else:
            param_values = (BASIS_CONFIG[name] or {}).get("values", [None])
        fixed = (BASIS_CONFIG[name] or {}).get("fixed")

        p_curve = plot_curve_grid(
            name, k_values=k_values_1d, param_values=param_values, fixed=fixed, save_dir=save_dir,
        )
        p_prob = plot_probability_grid(
            name, k_values=k_values_2d, param_values=param_values, fixed=fixed,
            X_train=X_train, y_train=y_train, grid_res=grid_res, save_dir=save_dir,
        )
        summary.append((name, p_curve, p_prob))

    return summary


if __name__ == "__main__":
    saved = generate_all_figures()
    print(f"{len(saved)} bases processadas (2 matrizes cada) em ./figs/<basis>/:")
    for name, p_curve, p_prob in saved:
        print(f" - {name}: {p_curve.name}, {p_prob.name}")