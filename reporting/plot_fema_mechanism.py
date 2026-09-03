"""
plot_fema_mechanism.py
=======================

Reproduz as Fig. 4 e 5 do artigo do professor:

  - Fig. 4: "FEMa working mechanism" — um conjunto de treino 2D com 3
    classes, e as regiões de decisão obtidas pelo FEMa usando k = 1, 3
    e 5 vizinhos.
  - Fig. 5: mapa de probabilidade (grau de certeza) calculado pelo
    FEMa, em cinza (a) e colorido (b), com as amostras marcadas.

Igual ao script anterior (plot_basis_families.py), mas agora em 2D e
para classificação em vez de interpolação de uma curva. Reaproveita as
MESMAS fórmulas phi(d) e a MESMA normalização (compute_weights) de lá
— só muda o que é interpolado: em vez de um valor y escalar por
amostra, aqui é um vetor one-hot da classe da amostra.

Dependência: este arquivo importa `plot_basis_families.py` (deixe os
dois na mesma pasta).

Como usar
---------
$ python plot_fema_mechanism.py
    -> gera fig4_mecanismo_<base>.png e fig5_mapa_probabilidade_<base>.png
       em ./figs_fema/, usando a base "shepard" (você pode trocar no
       final do arquivo, na seção __main__, para qualquer chave de
       BASIS_FORMULAS).

Num notebook:
    from plot_fema_mechanism import plot_fema_mechanism, plot_probability_map
    plot_fema_mechanism("rbf_gaussian", param_value=1.0)
    plot_probability_map("shepard", param_value=2, k=5)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from plot_basis_families import BASIS_CONFIG, BASIS_FORMULAS

# ------------------------------------------------------------------
# 1. Conjunto de treino sintético (3 classes, espalhadas no [0,1]^2)
#    Seed fixa para o resultado ser reprodutível entre a Fig.4 e a
#    Fig.5 (o artigo usa o mesmo conjunto nas duas figuras).
# ------------------------------------------------------------------
CLASS_COLORS = ["red", "green", "blue"]
CLASS_RGB = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.75, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
DECISION_CMAP = ListedColormap(CLASS_COLORS)


def make_training_set(n_per_class: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.07, 0.93, size=(n_per_class * 3, 2))
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


# ------------------------------------------------------------------
# 2. Classificador FEMa genérico: para cada ponto de consulta, pega os
#    k vizinhos mais próximos do conjunto de treino, pesa via phi(d)
#    normalizado (igual a BaseBasis.compute_weights) e agrega o
#    one-hot da classe -> "escore" por classe (funciona como uma
#    probabilidade, já que os pesos somam 1). k=0 usa todos os pontos.
# ------------------------------------------------------------------
def _basis_kwargs(basis_name: str, param_value, fixed: dict | None):
    config = BASIS_CONFIG[basis_name]
    kwargs = dict(fixed or {})
    if param_value is not None and config is not None:
        param_key = config["param"].split(" ")[0]  # "z (r0)" -> "z"
        kwargs[param_key] = param_value
    return kwargs


def fema_predict(
    query_pts: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    basis_name: str,
    param_value=None,
    fixed: dict | None = None,
    n_classes: int = 3,
):
    formula = BASIS_FORMULAS[basis_name]
    kwargs = _basis_kwargs(basis_name, param_value, fixed)

    n_train = len(X_train)
    k_eff = n_train if (k is None or k <= 0 or k > n_train) else k

    onehot = np.zeros((n_train, n_classes))
    onehot[np.arange(n_train), y_train] = 1.0

    scores = np.empty((query_pts.shape[0], n_classes))
    for i, q in enumerate(query_pts):
        d = np.linalg.norm(X_train - q, axis=1)
        idx = np.argsort(d)[:k_eff]
        dk = d[idx]
        phi = np.asarray(formula(dk, **kwargs), dtype=float)
        total = phi.sum()
        w = np.full_like(phi, 1.0 / phi.size) if total == 0 else phi / total
        scores[i] = w @ onehot[idx]

    pred = np.argmax(scores, axis=1)
    return pred, scores


# ------------------------------------------------------------------
# 3. Fig. 4 — mecanismo de classificação variando k
# ------------------------------------------------------------------
def plot_fema_mechanism(
    basis_name: str,
    param_value=None,
    fixed: dict | None = None,
    k_values=(1, 3, 5),
    X_train=None,
    y_train=None,
    grid_res: int = 250,
    save_dir: str = "figs_fema",
):
    if X_train is None or y_train is None:
        X_train, y_train = make_training_set()

    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    labels = ["(b)", "(c)", "(d)", "(e)", "(f)"]
    n_panels = 1 + len(k_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.8))

    # (a) só o conjunto de treino, sem classificação
    ax0 = axes[0]
    for c in range(3):
        pts = X_train[y_train == c]
        ax0.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[c], s=22)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel("(a)")

    for ax, k, lab in zip(axes[1:], k_values, labels):
        pred, _ = fema_predict(grid_pts, X_train, y_train, k, basis_name, param_value, fixed)
        img = pred.reshape(grid_res, grid_res)
        ax.imshow(img, extent=(0, 1, 0, 1), origin="lower", cmap=DECISION_CMAP, vmin=0, vmax=2)
        ax.scatter(X_train[:, 0], X_train[:, 1], color="black", s=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{lab} k = {k}")

    param_txt = "" if param_value is None else f", {BASIS_CONFIG[basis_name]['param']} = {param_value}"
    fig.suptitle(f"Mecanismo FEMa — base: {basis_name}{param_txt}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig4_mecanismo_{basis_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------
# 4. Fig. 5 — mapa de probabilidade (grau de certeza)
# ------------------------------------------------------------------
def plot_probability_map(
    basis_name: str,
    param_value=None,
    fixed: dict | None = None,
    k: int = 0,
    X_train=None,
    y_train=None,
    grid_res: int = 250,
    save_dir: str = "figs_fema",
):
    if X_train is None or y_train is None:
        X_train, y_train = make_training_set()

    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    pred, scores = fema_predict(grid_pts, X_train, y_train, k, basis_name, param_value, fixed)
    certainty = scores.max(axis=1).reshape(grid_res, grid_res)
    pred_img = pred.reshape(grid_res, grid_res)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))

    # (a) mapa em escala de cinza (grau de certeza puro)
    axes[0].imshow(certainty, extent=(0, 1, 0, 1), origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=[CLASS_COLORS[c] for c in y_train], s=16)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("(a)")

    # (b) mapa colorido pela classe vencedora, com brilho = certeza
    rgb = CLASS_RGB[pred_img] * certainty[..., None]
    axes[1].imshow(rgb, extent=(0, 1, 0, 1), origin="lower")
    axes[1].scatter(
        X_train[:, 0], X_train[:, 1],
        c=[CLASS_COLORS[c] for c in y_train],
        edgecolor="white", linewidth=0.6, s=16,
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("(b)")

    k_txt = "todos" if (k is None or k <= 0) else str(k)
    param_txt = "" if param_value is None else f", {BASIS_CONFIG[basis_name]['param']} = {param_value}"
    fig.suptitle(f"Mapa de probabilidade — base: {basis_name}{param_txt}, k = {k_txt}")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig5_mapa_probabilidade_{basis_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------
# 5. Roda as 21 bases de uma vez (Fig.4 + Fig.5 para cada uma),
#    sempre no MESMO conjunto de treino, pra ficar comparável.
#
#    Hiperparâmetro usado por base: como aqui quem varia é k (não faz
#    sentido também variar z/epsilon/etc. ao mesmo tempo), pego o
#    valor DO MEIO do sweep já definido em BASIS_CONFIG (o mesmo dict
#    usado no plot_basis_families.py) como um valor "representativo".
#    Ajuste PARAM_OVERRIDES abaixo se quiser um valor específico para
#    alguma base em particular.
# ------------------------------------------------------------------
PARAM_OVERRIDES: dict = {
    # "shepard": 2,  # exemplo: force z=2 em vez do valor do meio (3)
}


def _default_param_for(basis_name: str):
    if basis_name in PARAM_OVERRIDES:
        return PARAM_OVERRIDES[basis_name]
    config = BASIS_CONFIG[basis_name]
    if config is None:
        return None
    values = config["values"]
    return values[len(values) // 2]


def plot_all_fema(
    k_values=(1, 3, 5),
    prob_k: int = 0,
    save_dir: str = "figs_fema",
    X_train=None,
    y_train=None,
):
    """
    Gera fig4_mecanismo_<base>.png e fig5_mapa_probabilidade_<base>.png
    para TODAS as bases de BASIS_FORMULAS, no mesmo conjunto de treino.
    """
    if X_train is None or y_train is None:
        X_train, y_train = make_training_set()

    saved = []
    for name in BASIS_FORMULAS:
        config = BASIS_CONFIG[name]
        param_value = _default_param_for(name)
        fixed = config.get("fixed") if config is not None else None

        p4 = plot_fema_mechanism(
            name, param_value=param_value, fixed=fixed, k_values=k_values,
            X_train=X_train, y_train=y_train, save_dir=save_dir,
        )
        p5 = plot_probability_map(
            name, param_value=param_value, fixed=fixed, k=prob_k,
            X_train=X_train, y_train=y_train, save_dir=save_dir,
        )
        saved.append((name, p4, p5))

    return saved


if __name__ == "__main__":
    saved = plot_all_fema()
    print(f"{len(saved)} bases processadas (2 figuras cada) em ./figs_fema/:")
    for name, p4, p5 in saved:
        print(f" - {name}: {p4.name}, {p5.name}")