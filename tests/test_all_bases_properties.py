import numpy as np

def _safe(d: np.ndarray) -> np.ndarray:
    """Evita divisão por zero / log(0), igual ao DELTA do seu pacote."""
    return np.where(d == 0, DELTA, d)

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





# ============================================================
# Configuração
# ============================================================

X = np.array([0.0, 1.0, 2.0])

TOL = 1e-8


# Parâmetros de teste.
#
# Não estamos tentando encontrar os "melhores" parâmetros.
# Queremos apenas uma configuração válida para verificar
# as propriedades matemáticas.
#
BASIS = {
    "shepard":              (phi_shepard,              (2.0,)),
    "radial":               (phi_radial,               (1.0,)),
    "rbf_gaussian":         (phi_rbf_gaussian,         (1.0,)),
    "multiquadratic":       (phi_multiquadratic,       (1.0,)),
    "inverse_multiquadratic": (phi_inverse_multiquadratic, (1.0,)),
    "wendland_c2":          (phi_wendland_c2,          (1.0,)),
    "cubic_spline":         (phi_cubic_spline,         (1.0,)),
    "quartic_spline":       (phi_quartic_spline,       (1.0,)),
    "gen_exponential":      (phi_gen_exponential,      (1.0, 2.0)),
    "softmax_radial":       (phi_softmax_radial,       (1.0,)),
    "attention":            (phi_attention,            ()),
    "logarithmic":           (phi_logarithmic,          (1.0,)),
    "harmonic":              (phi_harmonic,             ()),
    "laplacian":             (phi_laplacian,            (1.0,)),
    "cauchy":                (phi_cauchy,               (1.0,)),
    "student_t":             (phi_student_t,            (2.0,)),
    "cosine":                (phi_cosine,               (1.0,)),
    "sigmoidal":             (phi_sigmoidal,            (1.0, 0.0)),
    "lorentzian":            (phi_lorentzian,           ()),
    "entropic":              (phi_entropic,             (1.0,)),
    "rational_quadratic":    (phi_rational_quadratic,   (1.0, 1.0)),
}


# ============================================================
# Helpers
# ============================================================

def evaluate_phi(func, params, distances):
    """
    Avalia a função e converte para float ndarray.
    """
    values = func(distances, *params)
    return np.asarray(values, dtype=float)


def build_Z(func, params, X):
    """
    Z_ij = phi(||x_i - x_j||)
    """
    n = len(X)

    Z = np.zeros((n, n))

    for i in range(n):
        distances = np.abs(X[i] - X)
        Z[i, :] = evaluate_phi(func, params, distances)

    return Z


def build_phi_vector(func, params, x, X):
    """
    phi_i(x) = phi(||x - x_i||)
    """
    distances = np.abs(x - X)
    return evaluate_phi(func, params, distances)


def is_identity(M, tol=TOL):
    return np.allclose(M, np.eye(M.shape[0]), atol=tol, rtol=tol)


def is_partition_of_unity(values, tol=TOL):
    return np.isclose(
        np.sum(values),
        1.0,
        atol=tol,
        rtol=tol,
    )


# ============================================================
# Teste principal
# ============================================================

def test_all_bases():

    print()
    print("=" * 100)
    print("TESTE DAS BASES FEMa")
    print("=" * 100)
    print()

    results = []

    for name, (func, params) in BASIS.items():

        print("-" * 100)
        print(f"BASE: {name}")
        print("-" * 100)

        # ----------------------------------------------------
        # 1. Construir Z
        # ----------------------------------------------------

        try:
            Z = build_Z(func, params, X)
        except Exception as e:
            print(f"ERRO ao construir Z: {e}")
            results.append((name, "ERROR", str(e)))
            continue

        print("Z =")
        print(np.round(Z, 6))

        # ----------------------------------------------------
        # 2. Raw basis é cardinal?
        # ----------------------------------------------------

        raw_cardinal = is_identity(Z)

        print(f"Raw cardinal?              {raw_cardinal}")

        # ----------------------------------------------------
        # 3. Z é invertível?
        # ----------------------------------------------------

        det_Z = np.linalg.det(Z)

        try:
            cond_Z = np.linalg.cond(Z)
        except Exception:
            cond_Z = np.inf

        invertible = np.isfinite(cond_Z) and cond_Z < 1e12

        print(f"det(Z):                    {det_Z:.6e}")
        print(f"cond(Z):                   {cond_Z:.6e}")
        print(f"Z invertível?              {invertible}")

        # ----------------------------------------------------
        # 4. Construir Z^-1 phi
        # ----------------------------------------------------

        interpolating_cardinal = False
        interpolation_matrix = None

        if invertible:

            try:
                Z_inv = np.linalg.inv(Z)

                # Em cada nó x_j:
                #
                # phi(x_j)
                #
                # gera a coluna/linha correspondente de Z.
                #
                # Testamos diretamente a matriz resultante.

                interpolation_matrix = Z_inv @ Z

                interpolating_cardinal = is_identity(
                    interpolation_matrix
                )

                print()
                print("Z^-1 @ Z =")
                print(np.round(interpolation_matrix, 6))

                print(
                    f"Z^-1 phi cardinal?       "
                    f"{interpolating_cardinal}"
                )

            except Exception as e:
                print(f"Erro ao construir Z^-1 phi: {e}")

        # ----------------------------------------------------
        # 5. Testar vários pontos
        # ----------------------------------------------------

        test_points = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        raw_pou = True
        interp_pou = True
        normalized_interp_pou = True

        max_raw_sum_error = 0.0
        max_interp_sum_error = 0.0

        for x in test_points:

            phi = build_phi_vector(
                func,
                params,
                x,
                X,
            )

            # ------------------------------------------------
            # Raw
            # ------------------------------------------------

            raw_sum = np.sum(phi)

            raw_error = abs(raw_sum - 1.0)

            max_raw_sum_error = max(
                max_raw_sum_error,
                raw_error,
            )

            if raw_error > TOL:
                raw_pou = False

            # ------------------------------------------------
            # Interpolante
            # ------------------------------------------------

            if invertible:

                interp_phi = Z_inv @ phi

                interp_sum = np.sum(interp_phi)

                interp_error = abs(interp_sum - 1.0)

                max_interp_sum_error = max(
                    max_interp_sum_error,
                    interp_error,
                )

                if interp_error > TOL:
                    interp_pou = False

                # ------------------------------------------------
                # Interpolante + normalização
                # ------------------------------------------------

                if abs(interp_sum) > TOL:

                    normalized = interp_phi / interp_sum

                    normalized_sum = np.sum(normalized)

                    if not np.isclose(
                        normalized_sum,
                        1.0,
                        atol=TOL,
                        rtol=TOL,
                    ):
                        normalized_interp_pou = False

        # ----------------------------------------------------
        # 6. Verificar cardinalidade após normalização
        # ----------------------------------------------------

        normalized_preserves_cardinality = False

        if invertible:

            normalized_preserves_cardinality = True

            for j, x in enumerate(X):

                phi = build_phi_vector(
                    func,
                    params,
                    x,
                    X,
                )

                interp_phi = Z_inv @ phi

                total = np.sum(interp_phi)

                if abs(total) <= TOL:
                    normalized_preserves_cardinality = False
                    break

                normalized = interp_phi / total

                expected = np.zeros(len(X))
                expected[j] = 1.0

                if not np.allclose(
                    normalized,
                    expected,
                    atol=TOL,
                    rtol=TOL,
                ):
                    normalized_preserves_cardinality = False
                    break

        # ----------------------------------------------------
        # 7. Resultado
        # ----------------------------------------------------

        print()
        print(f"Raw partition of unity?             {raw_pou}")
        print(f"Max raw POU error:                  {max_raw_sum_error:.6e}")

        if invertible:
            print(
                f"Interpolante partition of unity?   "
                f"{interp_pou}"
            )

            print(
                f"Max interpolante POU error:         "
                f"{max_interp_sum_error:.6e}"
            )

            print(
                f"Interpolante + norm POU?            "
                f"{normalized_interp_pou}"
            )

            print(
                f"Norm. preserva cardinalidade?       "
                f"{normalized_preserves_cardinality}"
            )

        results.append({
            "name": name,
            "raw_cardinal": raw_cardinal,
            "invertible": invertible,
            "interpolating_cardinal": interpolating_cardinal,
            "raw_pou": raw_pou,
            "interp_pou": interp_pou if invertible else None,
            "normalized_interp_pou": (
                normalized_interp_pou if invertible else None
            ),
            "normalized_preserves_cardinality": (
                normalized_preserves_cardinality
                if invertible
                else None
            ),
            "condition_number": cond_Z,
        })

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print()
    print("=" * 120)
    print("RESUMO")
    print("=" * 120)

    print(
        f"{'Base':<28}"
        f"{'Raw I':<8}"
        f"{'Inv':<8}"
        f"{'Z^-1 I':<10}"
        f"{'Raw POU':<10}"
        f"{'Interp POU':<12}"
        f"{'Norm POU':<10}"
        f"{'Norm Card':<10}"
    )

    print("-" * 120)

    for r in results:

        if isinstance(r, tuple):
            continue

        print(
            f"{r['name']:<28}"
            f"{str(r['raw_cardinal']):<8}"
            f"{str(r['invertible']):<8}"
            f"{str(r['interpolating_cardinal']):<10}"
            f"{str(r['raw_pou']):<10}"
            f"{str(r['interp_pou']):<12}"
            f"{str(r['normalized_interp_pou']):<10}"
            f"{str(r['normalized_preserves_cardinality']):<10}"
        )

    print()
    print("=" * 120)


if __name__ == "__main__":
    test_all_bases()