import numpy as np


# ============================================================
# 1. Função de base
# ============================================================

def cauchy(r):
    return 1.0 / (1.0 + r**2)


# ============================================================
# 2. Dados controlados
# ============================================================

X = np.array([0.0, 1.0, 2.0])

y = np.array([10.0, 20.0, 30.0])


# ============================================================
# 3. Avaliação das funções de base originais
# ============================================================

def evaluate_original_basis(x):
    """
    Retorna:

        [phi_1(x), phi_2(x), ..., phi_n(x)]
    """
    return np.array([
        cauchy(abs(x - xi))
        for xi in X
    ])


# ============================================================
# 4. Construção da matriz Z
#
# Z[i, j] = phi_i(x_j)
# ============================================================

n = len(X)

Z = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        Z[i, j] = cauchy(abs(X[j] - X[i]))


# ============================================================
# 5. Construção da base interpoladora
#
# phi_hat(x) = Z^-1 @ phi(x)
#
# Usamos solve em vez de inv para a construção.
# ============================================================

Z_inv = np.linalg.inv(Z)


def evaluate_interpolating_basis(x):
    """
    Avalia a base cardinal/interpoladora.

    phi_hat(x) = Z^-1 @ phi(x)
    """
    phi = evaluate_original_basis(x)

    return Z_inv @ phi


# ============================================================
# 6. MÉTODO ATUAL DO FEMa
#
# w_i(x) = phi_i(x) / sum(phi_j(x))
# ============================================================

def evaluate_current_weights(x):
    phi = evaluate_original_basis(x)

    total = phi.sum()

    if np.isclose(total, 0.0):
        raise ZeroDivisionError(
            "A soma das funções de base é zero."
        )

    return phi / total


def predict_current(x):
    weights = evaluate_current_weights(x)

    return np.dot(weights, y)


# ============================================================
# 7. MÉTODO INTERPOLADOR
#
# f_hat(x) = sum(phi_hat_i(x) * y_i)
# ============================================================

def predict_interpolating(x):
    phi_hat = evaluate_interpolating_basis(x)

    return np.dot(phi_hat, y)


# ============================================================
# 8. Informações iniciais
# ============================================================

print("=" * 80)
print("COMPARAÇÃO: MÉTODO ATUAL × MÉTODO INTERPOLADOR")
print("=" * 80)

print("\nPontos de treinamento:")
print(X)

print("\nValores:")
print(y)

print("\nMatriz Z:")
print(Z)

print("\nZ é identidade?")
print(np.allclose(Z, np.eye(n)))


# ============================================================
# 9. Comparação das bases em pontos de treinamento
# ============================================================

print("\n" + "=" * 80)
print("BASES NOS PONTOS DE TREINAMENTO")
print("=" * 80)

for x in X:

    phi = evaluate_original_basis(x)
    weights = evaluate_current_weights(x)
    phi_hat = evaluate_interpolating_basis(x)

    print(f"\nx = {x:.1f}")

    print("  phi original:")
    print(f"    {phi}")

    print("  pesos atuais (phi / sum(phi)):")
    print(f"    {weights}")

    print("  base interpoladora (Z^-1 @ phi):")
    print(f"    {phi_hat}")


# ============================================================
# 10. Comparação das predições
# ============================================================

print("\n" + "=" * 80)
print("COMPARAÇÃO DAS PREDIÇÕES")
print("=" * 80)

print(
    f"\n{'x':>6} | "
    f"{'Atual':>14} | "
    f"{'Interpolador':>14} | "
    f"{'Diferença':>14}"
)

print("-" * 80)

test_points = np.array([
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
])

for x in test_points:

    current = predict_current(x)
    interpolating = predict_interpolating(x)

    difference = interpolating - current

    print(
        f"{x:6.2f} | "
        f"{current:14.6f} | "
        f"{interpolating:14.6f} | "
        f"{difference:14.6f}"
    )


# ============================================================
# 11. Verificação da interpolação nos pontos de treinamento
# ============================================================

print("\n" + "=" * 80)
print("VERIFICAÇÃO NOS PONTOS DE TREINAMENTO")
print("=" * 80)

for x, expected in zip(X, y):

    current = predict_current(x)
    interpolating = predict_interpolating(x)

    current_error = abs(current - expected)
    interpolating_error = abs(interpolating - expected)

    print(
        f"x = {x:.1f} | "
        f"y = {expected:.6f} | "
        f"atual = {current:.6f} "
        f"(erro = {current_error:.2e}) | "
        f"interpolador = {interpolating:.6f} "
        f"(erro = {interpolating_error:.2e})"
    )


# ============================================================
# 12. Verificação da propriedade de partição da unidade
# ============================================================

print("\n" + "=" * 80)
print("SOMA DAS FUNÇÕES / PESOS")
print("=" * 80)

for x in test_points:

    current_weights = evaluate_current_weights(x)
    phi_hat = evaluate_interpolating_basis(x)

    print(
        f"x = {x:.2f} | "
        f"sum(atual) = {current_weights.sum():.12f} | "
        f"sum(interpolador) = {phi_hat.sum():.12f}"
    )


# ============================================================
# 13. Resultado final
# ============================================================

print("\n" + "=" * 80)
print("RESULTADO")
print("=" * 80)

training_interpolating = all(
    np.isclose(
        predict_interpolating(x),
        expected
    )
    for x, expected in zip(X, y)
)

training_current = all(
    np.isclose(
        predict_current(x),
        expected
    )
    for x, expected in zip(X, y)
)

print(
    "\nMétodo atual interpola os pontos de treinamento?"
    f" {training_current}"
)

print(
    "Método com Z^-1 interpola os pontos de treinamento?"
    f" {training_interpolating}"
)

if training_interpolating:
    print(
        "\nPASSOU: a construção com Z^-1 "
        "produz interpolação nos pontos de treinamento."
    )
else:
    print(
        "\nFALHOU: a construção com Z^-1 "
        "não produziu interpolação."
    )