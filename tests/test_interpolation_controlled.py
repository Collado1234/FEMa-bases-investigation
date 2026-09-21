import numpy as np


# ============================================================
# 1. Função de base
# ============================================================

def cauchy(r):
    return 1.0 / (1.0 + r**2)


# ============================================================
# 2. Pontos de treinamento e valores conhecidos
# ============================================================

X = np.array([0.0, 1.0, 2.0])

y = np.array([10.0, 20.0, 30.0])


# ============================================================
# 3. Construção da matriz Z
#
# Z[i, j] = phi_i(x_j)
# ============================================================

n = len(X)

Z = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance = abs(X[j] - X[i])
        Z[i, j] = cauchy(distance)


print("=" * 60)
print("TESTE CONTROLADO DE INTERPOLAÇÃO - CAUCHY")
print("=" * 60)

print("\nPontos X:")
print(X)

print("\nValores y:")
print(y)

print("\nMatriz Z:")
print(Z)


# ============================================================
# 4. Verificar se a base original já é interpoladora
# ============================================================

is_identity = np.allclose(Z, np.eye(n))

print("\nA base original já é interpoladora?")
print(is_identity)


# ============================================================
# 5. Construir Z^-1
# ============================================================

Z_inv = np.linalg.inv(Z)

print("\nZ^-1:")
print(Z_inv)


# ============================================================
# 6. Construir a base interpoladora
#
# phi_hat(x) = Z^-1 @ phi(x)
# ============================================================

Z_interpolated = Z_inv @ Z

print("\nZ^-1 @ Z:")
print(Z_interpolated)

print("\nA base transformada é cardinal?")
print(np.allclose(Z_interpolated, np.eye(n)))


# ============================================================
# 7. Função para avaliar as bases originais em um ponto x
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
# 8. Função para avaliar as bases interpoladoras em x
#
# phi_hat(x) = Z^-1 @ phi(x)
# ============================================================

def evaluate_interpolating_basis(x):
    phi = evaluate_original_basis(x)

    phi_hat = Z_inv @ phi

    return phi_hat


# ============================================================
# 9. Interpolador FEMa
#
# f_hat(x) = sum_i phi_hat_i(x) * y_i
# ============================================================

def interpolate(x):
    phi_hat = evaluate_interpolating_basis(x)

    return np.dot(phi_hat, y)


# ============================================================
# 10. Testar os próprios pontos de treinamento
# ============================================================

print("\n" + "=" * 60)
print("TESTE NOS PONTOS DE TREINAMENTO")
print("=" * 60)

for xi, yi in zip(X, y):
    prediction = interpolate(xi)

    print(
        f"x = {xi:.1f} | "
        f"y = {yi:.6f} | "
        f"f_hat(x) = {prediction:.6f} | "
        f"erro = {abs(prediction - yi):.2e}"
    )


# ============================================================
# 11. Testar pontos intermediários
# ============================================================

print("\n" + "=" * 60)
print("TESTE EM PONTOS INTERMEDIÁRIOS")
print("=" * 60)

test_points = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
]

for x in test_points:
    phi = evaluate_original_basis(x)
    phi_hat = evaluate_interpolating_basis(x)
    prediction = interpolate(x)

    print(
        f"x = {x:.2f} | "
        f"phi_hat = {phi_hat} | "
        f"f_hat(x) = {prediction:.6f}"
    )


# ============================================================
# 12. Verificação final da propriedade cardinal
# ============================================================

print("\n" + "=" * 60)
print("VERIFICAÇÃO DA PROPRIEDADE CARDINAL")
print("=" * 60)

cardinal_ok = True

for j, xj in enumerate(X):

    phi_hat = evaluate_interpolating_basis(xj)

    expected = np.zeros(n)
    expected[j] = 1.0

    ok = np.allclose(phi_hat, expected)

    print(
        f"x = {xj:.1f} | "
        f"phi_hat = {phi_hat} | "
        f"esperado = {expected} | "
        f"OK = {ok}"
    )

    if not ok:
        cardinal_ok = False


print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

if cardinal_ok:
    print("PASSOU: a base de Cauchy foi transformada em uma base cardinal.")
else:
    print("FALHOU: a base transformada não é cardinal.")