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

n = len(X)


# ============================================================
# 3. Avaliar a base original
# ============================================================

def evaluate_original_basis(x):
    return np.array([
        cauchy(abs(x - xi))
        for xi in X
    ])


# ============================================================
# 4. Construir Z
#
# Z[i, j] = phi_i(x_j)
# ============================================================

Z = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        Z[i, j] = cauchy(abs(X[j] - X[i]))


# ============================================================
# 5. Construir a base interpoladora
#
# phi_hat(x) = Z^-1 @ phi(x)
# ============================================================

Z_inv = np.linalg.inv(Z)


def evaluate_interpolating_basis(x):
    phi = evaluate_original_basis(x)

    return Z_inv @ phi


# ============================================================
# 6. Três/quatro versões das funções
# ============================================================

def original_basis(x):
    """
    Base original:
        phi
    """
    return evaluate_original_basis(x)


def current_normalized_basis(x):
    """
    Método atual:
        phi / sum(phi)
    """
    phi = evaluate_original_basis(x)

    return phi / phi.sum()


def interpolating_basis(x):
    """
    Base interpoladora:
        Z^-1 @ phi
    """
    return evaluate_interpolating_basis(x)


def interpolating_normalized_basis(x):
    """
    Base interpoladora + normalização:

        (Z^-1 @ phi)
        ----------------
        sum(Z^-1 @ phi)
    """
    phi_hat = evaluate_interpolating_basis(x)

    return phi_hat / phi_hat.sum()


# ============================================================
# 7. Predição
# ============================================================

def predict(basis_function, x):
    basis = basis_function(x)

    return np.dot(basis, y)


# ============================================================
# 8. Cabeçalho
# ============================================================

print("=" * 100)
print("TESTE: INTERPOLAÇÃO × PARTIÇÃO DA UNIDADE × NORMALIZAÇÃO")
print("=" * 100)

print("\nPontos:")
print(X)

print("\nValores:")
print(y)

print("\nMatriz Z:")
print(Z)

print("\nZ é identidade?")
print(np.allclose(Z, np.eye(n)))


# ============================================================
# 9. Comparação das quatro versões
# ============================================================

print("\n" + "=" * 100)
print("COMPARAÇÃO DAS BASES")
print("=" * 100)

test_points = np.array([
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
])

for x in test_points:

    original = original_basis(x)
    current = current_normalized_basis(x)
    interpolating = interpolating_basis(x)
    interpolating_normalized = interpolating_normalized_basis(x)

    print(f"\nx = {x:.2f}")

    print(f"  Original:                   {original}")
    print(f"  Atual normalizada:          {current}")
    print(f"  Interpoladora Z^-1:         {interpolating}")
    print(f"  Interpoladora normalizada:  {interpolating_normalized}")


# ============================================================
# 10. Soma das bases
# ============================================================

print("\n" + "=" * 100)
print("PARTIÇÃO DA UNIDADE")
print("=" * 100)

print(
    f"\n{'x':>6} | "
    f"{'Original':>12} | "
    f"{'Atual':>12} | "
    f"{'Z^-1':>12} | "
    f"{'Z^-1 + norm.':>14}"
)

print("-" * 100)

for x in test_points:

    original_sum = original_basis(x).sum()
    current_sum = current_normalized_basis(x).sum()
    interpolating_sum = interpolating_basis(x).sum()
    interpolating_normalized_sum = interpolating_normalized_basis(x).sum()

    print(
        f"{x:6.2f} | "
        f"{original_sum:12.6f} | "
        f"{current_sum:12.6f} | "
        f"{interpolating_sum:12.6f} | "
        f"{interpolating_normalized_sum:14.6f}"
    )


# ============================================================
# 11. Predições
# ============================================================

print("\n" + "=" * 100)
print("PREDIÇÕES")
print("=" * 100)

print(
    f"\n{'x':>6} | "
    f"{'Original':>12} | "
    f"{'Atual':>12} | "
    f"{'Z^-1':>12} | "
    f"{'Z^-1 + norm.':>14}"
)

print("-" * 100)

for x in test_points:

    original_prediction = predict(original_basis, x)
    current_prediction = predict(current_normalized_basis, x)
    interpolating_prediction = predict(interpolating_basis, x)
    interpolating_normalized_prediction = predict(
        interpolating_normalized_basis,
        x
    )

    print(
        f"{x:6.2f} | "
        f"{original_prediction:12.6f} | "
        f"{current_prediction:12.6f} | "
        f"{interpolating_prediction:12.6f} | "
        f"{interpolating_normalized_prediction:14.6f}"
    )


# ============================================================
# 12. Teste nos pontos de treinamento
# ============================================================

print("\n" + "=" * 100)
print("INTERPOLAÇÃO NOS PONTOS DE TREINAMENTO")
print("=" * 100)

for x, expected in zip(X, y):

    original_prediction = predict(original_basis, x)
    current_prediction = predict(current_normalized_basis, x)
    interpolating_prediction = predict(interpolating_basis, x)
    interpolating_normalized_prediction = predict(
        interpolating_normalized_basis,
        x
    )

    print(f"\nx = {x:.1f} | esperado = {expected:.6f}")

    print(
        f"  Original:                  "
        f"{original_prediction:.6f} "
        f"(erro = {abs(original_prediction - expected):.2e})"
    )

    print(
        f"  Atual normalizada:         "
        f"{current_prediction:.6f} "
        f"(erro = {abs(current_prediction - expected):.2e})"
    )

    print(
        f"  Interpoladora Z^-1:        "
        f"{interpolating_prediction:.6f} "
        f"(erro = {abs(interpolating_prediction - expected):.2e})"
    )

    print(
        f"  Interpoladora normalizada: "
        f"{interpolating_normalized_prediction:.6f} "
        f"(erro = "
        f"{abs(interpolating_normalized_prediction - expected):.2e})"
    )


# ============================================================
# 13. Verificar cardinalidade
# ============================================================

print("\n" + "=" * 100)
print("CARDINALIDADE")
print("=" * 100)

for j, x in enumerate(X):

    expected = np.zeros(n)
    expected[j] = 1.0

    current = current_normalized_basis(x)
    interpolating = interpolating_basis(x)
    interpolating_normalized = interpolating_normalized_basis(x)

    print(f"\nx = {x:.1f}")
    print(f"  Esperado:                  {expected}")
    print(f"  Atual normalizada:         {current}")
    print(f"  Interpoladora Z^-1:        {interpolating}")
    print(f"  Interpoladora normalizada: {interpolating_normalized}")


# ============================================================
# 14. Testes automáticos
# ============================================================

print("\n" + "=" * 100)
print("TESTES AUTOMÁTICOS")
print("=" * 100)


# Partition of unity da versão atual
current_partition_ok = all(
    np.isclose(
        current_normalized_basis(x).sum(),
        1.0
    )
    for x in test_points
)


# Partition of unity após normalização
interpolating_normalized_partition_ok = all(
    np.isclose(
        interpolating_normalized_basis(x).sum(),
        1.0
    )
    for x in test_points
)


# Cardinalidade Z^-1
interpolating_cardinal_ok = all(
    np.allclose(
        interpolating_basis(x),
        np.eye(n)[j]
    )
    for j, x in enumerate(X)
)


# Cardinalidade Z^-1 + normalização
interpolating_normalized_cardinal_ok = all(
    np.allclose(
        interpolating_normalized_basis(x),
        np.eye(n)[j]
    )
    for j, x in enumerate(X)
)


print(
    f"\nAtual possui partition of unity? "
    f"{current_partition_ok}"
)

print(
    f"Z^-1 possui cardinalidade? "
    f"{interpolating_cardinal_ok}"
)

print(
    f"Z^-1 + normalização possui partition of unity? "
    f"{interpolating_normalized_partition_ok}"
)

print(
    f"Z^-1 + normalização mantém cardinalidade? "
    f"{interpolating_normalized_cardinal_ok}"
)