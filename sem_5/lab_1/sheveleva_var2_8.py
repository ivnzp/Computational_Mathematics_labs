import numpy as np

gamma0 = 5/3
rho0 = 1e-5
P0 = 3.848e3
U0 = 0

gamma3 = 7/5
C3 = 2.53248e4
U3 = 0
P3 = 3.04e9

X_val = P3 / P0
alpha0 = (gamma0 + 1) / (gamma0 - 1)
n_val = 2 * gamma3 / (gamma3 - 1)

mu = (U3 - U0) * np.sqrt((gamma0 - 1) * rho0 / (2 * P0))
rho3 = gamma3 * P3 / (C3**2)

v_val = (2 / (gamma3 - 1)) * np.sqrt(
    gamma3 * (gamma0 - 1) / 2 * (P3 / P0) * rho0 / rho3
)

print("Вычисленные параметры:")
print(f"alpha0 = {alpha0:.6f}")
print(f"n = {n_val:.6f}")
print(f"mu = {mu:.6e}")
print(f"v = {v_val:.6e}")
print(f"X = {X_val:.6e}")
print(f"rho3 = {rho3:.6e} г/см³")

coefficients = np.zeros(7)

coefficients[0] = X_val**2
coefficients[1] = -alpha0 * (v_val**2) * X_val
coefficients[2] = 2.0 * alpha0 * v_val * (v_val + mu) * X_val
coefficients[3] = -(2 + (v_val + mu)**2 * alpha0) * X_val
coefficients[4] = -(v_val**2)
coefficients[5] = 2.0 * v_val * (v_val + mu)
coefficients[6] = -(v_val + mu)**2 + 1.0

print("\nКоэффициенты:")
for i in range(7):
    print(f"c[{i}] = {coefficients[i]:.6e}")

A = coefficients[:-1][np.argmax(np.abs(coefficients[:-1]))]
B = coefficients[1:][np.argmax(np.abs(coefficients[1:]))]

print(f"\nA = {A:.6e}, B = {B:.6e}")

lowest_border = abs(coefficients[6]) / (abs(coefficients[6]) + abs(B))
highest_border = 1.0 + abs(A) / abs(coefficients[0])

print(f"Локализация: {lowest_border:.6e} <= |Z| <= {highest_border:.6e}")

def F(Z):
    return (
        X_val**2 * Z**(2 * n_val)
        - alpha0 * v_val**2 * X_val * Z**(n_val + 2)
        + 2 * alpha0 * v_val * (mu + v_val) * X_val * Z**(n_val + 1)
        - (2 + (mu + v_val)**2) * alpha0 * X_val * Z**n_val
        - v_val**2 * Z**2
        + 2 * v_val * (mu + v_val) * Z
        - (mu + v_val)**2 + 1
    )

def dF(Z):
    h = 1e-8
    return (F(Z + h) - F(Z - h)) / (2 * h)

def newton(x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx, dfx = F(x), dF(x)
        if dfx == 0:
            return None
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None

Z_solution = None
grid = np.linspace(1e-6, 10, 2000)
for a, b in zip(grid[:-1], grid[1:]):
    if F(a) * F(b) < 0:
        Z_solution = newton((a + b) / 2)
        break

if Z_solution is not None:
    print(f"\nНайденный корень: Z = {Z_solution:.6e}, F(Z) = {F(Z_solution):.2e}")

    P1 = P3 * Z_solution**n_val
    rho1 = rho3 * (P1 / P3)**(1 / gamma3)
    U1 = U3 + 2 * C3 / (gamma3 - 1) * (1 - (P1 / P3)**((gamma3 - 1) / (2 * gamma3)))
    C1 = np.sqrt(gamma3 * P1 / rho1)

    print("\nОбратный расчет:")
    print(f"P1 = {P1:.6e}")
    print(f"rho1 = {rho1:.6e}")
    print(f"U1 = {U1:.6e}")
    print(f"C1 = {C1:.6e}")
else:
    print("\nКорень не найден")




