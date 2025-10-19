import numpy as np

x = np.array([0.87267, 1.22173, 1.57080, 1.91986, 2.26893, 2.61799, 2.96706])
y = np.array([0.00123, 0.01343, 0.08411, 0.37341, 1.31146, 3.88447, 10.10742])

n = len(x)

dd = np.zeros((n, n))
dd[:, 0] = y

for j in range(1, n):
    for i in range(n - j):
        dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])

b = dd[0, :n].copy()

poly = np.poly1d([0.0])
for k in range(n):
    term = np.poly1d([1.0])
    for j in range(k):
        term *= np.poly1d([1.0, -x[j]])
    term *= b[k]
    poly += term

standard_coeffs = poly.coefficients

print("\nКоэффициенты стандартного полинома:")
for i, c in enumerate(standard_coeffs):
    power = len(standard_coeffs) - i - 1
    print(f"a_{power} = {c:.12g}")

x0 = float(input("\nВведите значение x0 для полинома: "))
y0 = poly(x0)
print(f"P({x0}) = {y0}")

poly_der = poly.deriv() # Производная интерполяционного полинома
Pp = poly_der(x)  # значения производной в узлах

spline_coeffs = []

for i in range(n - 1):
    xi, xi1 = x[i], x[i + 1]
    fi, fi1 = y[i], y[i + 1]
    dpi, dpi1 = Pp[i], Pp[i + 1]
    h = xi1 - xi
    denom = h ** 3

    a3 = (dpi1 * h - 2 * (fi1 - fi) + dpi * h) / denom
    a2 = (-dpi1 * h * (xi1 + 2 * xi) + 3 * (fi1 - fi) * (xi1 + xi) - dpi * h * (xi + 2 * xi1)) / denom
    a1 = (dpi1 * xi1 * (xi1 + 2 * xi) - 6 * (fi1 - fi) * xi1 + dpi * xi * (xi1 + 2 * xi)) / denom
    a0 = (-dpi1 * xi1 ** 2 * h + fi1 * xi1 ** 2 * (3 * xi - xi1) + fi * xi ** 2 * (xi - 3 * xi1)
          - dpi * xi ** 2 * h) / denom

    spline_coeffs.append((a3, a2, a1, a0))

print("\nКоэффициенты кубического сплайна по интервалам:")
for i, (a3, a2, a1, a0) in enumerate(spline_coeffs):
    print(f"[{x[i]:.5f}, {x[i+1]:.5f}]  →  a3={a3:.6f}, a2={a2:.6f}, a1={a1:.6f}, a0={a0:.6f}")


def spline(x0):
    for i in range(n - 1):
        if x[i] <= x0 <= x[i + 1]:
            a3, a2, a1, a0 = spline_coeffs[i]
            dx = x0 - x[i]
            return a3 * dx**3 + a2 * dx**2 + a1 * dx + a0
    return None


print("\nТаблица приближённых значений функции по кубическому сплайну:")
X_values = np.linspace(x[0], x[-1], 10)  # 10 точек внутри интервала интерполяции
for X in X_values:
    print(f"S({X:.5f}) = {spline(X):.8f}")