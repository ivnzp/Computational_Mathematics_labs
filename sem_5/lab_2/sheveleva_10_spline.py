import numpy as np

#x = np.array([0.87267, 1.22173, 1.57080, 1.91986, 2.26893, 2.61799, 2.96706])
#y = np.array([0.00123, 0.01343, 0.08411, 0.37341, 1.31146, 3.88447, 10.10742])

x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([5, 3, 11, 17, -15, -145, -457])

n = len(x)

dd = np.zeros((n, n))
dd[:, 0] = y

for j in range(1, n):
    for i in range(n - j):
        dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])

b = dd[0, :n].copy()

# Полином Ньютона
poly = np.poly1d([0.0])
for k in range(n):
    term = np.poly1d([1.0])
    for j in range(k):
        term *= np.poly1d([1.0, -x[j]])
    term *= b[k]
    poly += term

standard_coeffs = poly.coefficients

print("Коэффициенты стандартного полинома:")
for i, c in enumerate(standard_coeffs):
    power = len(standard_coeffs) - i - 1
    print(f"a_{power} = {c:.12g}")

# Производная полинома
poly_der = poly.deriv()
Pp = poly_der(x)  # значения производной в узлах

print("\nПроизводные в узлах:")
for i in range(n):
    print(f"P'({x[i]:.5f}) = {Pp[i]:.12g}")


spline_coeffs_cpp = []

for i in range(n - 1):
    xi, xi1 = x[i], x[i + 1]
    fi, fi1 = y[i], y[i + 1]
    mi, mi1 = Pp[i], Pp[i + 1]  # производные в узлах
    h = xi1 - xi
    h3 = h * h * h

    num_a3 = mi1 * h + mi * h - 2 * (fi1 - fi)
    a3 = num_a3 / h3

    num_a2 = -mi1 * h * (xi1 + 2 * xi) + 3 * (fi1 - fi) * (xi1 + xi) - mi * h * (xi + 2 * xi1)
    a2 = num_a2 / h3

    num_a1 = mi1 * xi * (2 * xi1 + xi) * h - 6 * (fi1 - fi) * xi * xi1 + mi * xi1 * (xi1 + 2 * xi) * h
    a1 = num_a1 / h3

    num_a0 = -mi1 * pow(xi, 2) * xi1 * h + fi1 * pow(xi, 2) * (3 * xi1 - xi) + fi * pow(xi1, 2) * (
                xi1 - 3 * xi) - mi * xi * pow(xi1, 2) * h
    a0 = num_a0 / h3

    spline_coeffs_cpp.append((a0, a1, a2, a3, xi))

print("\nКоэффициенты кубического сплайна:")
for i, (a0, a1, a2, a3, start) in enumerate(spline_coeffs_cpp):
    print(f"[{x[i]:.5f}, {x[i + 1]:.5f}]")
    print(f"  a0 = {a0:.12g}")
    print(f"  a1 = {a1:.12g}")
    print(f"  a2 = {a2:.12g}")
    print(f"  a3 = {a3:.12g}")


def spline_cpp(x0, spline_coeffs, nodes):
    for i in range(len(nodes) - 1):
        if nodes[i] <= x0 <= nodes[i + 1]:
            a0, a1, a2, a3, start = spline_coeffs[i]
            return a0 + a1 * x0 + a2 * x0 ** 2 + a3 * x0 ** 3
    return None


print("\nЗначения сплайна в точках")
test_points = [1.0, 1.65, 2.0, 2.5]
for x0 in test_points:
    result = spline_cpp(x0, spline_coeffs_cpp, x)
    print(f"S({x0}) = {result:.12g}")

print("\nЗначения интерполяционного многочлена в точках:")
test_points = [1.0, 1.65, 2.0, 2.5]
for x0 in test_points:
    result = poly(x0)
    print(f"P({x0}) = {result:.12g}")
