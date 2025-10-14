import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.log(1.01 + x)


def chebyshev_nodes(n):
    k = np.arange(0, n + 1)
    return np.cos((2 * k + 1) / (2 * (n + 1)) * np.pi)


def uniform_nodes(n):
    return np.linspace(-1, 1, n + 1)


def lagrange_interp(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    P = np.zeros_like(x_eval)
    for j in range(n):
        Lj = np.ones_like(x_eval)
        for m in range(n):
            if m != j:
                Lj *= (x_eval - x_nodes[m]) / (x_nodes[j] - x_nodes[m])
        P += y_nodes[j] * Lj
    return P


x_dense = np.linspace(-1, 1, 10000)
y_true = f(x_dense)

n_list = [i for i in range(2, 50)]
results = []

for n in n_list:
    xu = uniform_nodes(n)
    yu = f(xu)
    p_u = lagrange_interp(xu, yu, x_dense)
    err_u = np.max(np.abs(y_true - p_u))

    xc = chebyshev_nodes(n)
    yc = f(xc)
    p_c = lagrange_interp(xc, yc, x_dense)
    err_c = np.max(np.abs(y_true - p_c))

    results.append((n, err_u, err_c))

n_values = [item[0] for item in results]
uniform_errors = [item[1] for item in results]
chebyshev_errors = [item[2] for item in results]


max_err_uniform = max(uniform_errors)
max_err_chebyshev = max(chebyshev_errors)

print(f"Максимум погрешности (равномерные узлы): {max_err_uniform:.6e}")
print(f"Максимум погрешности (Чебышевские узлы): {max_err_chebyshev:.6e}")


plt.figure(figsize=(9, 6))
plt.semilogy(n_values, uniform_errors, marker='o', label='равномерные узлы')
plt.semilogy(n_values, chebyshev_errors, marker='s', label='чебышевские узлы')
plt.title('Сравнение ошибок интерполяции')
plt.xlabel('степень n')
plt.ylabel('max |f - P_n|')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()



chosen_n = [5, 10, 15, 20]

plt.figure(figsize=(14, 10))
for i, n in enumerate(chosen_n, 1):
    plt.subplot(2, 2, i)

    xu = uniform_nodes(n)
    yu = f(xu)
    p_u = lagrange_interp(xu, yu, x_dense)

    xc = chebyshev_nodes(n)
    yc = f(xc)
    p_c = lagrange_interp(xc, yc, x_dense)

    plt.plot(x_dense, y_true, 'k-', lw=2, label="f(x)")
    plt.plot(x_dense, p_u, 'b--', lw=1.5, label="Интерп. (равномерные)")
    plt.plot(x_dense, p_c, 'g-.', lw=1.5, label="Интерп. (Чебышевские)")
    plt.scatter(xu, yu, color='blue', marker='o', s=30)
    plt.scatter(xc, yc, color='green', marker='s', s=30)

    plt.title(f"Интерполяция f(x) при n={n}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, ls="--", lw=0.5)
    if i == 1:
        plt.legend()

plt.tight_layout()
plt.show()

