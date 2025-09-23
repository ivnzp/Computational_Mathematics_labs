import numpy as np

gamma0 = 5 / 3
rho0 = 1e-5
P0 = 3.848e3
U0 = 0

gamma3 = 7 / 5
C3 = 2.53248e4
U3 = 0
P3 = 3.04e9

X_val = P3 / P0
alpha0 = (gamma0 + 1) / (gamma0 - 1)
n_val = 2 * gamma3 / (gamma3 - 1)

mu = (U3 - U0) * np.sqrt((gamma0 - 1) * rho0 / (2 * P0))
rho3 = gamma3 * P3 / (C3 ** 2)

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
coefficients[0] = X_val ** 2
coefficients[1] = - alpha0 * v_val ** 2 * X_val
coefficients[2] = 2.0 * alpha0 * v_val * (mu + v_val) * X_val
coefficients[3] = - (2 + (mu + v_val) ** 2 * alpha0) * X_val
coefficients[4] = - v_val ** 2
coefficients[5] = 2.0 * v_val * (mu + v_val)
coefficients[6] = - (v_val + mu) ** 2 + 1.0

print("\nКоэффициенты:")
for i in range(7):
    print(f"c[{i}] = {coefficients[i]:.6e}")


A = np.max(np.abs(coefficients[0:6]))
B = np.max(np.abs(coefficients[1:7]))
lowest_border = abs(coefficients[6]) / (abs(coefficients[6]) + B)
highest_border = 1.0 + A / abs(coefficients[0])

print(f"\nЛокализация: {lowest_border:.6e} <= |Z| <= {highest_border:.6e}")

step = 0.01
z1 = 0.0
z2 = 0.0
found = False
z_curr = lowest_border + step

while z_curr <= highest_border:
    f_curr = (coefficients[0]*z_curr**(2*n_val) +
              coefficients[1]*z_curr**(n_val+2) +
              coefficients[2]*z_curr**(n_val+1) +
              coefficients[3]*z_curr**n_val +
              coefficients[4]*z_curr**2 +
              coefficients[5]*z_curr +
              coefficients[6])

    f_prev = (coefficients[0]*(z_curr-step)**(2*n_val) +
              coefficients[1]*(z_curr-step)**(n_val+2) +
              coefficients[2]*(z_curr-step)**(n_val+1) +
              coefficients[3]*(z_curr-step)**n_val +
              coefficients[4]*(z_curr-step)**2 +
              coefficients[5]*(z_curr-step) +
              coefficients[6])

    if f_curr * f_prev < 0:
        z1 = z_curr - step
        z2 = z_curr
        found = True
        print(f"Интервал со сменой знака: [{z1:.6e}, {z2:.6e}]")
        break
    z_curr += step

if not found:
    print("Корень не найден на заданном интервале.")
else:
    z = (z1 + z2) / 2.0
    z_prev = 0.0
    tol = 1e-12
    max_iter = 100

    for iter_count in range(max_iter):
        F_val = (coefficients[0]*z**(2*n_val) +
                 coefficients[1]*z**(n_val+2) +
                 coefficients[2]*z**(n_val+1) +
                 coefficients[3]*z**n_val +
                 coefficients[4]*z**2 +
                 coefficients[5]*z +
                 coefficients[6])

        dF_val = (coefficients[0]*2*n_val*z**(2*n_val-1) +
                  coefficients[1]*(n_val+2)*z**(n_val+1) +
                  coefficients[2]*(n_val+1)*z**n_val +
                  coefficients[3]*n_val*z**(n_val-1) +
                  coefficients[4]*2*z +
                  coefficients[5])

        z_new = z - F_val / dF_val

        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new

    print(f"\nНайденный корень: Z = {z:.12e}")

    P1 = P3 * z ** n_val
    P2 = P1
    C2 = (P2 / P3) ** ((gamma3 - 1) / (2 * gamma3))
    U2 = U3 + 2 * (C3 - C2) / (gamma3 - 1)
    U1 = U2
    rho1 = rho0 * ((gamma0 - 1) + (gamma0 + 1) * P1 / P0) / ((gamma0 + 1) + (gamma0 - 1) * P1 / P0)
    D = (rho1 * U1 - rho0 * U0) / (rho1 - rho0)

    print("\nОбратный расчет:")
    print(f"P1 = P2 = {P1:.6e}")
    print(f"C2 = {C2:.6e}")
    print(f"U1 = U2 = {U1:.6e}")
    print(f"rho1 = {rho1:.6e}")
    print(f"D = {D:.6e}")
