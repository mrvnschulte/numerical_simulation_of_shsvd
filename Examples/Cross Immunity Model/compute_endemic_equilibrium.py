from scipy.optimize import root
from invasion_reproduction_number import invasion_reproduction_number
import numpy as np
import matplotlib.pyplot as plt


sigma = 0.7
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01


def x_6(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return b / (beta_1 * i_1 + beta_2 * i_2 + b)


def x_1(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        (1 - sigma)
        * beta_1
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
    ) / (gamma_1 + beta_2 * i_2 + b)


def x_4(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        (1 - sigma)
        * beta_2
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
    ) / (gamma_2 + beta_1 * i_1 + b)


def x_7(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        gamma_1 * x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (beta_2 * i_2 + b)


def x_8(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        gamma_2 * x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (beta_1 * i_1 + b)


def x_3(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        beta_1
        * x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + beta_2
        * x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
    ) / (gamma_1 + gamma_2 + b)


def x_2(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        sigma
        * beta_1
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + beta_1
        * x_8(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + gamma_2 * x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (gamma_1 + b)


def x_5(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        sigma
        * beta_2
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
        + beta_2
        * x_7(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
        + gamma_1 * x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (gamma_2 + b)


def x_9(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    return (
        gamma_1 * x_2(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + gamma_2 * x_5(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / b


def equation(
    i,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    i_1, i_2 = i
    return [
        x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_2(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        - i_1,
        x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_5(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        - i_2,
    ]


sol = root(equation, [1, 1], args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b))
print(sol.x)
n = 101
i1_vals = np.linspace(0, 1, n)
i2_vals = np.linspace(0, 1, n)
I1, I2 = np.meshgrid(i1_vals, i2_vals, indexing="ij")

# Lösungen speichern

Sol1 = np.zeros((n, n))
Sol2 = np.zeros((n, n))
solutions = []

for j, i_1 in enumerate(i1_vals):
    for k, i_2 in enumerate(i2_vals):
        sol = root(
            equation,
            [i_1, i_2],
            args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        )
        Sol1[j, k] = np.round(sol.x[0], 5)
        Sol2[j, k] = np.round(sol.x[1], 5)
        if sol.success and np.linalg.norm(sol.fun) < 1e-3:
            solutions.append(tuple(np.round(sol.x, 5)))
            if sol.x[0] < -2:
                print(sol.fun)

uniq = sorted(set(solutions))
print("Found unique solutions:", uniq)


# # print(invasion_reproduction_number(beta_1 / gamma_1, beta_2 / gamma_2, sigma))
# # print(invasion_reproduction_number(beta_2 / gamma_2, beta_1 / gamma_1, sigma))
plt.figure(figsize=(8, 6))
cp = plt.contourf(I1, I2, Sol1, levels=50, cmap="viridis")
plt.colorbar(cp, label="i1 Lösung")
plt.xlabel("i_1")
plt.ylabel("i_2")
plt.title("Gefundene i1-Lösungen über das Grid (Sol1)")

plt.figure(figsize=(8, 6))
cp2 = plt.contourf(I1, I2, Sol2, levels=50, cmap="viridis")
plt.colorbar(cp2, label="i2 Lösung")
plt.xlabel("i_1")
plt.ylabel("i_2")
plt.title("Gefundene i2-Lösungen über das Grid (Sol2)")

# plt.show()

solutions = []
sol_grid = []  # speichert tuple (i1_sol, i2_sol) pro Gridpunkt

for j, i_1 in enumerate(i1_vals):
    for k, i_2 in enumerate(i2_vals):
        sol = root(
            equation,
            [i_1, i_2],
            args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        )
        Sol1[j, k] = np.round(sol.x[0], 5)
        Sol2[j, k] = np.round(sol.x[1], 5)
        if sol.success and np.linalg.norm(sol.fun) < 1e-6:
            tup = tuple(np.round(sol.x, 5))
            solutions.append(tup)
            sol_grid.append((j, k, tup))

uniq = sorted(set(solutions))
print("Found unique solutions:", uniq)

# Erstelle Mapping von Tupel -> Index in uniq

idx_map = {tup: idx for idx, tup in enumerate(uniq)}

# Neue Index-Arrays für Plot

Sol1_idx = np.full((n, n), -1, dtype=int)
Sol2_idx = np.full((n, n), -1, dtype=int)

for a, b, tup in sol_grid:
    idx = idx_map.get(tup, -1)
    Sol1_idx[a, b] = idx
    Sol2_idx[a, b] = (
        idx  # Falls du dieselbe Index-Lösung für i2 willst; ansonsten halten wir dasselbe
    )

# Plotten mit Index als Farbdarstellung

plt.figure(figsize=(8, 6))
cp = plt.contourf(I1, I2, Sol1_idx, levels=len(uniq), cmap="viridis")
plt.colorbar(cp, label="Index der i1-Lösung in uniq")
plt.xlabel("i_1")
plt.ylabel("i_2")
plt.title("Gefundene i1-Lösungen: Index in uniq")

plt.figure(figsize=(8, 6))
cp2 = plt.contourf(I1, I2, Sol2_idx, levels=len(uniq), cmap="viridis")
plt.colorbar(cp2, label="Index der i2-Lösung in uniq")
plt.xlabel("i_1")
plt.ylabel("i_2")
plt.title("Gefundene i2-Lösungen: Index in uniq")
