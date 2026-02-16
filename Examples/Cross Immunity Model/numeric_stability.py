import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def two_variant_model(t, x, beta1, beta2, gamma1, gamma2, sigma, b):

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x

    f1 = (
        -gamma1 * x1
        - beta2 * x1 * (x3 + x4 + x5)
        + (1 - sigma) * beta1 * x6 * (x1 + x2 + x3)
        - b * x1
    )
    f2 = (
        -gamma1 * x2
        + sigma * beta1 * x6 * (x1 + x2 + x3)
        + beta1 * x8 * (x1 + x2 + x3)
        + gamma2 * x3
        - b * x2
    )
    f3 = (
        -gamma1 * x3
        - gamma2 * x3
        + beta1 * x4 * (x1 + x2 + x3)
        + beta2 * x1 * (x3 + x4 + x5)
        - b * x3
    )
    f4 = (
        -gamma2 * x4
        - beta1 * x4 * (x1 + x2 + x3)
        + (1 - sigma) * beta2 * x6 * (x3 + x4 + x5)
        - b * x4
    )
    f5 = (
        -gamma2 * x5
        + sigma * beta2 * x6 * (x3 + x4 + x5)
        + beta2 * x7 * (x3 + x4 + x5)
        + gamma1 * x3
        - b * x5
    )
    f6 = (
        -beta1 * x6 * (x1 + x2 + x3) - beta2 * x6 * (x3 + x4 + x5) + b - b * x6
    )
    f7 = -beta2 * x7 * (x3 + x4 + x5) + gamma1 * x1 - b * x7
    f8 = -beta1 * x8 * (x1 + x2 + x3) + gamma2 * x4 - b * x8
    f9 = gamma1 * x2 + gamma2 * x5 - b * x9

    f = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    return f


def lyapunov_candidate(x, x_bar, c):

    V = sum(
        c[i] * (x[i] - x_bar[i] - x_bar[i] * np.log(x[i] / x_bar[i]))
        for i in range(9)
    )

    return V


def lyapunov_derivative(
    t, x, x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b
):
    """
    Berechnet dV/dt für lyapunov_candidate an der Stelle x.

    t: Zeit (wird nur für Kompatibilität mit solve_ivp übergeben)
    x: aktueller Zustandsvektor (Länge 9)
    x_bar: Gleichgewicht (Länge 9)
    c: Gewichte (Länge 9)
    übrige Parameter: wie im two_variant_model
    """
    # rechte Seite des ODE-Systems
    f = two_variant_model(t, x, beta1, beta2, gamma1, gamma2, sigma, b)

    # dV/dt = sum_i c_i (1 - x_bar_i/x_i) * f_i
    dVdt = 0.0
    for i in range(9):
        dVdt += c[i] * (1.0 - x_bar[i] / x[i]) * f[i]

    return dVdt


def second_lyapunov_candidate(x, x_bar, c):

    i = [x[0] + x[1] + x[2], x[2] + x[3] + x[4]]
    i_bar = [x_bar[0] + x_bar[1] + x_bar[2], x_bar[2] + x_bar[3] + x_bar[4]]
    s = [x[3] + x[5] + x[7], x[0] + x[5] + x[6]]
    r = [x[4] + x[6] + x[8], x[1] + x[7] + x[8]]
    s_bar = [x_bar[3] + x_bar[5] + x_bar[7], x_bar[0] + x_bar[5] + x_bar[6]]
    r_bar = [x_bar[4] + x_bar[6] + x_bar[8], x_bar[1] + x_bar[7] + x_bar[8]]

    V = (
        i[0]
        - i_bar[0]
        - i_bar[0] * np.log(i[0] / i_bar[0])
        + i[1]
        - i_bar[1]
        - i_bar[1] * np.log(i[1] / i_bar[1])
        + s[0]
        - s_bar[0]
        - s_bar[0] * np.log(s[0] / s_bar[0])
        + s[1]
        - s_bar[1]
        - s_bar[1] * np.log(s[1] / s_bar[1])
        + r[0]
        - r_bar[0]
        - r_bar[0] * np.log(r[0] / r_bar[0])
        + r[1]
        - r_bar[1]
        - r_bar[1] * np.log(r[1] / r_bar[1])
    )

    return V


def L_function(
    x, x_bar, i_bar, b, H1, H2, H3, K1, K2, K3, F1, F2, F3, F4, G1, G2, G3, G4
):

    u = np.array(x) / np.array(x_bar)
    i = [x[0] + x[1] + x[2], x[2] + x[3] + x[4]]
    Phi = np.array(i) / np.array(i_bar)

    L = (
        b * np.log(1 / u[5])
        + H1 * np.log(u[0] / u[6])
        + H2 * np.log(u[1] / u[8])
        + H3 * np.log(u[2] / u[4])
        + K1 * np.log(u[2] / u[1])
        + K2 * np.log(u[3] / u[7])
        + K3 * np.log(u[4] / u[8])
        + Phi[0]
        * (
            F1 * np.log(u[5] / u[0])
            + F2 * np.log(u[5] / u[1])
            + F3 * np.log(u[7] / u[1])
            + F4 * np.log(u[3] / u[2])
        )
        + Phi[1]
        * (
            G1 * np.log(u[5] / u[3])
            + G2 * np.log(u[5] / u[4])
            + G3 * np.log(u[6] / u[4])
            + G4 * np.log(u[0] / u[2])
        )
    )

    return L


t0 = 0
tf = 400

t_eval = np.linspace(t0, tf, tf + 1)

x0 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.992, 0.001, 0.001, 0.001]

beta1 = 0.3
beta2 = 0.35
gamma1 = 0.1
gamma2 = 0.1
sigma = 0.5
b = 0.01
c = [1, 1, 2, 1, 1, 1, 1, 1, 1]

sol = solve_ivp(
    two_variant_model,
    (t0, tf),
    x0,
    t_eval=t_eval,
    args=(beta1, beta2, gamma1, gamma2, sigma, b),
)

plt.plot(t_eval, sol.y.T)
plt.show()
x_bar = sol.y.T[-1]

lyapunov = [lyapunov_candidate(x, x_bar, c) for x in sol.y.T]
lyapunov2 = [second_lyapunov_candidate(x, x_bar, c) for x in sol.y.T]


plt.plot(t_eval, lyapunov)
plt.plot(t_eval, lyapunov2)
plt.show()


i_bar = [x_bar[0] + x_bar[1] + x_bar[2], x_bar[2] + x_bar[3] + x_bar[4]]

H1 = gamma1 * x_bar[0]
H2 = gamma1 * x_bar[1]
H3 = gamma1 * x_bar[2]
K1 = gamma2 * x_bar[2]
K2 = gamma2 * x_bar[3]
K3 = gamma2 * x_bar[4]
F1 = (1 - sigma) * beta1 * x_bar[5] * i_bar[0]
F2 = sigma * beta1 * x_bar[5] * i_bar[0]
F3 = beta1 * x_bar[7] * i_bar[0]
F4 = beta1 * x_bar[3] * i_bar[0]
G1 = (1 - sigma) * beta2 * x_bar[5] * i_bar[1]
G2 = sigma * beta2 * x_bar[5] * i_bar[1]
G3 = beta2 * x_bar[6] * i_bar[1]
G4 = beta2 * x_bar[0] * i_bar[1]

L_array = [
    L_function(
        x,
        x_bar,
        i_bar,
        b,
        H1,
        H2,
        H3,
        K1,
        K2,
        K3,
        F1,
        F2,
        F3,
        F4,
        G1,
        G2,
        G3,
        G4,
    )
    for x in sol.y.T
]

plt.plot(t_eval, L_array)
plt.show()

import numpy as np
from scipy.optimize import minimize


def dot_V_numerical(u, params):
    """
    Berechnet dot_V für gegebene u und Parameter.

    u: array [u1, u2, ..., u9]
    params: dict mit F1, F2, ..., alpha1, ..., bar_x, etc.
    """
    u1, u2, u3, u4, u5, u6, u7, u8, u9 = u

    F1, F2, F3, F4 = params["F"]
    G1, G2, G3, G4 = params["G"]
    H1, H2, H3 = params["H"]
    K1, K2, K3 = params["K"]
    b = params["b"]
    bar_x = params["bar_x"]
    bar_S1 = bar_x[0] + bar_x[1] + bar_x[2]
    bar_S2 = bar_x[2] + bar_x[3] + bar_x[4]

    alpha1, alpha2, alpha3 = (
        bar_x[0] / bar_S1,
        bar_x[1] / bar_S1,
        bar_x[2] / bar_S1,
    )
    beta3, beta4, beta5 = (
        bar_x[2] / bar_S2,
        bar_x[3] / bar_S2,
        bar_x[4] / bar_S2,
    )

    phi1 = alpha1 * u1 + alpha2 * u2 + alpha3 * u3
    phi2 = beta3 * u3 + beta4 * u4 + beta5 * u5

    dV = (
        b * (1 - 1 / u6)
        + H1 * (1 - u1 / u7)
        + H2 * (1 - u2 / u9)
        + H3 * (1 - u3 / u5)
        + K1 * (1 - u3 / u2)
        + K2 * (1 - u4 / u8)
        + K3 * (1 - u5 / u9)
        + phi1
        * (
            F1 * (1 - u6 / u1)
            + F2 * (1 - u6 / u2)
            + F3 * (1 - u8 / u2)
            + F4 * (1 - u4 / u3)
        )
        + phi2
        * (
            G1 * (1 - u6 / u4)
            + G2 * (1 - u6 / u5)
            + G3 * (1 - u7 / u5)
            + G4 * (1 - u1 / u3)
        )
    )

    return dV


def verify_dot_V_negative(params, n_samples=10000):
    """
    Verifiziert numerisch, dass dot_V <= 0 für zufällige u mit sum(bar_x * u) = 1.
    """
    bar_x = params["bar_x"]
    max_dV = -np.inf
    worst_u = None

    for _ in range(n_samples):
        # Generiere zufälliges u mit Nebenbedingung
        u = np.random.exponential(1, 9)
        u = u / np.dot(bar_x, u)  # Normiere so dass sum(bar_x * u) = 1

        dV = dot_V_numerical(u, params)

        if dV > max_dV:
            max_dV = dV
            worst_u = u.copy()

    return max_dV, worst_u


# Beispielparameter (aus Gleichgewicht berechnet)

params_example = {
    "F": [F1, F2, F3, F4],
    "G": [G1, G2, G3, G4],
    "H": [H1, H2, H3],
    "K": [K1, K2, K3],
    "b": b,
    "bar_x": np.array(x_bar),
}

max_dV, worst_u = verify_dot_V_negative(params_example)
print(f"Maximales dot_V gefunden: {max_dV:.10f}")
print(f"Bei u = {worst_u}")

worst_x = worst_u * x_bar
print(f"Bei x = {worst_x}")

sol = solve_ivp(
    two_variant_model,
    (t0, tf),
    worst_x,
    t_eval=t_eval,
    args=(beta1, beta2, gamma1, gamma2, sigma, b),
)

plt.plot(t_eval, sol.y.T)
plt.show()
x_bar = sol.y.T[-1]

lyapunov = [lyapunov_candidate(x, x_bar, c) for x in sol.y.T]
lyapunov2 = [second_lyapunov_candidate(x, x_bar, c) for x in sol.y.T]


plt.plot(t_eval, lyapunov)
plt.plot(t_eval, lyapunov2)
plt.show()


# def random_point_on_simplex(n):
#     """
#     Zufälliger Punkt auf dem Simplex {x_i > 0, sum x_i = 1}.
#     """
#     v = np.random.exponential(1.0, n)
#     return v / v.sum()


# def find_max_lyapunov_derivative(
#     x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b, n_starts=20, eps=1e-8
# ):
#     """
#     Findet numerisch das Maximum von dV/dt auf dem Simplex
#         S = { x in R^9_+ : sum x_i = 1 }.

#     x_bar, c: wie oben
#     Parameter: wie im Modell
#     n_starts: Anzahl zufälliger Startpunkte (Multi-Start)
#     eps: untere Schranke für x_i, um Division durch 0 zu vermeiden
#     """

#     def objective(x):
#         # Sicherstellen, dass wir nicht in den Bereich x_i <= 0 geraten
#         x_clipped = np.clip(x, eps, 1.0)
#         x_clipped = (
#             x_clipped / x_clipped.sum()
#         )  # zurück auf den Simplex projizieren
#         return -lyapunov_derivative(
#             0.0, x_clipped, x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b
#         )

#     # Nebenbedingung: sum x_i = 1
#     cons = (
#         {
#             "type": "eq",
#             "fun": lambda x: np.sum(x) - 1.0,
#         },
#     )

#     # Schranken: x_i >= eps
#     bounds = [(eps, 1.0) for _ in range(9)]

#     best_val = -np.inf
#     best_x = None

#     for _ in range(n_starts):
#         x0 = random_point_on_simplex(9)

#         res = minimize(
#             objective,
#             x0,
#             method="SLSQP",
#             bounds=bounds,
#             constraints=cons,
#             options={"ftol": 1e-9, "maxiter": 1000},
#         )

#         if not res.success:
#             continue

#         x_opt = res.x
#         x_opt = np.clip(x_opt, eps, 1.0)
#         x_opt = x_opt / x_opt.sum()

#         dVdt_val = lyapunov_derivative(
#             0.0, x_opt, x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b
#         )

#         if dVdt_val > best_val:
#             best_val = dVdt_val
#             best_x = x_opt

#     return best_val, best_x


# # Beispielaufruf (nachdem du x_bar, c, beta1, ... bereits definiert hast)

# max_dV, argmax_x = find_max_lyapunov_derivative(
#     x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b, n_starts=50
# )

# print("Maximales dV/dt:", max_dV)
# print("bei x =", argmax_x)

# # Optional: Trajektorie von diesem Startpunkt simulieren

# sol_worst = solve_ivp(
#     two_variant_model,
#     (t0, tf),
#     argmax_x,
#     t_eval=t_eval,
#     args=(beta1, beta2, gamma1, gamma2, sigma, b),
# )

# dV_worst = [
#     lyapunov_derivative(t, x, x_bar, c, beta1, beta2, gamma1, gamma2, sigma, b)
#     for t, x in zip(t_eval, sol_worst.y.T)
# ]

# plt.figure()
# plt.plot(t_eval, dV_worst)
# plt.axhline(0, color="k", linewidth=0.5)
# plt.xlabel("t")
# plt.ylabel("dV/dt")
# plt.show()


# def minimize_max_dVdt_over_c(
#     x_bar,
#     beta1,
#     beta2,
#     gamma1,
#     gamma2,
#     sigma,
#     b,
#     n_outer_starts=10,
#     n_inner_starts=20,
# ):
#     """
#     Minimiert numerisch max_x dV/dt(x; c) über c>0 (mit Normierung von c).

#     Rückgabe:
#         best_max_dVdt : minimiertes Maximum von dV/dt
#         best_c        : zugehöriger c-Vektor (normiert)
#         best_x        : x, an dem dieses Maximum ungefähr angenommen wird
#     """

#     def normalize_c(c_raw):
#         # Erzwinge Positivität
#         c_pos = np.clip(c_raw, 1e-6, None)
#         # Normiere z.B. auf Summe 1
#         return c_pos / c_pos.sum()

#     def objective_c(c_raw):
#         c_norm = normalize_c(c_raw)
#         max_dV, _ = find_max_lyapunov_derivative(
#             x_bar,
#             c_norm,
#             beta1,
#             beta2,
#             gamma1,
#             gamma2,
#             sigma,
#             b,
#             n_starts=n_inner_starts,
#         )
#         return max_dV  # Minimieren dieses Maximums

#     bounds_c = [(1e-6, 10.0) for _ in range(9)]
#     best_val = np.inf
#     best_c = None
#     best_x = None

#     for _ in range(n_outer_starts):
#         # zufälliger Start für c
#         c0 = np.random.uniform(0.5, 1.5, 9)

#         res = minimize(
#             objective_c,
#             c0,
#             method="SLSQP",
#             bounds=bounds_c,
#             options={"ftol": 1e-3, "maxiter": 50},
#         )
#         if not res.success:
#             continue

#         c_candidate = normalize_c(res.x)
#         max_dV_candidate, x_candidate = find_max_lyapunov_derivative(
#             x_bar,
#             c_candidate,
#             beta1,
#             beta2,
#             gamma1,
#             gamma2,
#             sigma,
#             b,
#             n_starts=n_inner_starts,
#         )

#         if max_dV_candidate < best_val:
#             best_val = max_dV_candidate
#             best_c = c_candidate
#             best_x = x_candidate

#     return best_val, best_c, best_x


# best_max_dVdt, best_c, worst_x_for_best_c = minimize_max_dVdt_over_c(
#     x_bar,
#     beta1,
#     beta2,
#     gamma1,
#     gamma2,
#     sigma,
#     b,
#     n_outer_starts=10,
#     n_inner_starts=20,
# )

# print("Minimiertes Maximum von dV/dt:", best_max_dVdt)
# print("Optimales c:", best_c)
# print("x, an dem das Maximum (für dieses c) auftritt:", worst_x_for_best_c)

import sympy as sp
import numpy as np

# SymPy-Variablen und Parameter

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9")
gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
    "gamma1 gamma2 beta1 beta2 sigma b", positive=True
)

i1_expr = x1 + x2 + x3
i2_expr = x3 + x4 + x5

f1 = (
    -gamma1 * x1
    - beta2 * x1 * i2_expr
    + (1 - sigma) * beta1 * x6 * i1_expr
    - b * x1
)
f2 = (
    -gamma1 * x2
    + sigma * beta1 * x6 * i1_expr
    + beta1 * x8 * i1_expr
    + gamma2 * x3
    - b * x2
)
f3 = (
    -gamma1 * x3
    - gamma2 * x3
    + beta1 * x4 * i1_expr
    + beta2 * x1 * i2_expr
    - b * x3
)
f4 = (
    -gamma2 * x4
    - beta1 * x4 * i1_expr
    + (1 - sigma) * beta2 * x6 * i2_expr
    - b * x4
)
f5 = (
    -gamma2 * x5
    + sigma * beta2 * x6 * i2_expr
    + beta2 * x7 * i2_expr
    + gamma1 * x3
    - b * x5
)
f6 = -beta1 * x6 * i1_expr - beta2 * x6 * i2_expr + b - b * x6
f7 = -beta2 * x7 * i2_expr + gamma1 * x1 - b * x7
f8 = -beta1 * x8 * i1_expr + gamma2 * x4 - b * x8
f9 = gamma1 * x2 + gamma2 * x5 - b * x9

F = sp.Matrix([f1, f2, f3, f4, f5, f6, f7, f8, f9])
X = sp.Matrix([x1, x2, x3, x4, x5, x6, x7, x8, x9])

# Jacobi-Matrix

J_x = F.jacobian(X)

# --- Parameterwerte einsetzen (aus der Simulation) ---

param_subs = {
    beta1: 0.3,
    beta2: 0.35,
    gamma1: 0.1,
    gamma2: 0.1,
    sigma: 0.5,
    b: 0.01,
}

J_param = J_x.subs(param_subs)

# --- Gleichgewichtsnäherung x_bar einsetzen ---

# Angenommen, x_bar = sol.y[:, -1] aus obigem Simulationscode

# (hier z.B. hart kodiert oder via Übergabe)

# Beispiel: direkt als Numpy-Array übergeben

# x_bar = np.array([...])  # in deinem Code kommt das aus solve_ivp

x_subs = {
    x1: float(x_bar[0]),
    x2: float(x_bar[1]),
    x3: float(x_bar[2]),
    x4: float(x_bar[3]),
    x5: float(x_bar[4]),
    x6: float(x_bar[5]),
    x7: float(x_bar[6]),
    x8: float(x_bar[7]),
    x9: float(x_bar[8]),
}

J_star_num = J_param.subs(x_subs).evalf()

print("Jacobi-Matrix J(x_bar):")
sp.pprint(J_star_num)

# --- Eigenwerte von J(x_bar) ---

eigs_num = J_star_num.eigenvals()
print("\nEigenwerte (numerisch):")
for ev, mult in eigs_num.items():
    print("Eigenwert:", complex(ev), "Multiplizität:", mult)
