import sympy as sp

# Ursprungsvariablen

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9")
gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
    "gamma1 gamma2 beta1 beta2 sigma b", positive=True
)

# i1, i2

i1_expr = x1 + x2 + x3
i2_expr = x3 + x4 + x5

# Rechte Seiten f1,...,f9

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

F = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

J = sp.Matrix(F).jacobian(xs)

print(J)
sp.print_latex(J)

# # ------------------------------------------------------------------

# # Lyapunov-Funktion V(x) = sum_i c_i (x_i - xbar_i - xbar_i ln(x_i/xbar_i))

# # ------------------------------------------------------------------

# # Symbole für Gleichgewichtswerte x̄_i und Gewichte c_i

# xbar = sp.symbols("xbar1:10", positive=True)  # xbar1,...,xbar9
# c = sp.symbols("c1:10", positive=True)  # c1,...,c9

# # V definieren

# V = sum(
#     c[i] * (xs[i] - xbar[i] - xbar[i] * sp.log(xs[i] / xbar[i]))
#     for i in range(9)
# )

# # Ableitung von V entlang der Trajektorien: V̇ = ∑ (∂V/∂x_i) f_i

# V_dot = sum(sp.diff(V, xs[i]) * F[i] for i in range(9))

# # Vereinfachen (das kann etwas dauern)

# V_dot_simplified = sp.simplify(V_dot)

# print(f"Lyapunov-Funktion V(x): {V}")
# # sp.pprint(V)

# print(f"\nZeitableitung V_dot(x): {V_dot_simplified}")
# # sp.pprint(V_dot_simplified)

# grad_V_dot = sp.Matrix([sp.diff(V_dot, xs[i]) for i in range(9)])
# grad_V_dot_simplified = sp.Matrix(
#     [sp.simplify(sp.diff(V_dot, xs[i])) for i in range(9)]
# )
