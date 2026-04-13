import sympy as sp

"""
Computes the linearized system of the two-variant model symbolically
"""

# Define symbolic variables

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9")
gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
    "gamma1 gamma2 beta1 beta2 sigma b", positive=True
)

i1_expr = x1 + x2 + x3
i2_expr = x3 + x4 + x5

# Define RHS of the system

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
