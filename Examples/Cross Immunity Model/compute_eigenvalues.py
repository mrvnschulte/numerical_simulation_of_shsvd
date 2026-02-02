# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:52:35 2025

@author: schulte
"""
from sympy import symbols, Matrix, solve

"""
Computes the eigenvalues and eigenvectors of the 2-variant transmission matrix
"""

# sigma = symbols("sigma")
# r01 = symbols("r01")
# r02 = symbols("r02")

# A = Matrix(
#     [
#         [(1 - sigma) * r01, 0, (1 - sigma) * r01, 0],
#         [0, (1 - sigma) * r02, 0, (1 - sigma) * r02],
#         [sigma * r01, 0, sigma * r01, 0],
#         [0, sigma * r02, 0, sigma * r02],
#     ]
# )

# eigenvectors = A.eigenvects()

# for eigenvalue, multiplicity, vects in eigenvectors:
#     print(
#         f"Eigenvalue: {eigenvalue}, Multiplicity: {multiplicity}, "
#         f"Eigenvectors: {vects}"
#     )

# sigma_12 = symbols("sigma_12")
# sigma_13 = symbols("sigma_13")

# FV = Matrix(
#     [
#         [
#             (1 - sigma_12) * (1 - sigma_13) * r01,
#             (1 - sigma_12) * (1 - sigma_13) * r01,
#             (1 - sigma_12) * (1 - sigma_13) * r01,
#             (1 - sigma_12) * (1 - sigma_13) * r01,
#         ],
#         [
#             sigma_12 * (1 - sigma_13) * r01,
#             sigma_12 * (1 - sigma_13) * r01,
#             sigma_12 * (1 - sigma_13) * r01,
#             sigma_12 * (1 - sigma_13) * r01,
#         ],
#         [
#             sigma_13 * (1 - sigma_12) * r01,
#             sigma_13 * (1 - sigma_12) * r01,
#             sigma_13 * (1 - sigma_12) * r01,
#             sigma_13 * (1 - sigma_12) * r01,
#         ],
#         [
#             sigma_13 * sigma_12 * r01,
#             sigma_13 * sigma_12 * r01,
#             sigma_13 * sigma_12 * r01,
#             sigma_13 * sigma_12 * r01,
#         ],
#     ]
# )

# eigenvectors = FV.eigenvects()

# for eigenvalue, multiplicity, vects in eigenvectors:
#     print(
#         f"Eigenvalue: {eigenvalue}, Multiplicity: {multiplicity}, "
#         f"Eigenvectors: {vects}"
#     )

beta_1 = symbols("beta_1")
beta_2 = symbols("beta_2")
gamma_1 = symbols("gamma_1")
gamma_2 = symbols("gamma_2")
sigma = symbols("sigma")
b = symbols("b")

x_4 = (1 - sigma) * b * (1 / (gamma_2 + b) - 1 / beta_2)
x_5 = sigma * b * (1 / (gamma_2 + b) - 1 / beta_2)
x_6 = (gamma_2 + b) / beta_2
x_8 = gamma_2 / b * (1 - sigma) * b * (1 / (gamma_2 + b) - 1 / beta_2)

first_sum = (
    1 / (gamma_1 + beta_2 * (x_4 + x_5) + b)
    + gamma_2
    * beta_2
    * (x_4 + x_5)
    / (
        (gamma_1 + b)
        * (gamma_1 + beta_2 * (x_4 + x_5) + b)
        * (gamma_1 + gamma_2 + b)
    )
    + beta_2
    * (x_4 + x_5)
    / ((gamma_1 + beta_2 * (x_4 + x_5) + b) * (gamma_1 + gamma_2 + b))
)
second_sum = gamma_2 / ((gamma_1 + b) * (gamma_1 + gamma_2 + b)) + 1 / (
    (gamma_1 + gamma_2 + b)
)

FV_invasion = Matrix(
    [
        [
            (1 - sigma) * beta_1 * x_6 * first_sum,
            (sigma * beta_1 * x_6 + beta_1 * x_8) * first_sum,
            beta_1 * x_4 * first_sum,
        ],
        [
            (1 - sigma) * beta_1 * x_6 / (gamma_1 + b),
            (sigma * beta_1 * x_6 + beta_1 * x_8) / (gamma_1 + b),
            beta_1 * x_4 / (gamma_1 + b),
        ],
        [
            (1 - sigma) * beta_1 * x_6 * second_sum,
            (sigma * beta_1 * x_6 + beta_1 * x_8) * second_sum,
            beta_1 * x_4 * second_sum,
        ],
    ]
).transpose()
print(FV_invasion.eigenvals(simplify=True))

# Solve system of next-generation matrix paper
ny = symbols("ny")
i = symbols("i1:3")
print(
    solve(
        [
            beta_1 * i[0] * (1 - i[0] - i[1])
            - (b + gamma_1) * i[0]
            + ny * i[0] * i[1],
            beta_2 * i[1] * (1 - i[0] - i[1])
            - (b + gamma_2) * i[1]
            - ny * i[0] * i[1],
        ],
        [i[0], i[1]],
    )
)

x = symbols("x1:10")
dx_1 = (
    -gamma_1 * x[0]
    - beta_2 * x[0] * (x[2] + x[3] + x[4])
    + (1 - sigma) * beta_1 * x[5] * (x[0] + x[1] + x[2])
    - b * x[0]
)
dx_2 = (
    -gamma_1 * x[1]
    + sigma * beta_1 * x[5] * (x[0] + x[1] + x[2])
    + beta_1 * x[7] * (x[0] + x[1] + x[2])
    + gamma_2 * x[2]
    - b * x[1]
)
dx_3 = (
    -gamma_1 * x[2]
    - gamma_2 * x[2]
    + beta_1 * x[3] * (x[0] + x[1] + x[2])
    + beta_2 * x[0] * (x[2] + x[3] + x[4])
    - b * x[2]
)
dx_4 = (
    -gamma_2 * x[3]
    - beta_1 * x[3] * (x[0] + x[1] + x[2])
    + (1 - sigma) * beta_2 * x[5] * (x[2] + x[3] + x[4])
    - b * x[3]
)
dx_5 = (
    -gamma_2 * x[4]
    + sigma * beta_2 * x[5] * (x[2] + x[3] + x[4])
    + beta_2 * x[6] * (x[2] + x[3] + x[4])
    + gamma_1 * x[2]
    - b * x[4]
)
dx_6 = (
    -beta_1 * x[5] * (x[0] + x[1] + x[2])
    - beta_2 * x[5] * (x[2] + x[3] + x[4])
    + b
    - b * x[5]
)
dx_7 = -beta_2 * x[6] * (x[2] + x[3] + x[4]) + gamma_1 * x[0] - b * x[6]
dx_8 = -beta_1 * x[7] * (x[0] + x[1] + x[2]) + gamma_2 * x[3] - b * x[7]
dx_9 = gamma_1 * x[1] + gamma_2 * x[4] - b * x[8]

solve(
    [dx_1, dx_2, dx_3, dx_4, dx_5, dx_6, dx_7, dx_8, dx_9],
    *x,
    dict=True,
    simplify=False
)
