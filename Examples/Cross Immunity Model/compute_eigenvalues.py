# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:52:35 2025

@author: schulte
"""
from sympy import symbols, Matrix, simplify

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
