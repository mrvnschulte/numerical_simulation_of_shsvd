from sympy import symbols, Matrix
from cross_immunity_model import (
    map_x_to_I,
    variant_in_set_matrix,
    compute_gamma_matrix,
    construct_transition_matrix,
    shift_matrix,
    map_I_to_x,
)
import numpy as np


def build_symbolic_system(n):
    N = frozenset(range(1, n + 1))
    size = 2**n

    x = [symbols(f"x_{k}") for k in range(3**n)]
    gamma = np.array([symbols(f"gamma_{k}") for k in N], dtype=object)
    beta = np.array([symbols(f"beta_{k}") for k in N], dtype=object)
    sigma = np.array(
        [[symbols(f"sigma_{k}{m}") for m in N] for k in N],
        dtype=object,
    )
    b = symbols("b")

    I = map_x_to_I(x, dtype=object)

    V = variant_in_set_matrix(n)

    # Compute the infectious with each variant
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    # Compute the recoveries for each infection set
    gamma_sum = (V.T @ gamma).flatten()
    # Compute the transmission factor for each variant
    transmission_vector = beta * I_variant
    # TO-DO: Comment
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    gamma_matrix = compute_gamma_matrix(n, gamma)
    T = construct_transition_matrix(n, sigma, dtype=object)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T

    dI = (
        -(gamma_sum * I.T).T
        - beta_sum * I
        + gamma_matrix @ I
        + np.sum(S @ I @ W, axis=0)
        - b * I
    )
    dI[0, 0] += b

    dx = map_I_to_x(dI, dtype=object)

    return dx, x, beta, gamma, sigma, b


if __name__ == "__main__":
    dx, x, beta, gamma, sigma = build_symbolic_system(3)

    dx_mat = Matrix(dx)
    F = dx_mat.jacobian(x)
    sub_dict = dict(zip(beta, [0.3, 0.35, 0.3]))
    dx_mat = dx_mat.subs(sub_dict)
