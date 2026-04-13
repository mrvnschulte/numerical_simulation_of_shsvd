from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, least_squares
import sympy as sp

from cross_immunity_model import (
    is_subset,
    variant_in_set_matrix,
    compute_gamma_matrix,
    construct_transition_matrix,
    shift_matrix,
    map_I_to_x,
)
from cross_immunity_model_symbolic import build_symbolic_system
from invasion_reproduction_number import invasion_reproduction_number


def hamming_weight_iterator(size):
    for weight in range(size.bit_length()):
        for j in range(size):
            if bin(j).count("1") == weight:
                yield j


def compute_equilibrium(I_variant, beta, gamma, sigma, V, G, T, S, b):

    n_variants = len(I_variant)
    size = 2**n_variants

    I_star = np.zeros((size, size))

    transmission_vector = beta * I_variant
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    W = transmission_vector[:, np.newaxis, np.newaxis] * T
    gamma_sum = (V.T @ gamma).flatten()

    I_star[0, 0] = b / (gamma_sum[0] + beta_sum[0] + b)

    iteration_list = list(hamming_weight_iterator(size))

    for j in iteration_list:
        if j == 0:
            continue
        for i in reversed(iteration_list):
            if is_subset(i, j):
                I_star[i, j] = (
                    np.sum(S @ I_star @ W, axis=0)[i, j] + (G @ I_star)[i, j]
                ) / (gamma_sum[i] + beta_sum[j] + b)

    return I_star


def equilibrium_equation(I_variant, beta, gamma, sigma, V, G, T, S, b):

    I_star = compute_equilibrium(I_variant, beta, gamma, sigma, V, G, T, S, b)
    n_variants = len(beta)
    size = 2**n_variants
    I_star_variant = (V @ I_star @ np.ones((size, 1))).flatten()
    return I_star_variant - I_variant


def solve_equilibrium_least_squares(
    beta, gamma, sigma, V, G, T, S, b, I_variant_guess, bounds=None
):
    """
    Lösung der Gleichung equilibrium als Least-Squares-Problem mit optionalen Bounds.
    - I_variant_guess: Startwert als 1D-Array der Länge size
    - bounds: Tuple (lower, upper) oder None. Beide sind 1D-Arrays gleicher Länge wie I_variant_guess

    """
    if bounds is None:
        bounds = (None, None)

    lb, ub = bounds
    # Falls Bounds nicht gegeben, setze leer (unbeschränkt)
    if lb is not None or ub is not None:
        # least_squares erwartet Bounds als zwei Arrays gleicher Länge wie x
        res = least_squares(
            equilibrium_equation,
            I_variant_guess,
            bounds=(lb, ub),
            args=(beta, gamma, sigma, V, G, T, S, b),
        )
    else:
        res = least_squares(
            equilibrium_equation,
            I_variant_guess,
            args=(beta, gamma, sigma, V, G, T, S, b),
        )

    return res


if __name__ == "__main__":
    n_variants = 3
    size = 2**n_variants
    beta = [0.35, 0.29, 0.3]
    gamma = [0.1, 0.1, 0.1]
    sigma_12 = 0.9
    sigma_23 = 0.95
    sigma = np.array(
        [
            [1, sigma_12, sigma_12 * sigma_23],
            [sigma_12, 1, sigma_23],
            [sigma_12 * sigma_23, sigma_23, 1],
        ]
    )
    b = 0.01
    V = variant_in_set_matrix(n_variants)
    G = compute_gamma_matrix(n_variants, gamma)
    T = construct_transition_matrix(n_variants, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n_variants)])

    I_variant_guess = [0.4, 0.1, 0]

    sol = solve_equilibrium_least_squares(
        beta, gamma, sigma, V, G, T, S, b, I_variant_guess, bounds=None
    )
    print(f"Found solution: {sol.x}")

    I_eq = compute_equilibrium(sol.x, beta, gamma, sigma, V, G, T, S, b)
    x_eq = map_I_to_x(I_eq)

    S_variant = (
        np.ones((size, 1)).T @ I_eq @ (np.ones_like(V) - V).T
    ).flatten()
    r0 = np.array(beta) / (np.array(gamma) + b)
    r0_3to12 = S_variant[2] * r0[2]
    r0_3to1 = invasion_reproduction_number(r0[2], r0[0], sigma[1, 0])
    r0_3to2 = invasion_reproduction_number(r0[2], r0[1], sigma[2, 1])
    print(r0_3to12)
    print(r0_3to1)
    print(r0_3to2)

    dx, x, beta_symb, gamma_symb, sigma_symb, b_symb = build_symbolic_system(
        n_variants
    )

    dx_mat = sp.Matrix(dx)
    J_symb = dx_mat.jacobian(x)
    sub_dict = (
        dict(zip(beta_symb, beta))
        | dict(zip(gamma_symb, gamma))
        | dict(zip(sigma_symb.flatten(), sigma.flatten()))
        | dict(zip(x, x_eq))
        | {b_symb: b}
    )

    J = J_symb.subs(sub_dict)
    J_num = np.array(J.evalf().tolist(), dtype=float)
    eigvals, eigvects = np.linalg.eig(J_num)
    max_real_part = np.max(eigvals.real)
    print(f"Eigenvalues: {eigvals}")
    print(f"With maximal real part: {max_real_part}")

    # resolution = 101
    # beta_3 = np.linspace(0.1, 0.3, resolution)
    # sigma_12 = 0.5
    # sigma_23 = np.linspace(0, 1, resolution)
    # gamma = np.array([0.1, 0.1, 0.1])
    # b = 0.01
    # r0_3to12 = np.zeros((len(beta_3), len(sigma_23)))
    # r0_3to2 = np.zeros((len(beta_3), len(sigma_23)))
    # r0_3to1 = np.zeros((len(beta_3), len(sigma_23)))
    # I_star_3 = []

    # for i, b3 in enumerate(beta_3):
    #     for k, s in enumerate(sigma_23):
    #         beta = np.array([0.3, 0.35, b3])
    #         n_variants = len(gamma)
    #         size = 2**n_variants
    #         r0 = [beta[i] / (gamma[i] + b) for i in range(n_variants)]
    #         sigma = np.array(
    #             [
    #                 [1, sigma_12, sigma_12 * s],
    #                 [sigma_12, 1, s],
    #                 [sigma_12 * s, s, 1],
    #             ]
    #         )
    #         V = variant_in_set_matrix(n_variants)
    #         G = compute_gamma_matrix(n_variants, gamma)
    #         T = construct_transition_matrix(n_variants, sigma)
    #         S = np.array([shift_matrix(size, 2**m) for m in range(n_variants)])

    #         sol = root(
    #             equilibrium_equation,
    #             [1, 1, 0],
    #             args=(beta, gamma, sigma, V, G, T, S, b),
    #         )
    #         I_variant = sol.x
    #         if (
    #             I_variant[0] > 0
    #             and I_variant[1] > 0
    #             and np.isclose(I_variant[2], 0)
    #         ):
    #             I_star = compute_equilibrium(
    #                 I_variant, beta, gamma, sigma, V, G, T, S, b
    #             )

    #             S_variant = (
    #                 np.ones((size, 1)).T @ I_star @ (np.ones_like(V) - V).T
    #             ).flatten()
    #             r0_3to12[i, k] = S_variant[2] * r0[2]
    #             r0_3to1[i, k] = invasion_reproduction_number(
    #                 r0[2], r0[0], sigma[2, 0]
    #             )
    #             r0_3to2[i, k] = invasion_reproduction_number(
    #                 r0[2], r0[1], sigma[2, 1]
    #             )

    #             sol = root(
    #                 equilibrium_equation,
    #                 [1, 1, 1],
    #                 args=(beta, gamma, sigma, V, G, T, S, b),
    #             )
    #             I_star_3.append(sol.x[2])

    #         else:
    #             print(I_variant)

    #     plt.plot(sigma_23, r0_3to12[i], label=r"$R_{0,3\to12}$")
    #     plt.plot(sigma_23, r0_3to1[i], label=r"$R_{0,3\to1}$")
    #     plt.plot(sigma_23, r0_3to2[i], label=r"$R_{0,3\to2}$")
    #     plt.axhline(1, linestyle="dashed", color="gray")
    #     plt.legend()
    #     plt.show()
    #     # plt.plot(sigma_23, I_star_3)

    # colors = np.zeros((len(beta_3), len(sigma_23)), dtype=int)

    # for i in range(len(beta_3)):
    #     for k in range(len(sigma_23)):
    #         cond12 = r0_3to12[i, k] > 1
    #         cond2 = r0_3to2[i, k] > 1

    #         if cond2 and cond12:
    #             colors[i, k] = 0
    #         elif cond2 and not cond12:
    #             colors[i, k] = 1
    #         elif (not cond2) and cond12:
    #             colors[i, k] = 2
    #         else:
    #             colors[i, k] = 3

    # cmap = ListedColormap(["white", "blue", "green", "red"])

    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     colors,
    #     origin="lower",
    #     aspect="auto",
    #     extent=[min(sigma_23), max(sigma_23), min(beta_3), max(beta_3)],
    #     cmap=cmap,
    #     vmin=0,
    #     vmax=3,
    # )
    # legend_handles = [
    #     Patch(color="white", label="Can invade 2 and 12"),
    #     Patch(color="blue", label="Can invade 2, but not 12"),
    #     Patch(color="green", label="Can invade 12, but not 2"),
    #     Patch(color="red", label="Can invade neither"),
    # ]
    # plt.legend(handles=legend_handles, loc="upper left", frameon=True)

    # plt.xlabel(r"$\sigma_{23}$")
    # plt.ylabel(r"$\beta_3$")
    # plt.title("Invasion ability of the third variant")
    # plt.tight_layout()
    # plt.show()

    # # plt.figure(figsize=(8, 6))
    # # plt.contourf(
    # #     colors,
    # #     origin="lower",
    # #     aspect="auto",
    # #     extent=[min(sigma_23), max(sigma_23), min(beta_3), max(beta_3)],
    # #     cmap=cmap,
    # #     vmin=0,
    # #     vmax=3,
    # # )
    # # legend_handles = [
    # #     Patch(color="white", label="Can invade 2 and 12"),
    # #     Patch(color="blue", label="Can invade 2, but not 12"),
    # #     Patch(color="green", label="Can invade 12, but not 2"),
    # #     Patch(color="red", label="Can invade neither"),
    # # ]
    # # plt.legend(handles=legend_handles, loc="upper left", frameon=True)

    # # plt.xlabel(r"$\sigma_{23}$")
    # # plt.ylabel(r"$\beta_3$")
    # # plt.title("Invasion ability of the third variant")
    # # plt.tight_layout()
    # # plt.show()
