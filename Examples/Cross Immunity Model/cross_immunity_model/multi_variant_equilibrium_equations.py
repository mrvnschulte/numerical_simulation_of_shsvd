"""
Defines functions to compute the state at given equilibrium as well as
functions to find the equilibrium and one computing the eigenvalues of the
linearized system at equilibrium to check for stability.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
import sympy as sp

from .model_transformations import (
    infectious_per_variant,
    map_I_to_x,
    project_n_variants,
    reduce_variants,
)
from .symbolic_system import build_symbolic_system
from .utils import hamming_weight_iterator, is_subset, number_of_variants


def multi_variant_equilibrium_compartments(
    I_variant: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Computes the compartment sizes for multi-variant equilibria given the
    infectious fractions per variant.

    Parameters
    ----------
    I_variant : np.ndarray
        Infectious fraction per variant.
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.

    Returns
    -------
    I_star : np.ndarray
        Compartment sizes.
    """
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


def multi_variant_equilibrium_equation(
    I_variant: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Computes the difference between the given I_variant and the computed one
    according to the compartment equilibrium equations for the given I_variant.

    The function can be used to solve for the equilibrium I_variant.

    Parameters
    ----------
    I_variant : np.ndarray
        Infectious fraction per variant.
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.

    Returns
    -------
    diff : np.ndarray
        Difference between given and computed I_variant.
    """
    I_star = multi_variant_equilibrium_compartments(
        I_variant, beta, gamma, sigma, V, G, T, S, b
    )
    I_star_variant = infectious_per_variant(I_star, V, matrix_form=True)
    diff = I_star_variant - I_variant
    return diff


def solve_multi_variant_equilibrium_equation(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    I_variant_guess: ArrayLike,
    free_indices: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (0, 1),
) -> np.ndarray:
    """
    Solves the multi-variant equilibrium equation with given initial guess and
    optionally also the set of free variants and bounds.

    Parameters
    ----------
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.
    I_variant_guess : array-like
        Initial guess of equilibrium I_variant.
    free_indices : array-like or None, optional
        Indices of the variables which could exist. If None is given, all are
        used. The default is None.
    bounds : tuple[ArrayLike, ArrayLike], optional
        Bounds on the equilibrium I_variant. The default is (0, 1).

    Returns
    -------
    I_variant : np.ndarray
        Equilibrium I_variant.
    """
    n = len(I_variant_guess)

    lb, ub = bounds
    if not np.isscalar(lb):
        lb = reduce_variants(lb, free_indices)
    if not np.isscalar(ub):
        ub = reduce_variants(ub, free_indices)
    bounds = (lb, ub)

    def multi_variant_equilibrium_equation_reduced(I_variant_free):
        I_variant_full = project_n_variants(I_variant_free, n, free_indices)
        return multi_variant_equilibrium_equation(
            I_variant_full, beta, gamma, sigma, V, G, T, S, b
        )

    I_variant_guess = reduce_variants(I_variant_guess, free_indices)

    res = least_squares(
        multi_variant_equilibrium_equation_reduced,
        I_variant_guess,
        bounds=bounds,
    )

    I_variant = project_n_variants(res.x, n, free_indices)

    return I_variant


def find_multi_variant_equilibrium(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    I_variant_guess: ArrayLike,
    free_indices: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the multi-variant equilibrium equation with given initial guess and
    optionally also the set of free variants and bounds and computes the state
    at equilibrium.

    Parameters
    ----------
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.
    I_variant_guess : array-like
        Initial guess of equilibrium I_variant.
    free_indices : array-like or None, optional
        Indices of the variables which could exist. If None is given, all are
        used. The default is None.
    bounds : tuple[ArrayLike, ArrayLike], optional
        Bounds on the equilibrium I_variant. The default is (0, 1).

    Returns
    -------
    I_variant : np.ndarray
        Equilibrium I_variant.
    I_eq : np.ndarray
        Equilibrium state in matrix form.
    x_eq : np.ndarray
        Equilibrium state in vector form.

    """
    I_variant = solve_multi_variant_equilibrium_equation(
        beta,
        gamma,
        sigma,
        V,
        G,
        T,
        S,
        b,
        I_variant_guess,
        free_indices=free_indices,
        bounds=bounds,
    )
    I_eq = multi_variant_equilibrium_compartments(
        I_variant, beta, gamma, sigma, V, G, T, S, b
    )
    x_eq = map_I_to_x(I_eq)

    return I_variant, I_eq, x_eq


def eigenvalues_linearized_system(
    x_eq: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns the eigenvalues, its maximal real part and the eigenvectors of the
    linearized system at the equilibrium x_eq.

    Parameters
    ----------
    x_eq : array-like
        State at equilibrium.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    b : float
        Natural death and birth rate.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues of the linearized system at the equilibrium.
    max_real_part : float
        Maximal real part of the eigenvalues of the linearized system.
    eigvects : np.ndarray
        Eigenvectors of the linearized system at the equilibrium.
    """
    n_variants = number_of_variants(x_eq)
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

    return eigvals, max_real_part, eigvects
