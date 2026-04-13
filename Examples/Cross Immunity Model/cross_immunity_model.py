# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:07:54 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os
from numerical_simulation_of_shsvd import (
    solve_hsvd,
    solve_shsvd_single_run,
    solve_shsvd,
)


def two_variant_model(t, x, beta1, beta2, sigma, gamma1=0.1, gamma2=0.1):
    """
    Computes the derivatives of a two-variant model with cross-immunity.

    This function computes the derivatives of a two-variant SIR-like model with
    cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : list, or np.ndarray
        The current state of the system.
    beta1 : float
        Transmission rate of variant 1.
    beta2 : float
        Transmission rate of variant 2.
    sigma : float
        Cross-immunity factor between the two variants.
    gamma1 : float, optional
        Recovery rate of variant 1. The default is .1.
    gamma2 : float, optional
        Recovery rate of variant 2. The default is .1.

    Returns
    -------
    dx : list
        The derivatives of the variables.
    """

    i1, i12, i2, i21, s, r1, r2, r = x

    i1_ges = i1 + i12
    i2_ges = i2 + i21

    ds = -beta1 * s * i1_ges - beta2 * s * i2_ges
    di1 = (1 - sigma) * beta1 * s * i1_ges - gamma1 * i1
    dr1 = gamma1 * i1 - beta2 * r1 * i2_ges
    di21 = sigma * beta2 * s * i2_ges + beta2 * r1 * i2_ges - gamma2 * i21
    di2 = (1 - sigma) * beta2 * s * i2_ges - gamma2 * i2
    dr2 = gamma2 * i2 - beta1 * r2 * i1_ges
    di12 = sigma * beta1 * s * i1_ges + beta1 * r2 * i1_ges - gamma1 * i12
    dr = gamma2 * i21 + gamma1 * i12

    dx = [di1, di12, di2, di21, ds, dr1, dr2, dr]

    return dx


def multi_variant_model(t, x, beta, gamma, sigma):
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : list, or np.ndarray
        The current state of the system.
    beta : list, or np.ndarray
        Vector of transmission rates of the variants.
    gamma : list, or np.ndarray
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """

    # Compute number of variants and number of protection/infection groups
    n = int(round(np.log(len(x)) / np.log(3)))
    size = 2**n

    # Transform state vector into corresponding matrix
    I = map_x_to_I(x)
    # dI = np.zeros_like(I)

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
    T = construct_transition_matrix(n, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T

    dI = (
        -(gamma_sum * I.T).T
        - beta_sum * I
        + gamma_matrix @ I
        + np.sum(S @ I @ W, axis=0)
    )

    dx = map_I_to_x(dI)

    return dx


def multi_variant_model_for_loops(t, x, beta, gamma, sigma):
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : list, or np.ndarray
        The current state of the system.
    beta : list, or np.ndarray
        Vector of transmission rates of the variants.
    gamma : list, or np.ndarray
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """

    # Compute number of variants and number of protection/infection groups
    n = int(round(np.log(len(x)) / np.log(3)))
    size = 2**n

    # Transform state vector into corresponding matrix
    I = map_x_to_I(x)
    dI = np.zeros_like(I)

    # V = variant_in_set_matrix(n)  # It could be beneficial to store these
    # # matrices in a class at their first evaluation instead of constructing it
    # # in every step

    # # Compute the infectious with each variant
    # I_variant = (V @ I @ np.ones((size, 1))).flatten()
    # # Compute the recoveries for each infection set
    # gamma_sum = (V.T @ gamma).flatten()
    # # Compute the transmission factor for each variant
    # transmission_vector = beta * I_variant
    # # TO-DO: Comment
    # beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    # gamma_matrix = compute_gamma_matrix(n, gamma)
    # T = construct_transition_matrix(n, sigma)
    # S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    # W = transmission_vector[:, np.newaxis, np.newaxis] * T

    # dI = (
    #     -(gamma_sum * I.T).T
    #     - beta_sum * I
    #     + gamma_matrix @ I
    #     + np.sum(S @ I @ W, axis=0)
    # )
    I_variant = np.zeros(n)
    for k in range(n):
        for i in range(size):
            if variant_in_set(k + 1, i):
                for j in range(size):
                    if is_subset(i, j):
                        I_variant[k] += I[i, j]

    sigma_trans = np.ones((n, size, size))
    for k in range(n):
        for i in range(size):
            if not variant_in_set(k + 1, i):
                for j in range(size):
                    if variant_in_set(k + 1, j) and is_subset(i, j):
                        for m in range(n):
                            if m != k:
                                if variant_in_set(
                                    m + 1, j
                                ) and not variant_in_set(m + 1, i):
                                    sigma_trans[k, i, j] *= sigma[k, m]
                                if not variant_in_set(m + 1, j):
                                    sigma_trans[k, i, j] *= 1 - sigma[k, m]

    for i in range(size):
        for j in range(size):
            if is_subset(i, j):
                for k in range(n):
                    if variant_in_set(k + 1, i):
                        dI[i, j] -= gamma[k] * I[i, j]
                        i_minus_k = i ^ (2**k)
                        for l in range(size):
                            if (
                                is_subset(i_minus_k, l)
                                and is_subset(l, j)
                                and not variant_in_set(k + 1, l)
                            ):
                                dI[i, j] += (
                                    sigma_trans[k, l, j]
                                    * beta[k]
                                    * I[i_minus_k, l]
                                    * I_variant[k]
                                )
                    if not variant_in_set(k + 1, j):
                        dI[i, j] -= beta[k] * I[i, j] * I_variant[k]
                    if not variant_in_set(k + 1, i) and variant_in_set(
                        k + 1, j
                    ):
                        i_plus_k = i ^ (2**k)
                        dI[i, j] += gamma[k] * I[i_plus_k, j]

    dx = map_I_to_x(dI)

    return dx


def multi_variant_model_with_matrices(
    t, x, beta, gamma, sigma, V, G, T, S, n, b=0, compute_dn=True
):
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : list, or np.ndarray
        The current state of the system.
    beta : list, or np.ndarray
        Vector of transmission rates of the variants.
    gamma : list, or np.ndarray
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    TO_DO: Ergänzen

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """

    size = 2**n

    if compute_dn:
        I = map_x_to_I(x[:-1])
    else:
        I = map_x_to_I(x)

    # Compute the infectious with each variant
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    # Compute the recoveries for each infection set
    gamma_sum = (V.T @ gamma).flatten()
    # Compute the transmission factor for each variant
    transmission_vector = beta * I_variant
    # TO-DO: Comment
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    W = transmission_vector[:, np.newaxis, np.newaxis] * T

    dI = (
        -(gamma_sum * I.T).T
        - beta_sum * I
        + G @ I
        + np.sum(S @ I @ W, axis=0)
        - b * I
    )
    dI[0, 0] += b

    dx = map_I_to_x(dI)

    if compute_dn:
        S_variant = (
            np.ones((size, 1)).T @ I @ (np.ones_like(V) - V).T
        ).flatten()
        dn = np.sum(beta * S_variant * I_variant)
        dx = list(dx)
        dx.append(dn)

    return dx


def multi_variant_model_with_demographics(
    t, x, beta, gamma, sigma, b, V=None, G=None, T=None, S=None, n=None
):
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : list, or np.ndarray
        The current state of the system.
    beta : list, or np.ndarray
        Vector of transmission rates of the variants.
    gamma : list, or np.ndarray
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """

    # Compute number of variants and number of protection/infection groups
    if n is None:
        n = int(round(np.log(len(x)) / np.log(3)))
    size = 2**n

    if V is None:
        V = variant_in_set_matrix(n)
    if G is None:
        G = compute_gamma_matrix(n, gamma)
    if T is None:
        T = construct_transition_matrix(n, sigma)
    if S is None:
        S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    # W = transmission_vector[:, np.newaxis, np.newaxis] * T

    # for j in range(size):
    #     for k in range(j + 1):
    #         if is_subset(k, j):
    #             dI[k, j] = (
    #                 -b * I[k, j]
    #                 - gamma_sum[k] * I[k, j]
    #                 - beta_sum[j] * I[k, j]
    #                 + (G @ I)[k, j]
    #                 + (np.sum(S @ I @ W, axis=0))[k, j]
    #             )
    # dI[0, 0] += b

    # dx = map_I_to_x(dI)

    dx = multi_variant_model_with_matrices(
        t, x, beta, gamma, sigma, V, G, T, S, n, compute_dn=False
    )
    dx = np.array(dx)
    dx -= b * x
    dx[0] += b

    return dx


def new_variant_transition_function(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    jump_rate_bound=0.1,
):

    # Attention: Works only for two variants after jump
    flow_args = new_flow_args(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        beta_new,
        gamma_new,
        sigma_new,
    )

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    n = A.shape[0]
    zero_matrix = np.zeros((n, n))

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)

    return x, flow_args


def new_variant_transition_function_with_demographics(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    old_b=0.01,
    jump_rate_bound=0.1,
):
    # Attention: Works only for two variants after jump
    flow_args = new_flow_args_with_demographics(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        old_b,
        beta_new,
        gamma_new,
        sigma_new,
    )

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    n = A.shape[0]
    zero_matrix = np.zeros((n, n))

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)

    return x, flow_args


def new_variant_transition_function_with_matrices(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    jump_rate_bound=0.1,
    mutant_variant=-1,
    extinction_threshold=0,
):
    n = x[-1].copy()
    x = x[:-1].copy()
    # Attention: Works only for two variants after jump
    flow_args = new_flow_args_with_matrices(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        V,
        G,
        T,
        S,
        variants,
        old_b,
        beta_new,
        gamma_new,
        sigma_new,
    )

    if mutant_variant > 0:
        idx_mutant_variant = 2**mutant_variant
    else:
        idx_mutant_variant = mutant_variant
    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[idx_mutant_variant] = max(
        min(0.001, x[mutant_variant]),
        extinction_threshold + 0.1 * extinction_threshold,
    )
    x[mutant_variant] = max(x[mutant_variant] - 0.001, 0)
    x = np.array(x)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    zero_matrix = np.zeros_like(A)

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)
    x = np.append(x, n)

    sigma_new = np.append(sigma_new, sigma_new[-1])

    transition_args = (beta_new, gamma_new, sigma_new)

    return x, flow_args, transition_args


def new_variant_transition_function_stochastic(
    t,
    x,
    alpha=10,
    beta=2,
    k=1,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.05,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):
    I = map_x_to_I(x[:-1])
    size = 2**variants
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    I_total = np.sum(I_variant)
    mutated_variant = np.random.choice(variants, p=I_variant / I_total)

    sigma_variant = np.random.beta(alpha, beta)
    sigma_new = sigma_variant * np.array(old_sigma[mutated_variant])
    gamma_new = old_gamma[0]
    beta_new = max(
        np.random.normal(old_beta[mutated_variant], k * (1 - sigma_variant)), 0
    )

    x, flow_args, _ = new_variant_transition_function_with_matrices(
        t,
        x,
        beta_new,
        gamma_new,
        sigma_new,
        old_beta,
        old_gamma,
        old_sigma,
        V,
        G,
        T,
        S,
        variants,
        old_b,
        jump_rate_bound,
        mutated_variant,
        extinction_threshold,
    )

    jump_set_args = (
        extinction_threshold,
        flow_args[3],
        variants + 1,
    )

    if variants == 5:
        x, flow_args, jump_set_args = (
            extinction_transition_function_least_transmissible(
                t,
                x,
                *flow_args,
                extinction_threshold,
            )
        )

    return x, flow_args, jump_set_args


def new_flow_args(
    t, x, old_beta, old_gamma, old_sigma, beta=0.18, gamma=0.1, sigma=[0.5]
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return beta_new, gamma_new, sigma_new


def new_flow_args_with_matrices(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    V,
    G,
    T,
    S,
    n,
    old_b=0,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    sigma = np.array(sigma)
    old_sigma = np.array(old_sigma)
    sigma = sigma.reshape(-1, 1)
    sigma_new = np.block([[old_sigma, sigma], [sigma.T, np.array([[1]])]])

    V_new = np.block(
        [
            [V, V],
            [np.zeros((1, V.shape[1])), np.ones((1, V.shape[1]))],
        ]
    )
    G_new = np.block([[G, gamma * np.eye(len(G))], [np.zeros_like(G), G]])
    T_new = [
        np.block(
            [
                [(1 - sigma[i]) * T[i], sigma[i] * T[i]],
                [np.zeros_like(T[i]), T[i]],
            ]
        )
        for i in range(len(T))
    ]
    T_n = construct_transition_matrix_n(sigma_new[n])
    T_new.append(T_n)
    T_new = np.array(T_new)
    n += 1
    S_new = np.array([shift_matrix(2**n, 2**m) for m in range(n)])

    return beta_new, gamma_new, sigma_new, V_new, G_new, T_new, S_new, n, old_b


def new_flow_args_with_demographics(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    old_b,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return beta_new, gamma_new, sigma_new, old_b


def multi_variant_model_with_total_infections(t, x, beta, gamma, sigma, n):

    size = 2**n

    I = map_x_to_I(x[: 3**n])

    dI = np.zeros_like(I)

    V = variant_in_set_matrix(
        n
    )  # It could be beneficial to store these matrices in a class at their first evaluation instead of constructing it in every step
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    gamma_sum = (V.T @ gamma).flatten()
    transmission_vector = beta * I_variant
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    gamma_matrix = compute_gamma_matrix(n, gamma)
    T = construct_transition_matrix(n, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T
    S_variant = (np.ones((size, 1)).T @ I @ (np.ones_like(V) - V).T).flatten()
    dn = beta * S_variant * I_variant

    for j in range(size):
        for k in range(j + 1):
            if is_subset(k, j):
                dI[k, j] = (
                    -gamma_sum[k] * I[k, j]
                    - beta_sum[j] * I[k, j]
                    + (gamma_matrix @ I)[k, j]
                    + (np.sum(S @ I @ W, axis=0))[k, j]
                )

    dx = map_I_to_x(dI)
    dx = list(dx)
    dx.extend(dn)

    return dx


def new_variant_transition_function_with_total_infections(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    variants=1,
    jump_rate_bound=0.1,
):
    # Attention: Works only for two variants after jump
    new_flow_args = new_flow_args_with_total_infections(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        variants,
        beta_new,
        gamma_new,
        sigma_new,
    )

    n = x[-variants:].copy()
    x = x[:-variants].copy()
    x = np.array(x)

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    size = A.shape[0]
    zero_matrix = np.zeros((size, size))

    # Kombiniere die Arrays
    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)
    n_new = np.append(n, infectious_with_new_variant[-1])

    x = np.append(x, n_new)

    return x, new_flow_args, 1e-10


def new_flow_args_with_total_infections(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    old_n,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return (beta_new, gamma_new, sigma_new, old_n + 1)


def total_infections(x):
    if isinstance(x[0], (list, np.ndarray)):
        infections = [total_infections(y) for y in x]
        return infections
    else:
        infections = x[-1]
        if len(x) > 5:
            infections += x[-2]
    return infections


def map_x_to_I(x, dtype=None):
    """
    Maps vector representation x to matrix representation I

    Parameters
    ----------
    x : list, or np.ndarray
        Vector representation of state.

    Returns
    -------
    I : np.ndarray
        Matrix representation of state.
    """

    n = int(round(np.log(len(x)) / np.log(3)))

    I = np.zeros((2**n, 2**n), dtype=dtype)

    counter = 0

    for j in range(2**n):
        for k in range(j + 1):
            if is_subset(k, j):
                I[k, j] = x[counter]
                counter += 1

    return I


def map_I_to_x(I, dtype=None):
    """
    Maps matrix representation I back to vector representation x

    Parameters
    ----------
    I : np.ndarray
        Matrix representation of state.

    Returns
    -------
    x : np.ndarray
        Vector representation of state.
    """

    n = int(round(np.log2(I.shape[0])))  # Bestimme n aus der Matrixgröße
    size = 2**n  # Größe der Matrix

    # Berechne die Länge des Vektors x
    length_x = 3**n  # Anzahl der nicht-null Einträge
    x = np.zeros(length_x, dtype=dtype)  # Initialisiere den Vektor x

    counter = 0  # Zähler für den Vektor x

    # Fülle den Vektor x basierend auf der Matrix I
    for j in range(size):
        for k in range(j + 1):
            if is_subset(k, j):  # Überprüfe, ob k eine Teilmenge von j ist
                x[counter] = I[k, j]
                counter += 1

    return x


def is_subset(j, k):
    """
    Checks whether j is a subset of k using binary representations.

    Parameters
    ----------
    j : int
        Binary (integer) representation of set j.
    k : int
        Binary (integer) representation of set k.

    Returns
    -------
    value : bool
        True, if j is a subset of k.
    """

    value = j & k == j

    return value


def variant_in_set(i, j):
    """
    Checks whether variant i is in set j.

    Parameters
    ----------
    i : int
        Index of variant i.
    j : int
        Binary (integer) representation of set j.

    Returns
    -------
    bool
        True, if variant i is in set j.

    """
    return (j & (1 << (i - 1))) != 0


def variant_in_set_matrix(n):

    V = np.zeros((n, 2**n), dtype=int)

    indices = np.arange(2**n)

    for i in range(n):
        V[i] = (indices & (1 << i)) != 0

    return V


def compute_gamma_matrix(n, gamma, dtype=None):
    # Determine n based on gamma?
    # Basisfall: A_1
    A_n = np.array([[0, gamma[0]], [0, 0]], dtype=dtype)

    for i in range(1, n):
        # Größe von A_n bestimmen
        size = A_n.shape[0]

        # Erzeuge die Einheitsmatrix I der passenden Größe
        I = np.eye(size)

        # Konstruiere A_{n+1}
        A_n = np.block([[A_n, gamma[i] * I], [np.zeros_like(A_n), A_n]])

    return A_n


def construct_transition_matrix(n, sigma, dtype=None):

    size = 2**n
    T = np.zeros((n, size, size), dtype=dtype)
    sigma = np.array(sigma)

    for k in range(n):
        for i in range(size):
            if not variant_in_set(k + 1, i):
                for j in range(size):
                    if is_subset(i, j):
                        T[k, i, j] = 1
                        for m in range(n):
                            if variant_in_set(m + 1, j - i):
                                T[k, i, j] *= sigma[k, m]
                            if not variant_in_set(m + 1, j):
                                T[k, i, j] *= 1 - sigma[k, m]

    return T


def construct_transition_matrix_n(sigma):

    n = len(sigma)
    size = 2**n
    T = np.zeros((size, size))
    sigma = np.array(sigma)

    for i in range(size):
        if not variant_in_set(n, i):
            for j in range(size):
                if is_subset(i, j):
                    T[i, j] = 1
                    for m in range(n):
                        if variant_in_set(m + 1, j - i):
                            T[i, j] *= sigma[m]
                        if not variant_in_set(m + 1, j):
                            T[i, j] *= 1 - sigma[m]

    return T


def shift_matrix(size, shift):

    return np.eye(size, k=-shift)


def mutation_probability_function(t, x, m=0.1):
    if len(x) < 5:
        mutation_prob = m * x[2]
    else:
        mutation_prob = 0
    return mutation_prob


def mutation_probability_function_multiple_mutations(t, x, mutation_rate=0.1):

    I = map_x_to_I(x[:-1])
    infectious = np.sum(I[1:])
    mutation_probability = mutation_rate * infectious

    return mutation_probability


def mutation_probability_function_multiple_mutations_no_dn(
    t, x, mutation_rate=0.1
):

    I = map_x_to_I(x)
    infectious = np.sum(I[1:])
    mutation_probability = mutation_rate * infectious

    return mutation_probability


def extinction_function(
    t, x, extinction_threshold=0.0005, V=np.array([[0, 1]]), variants=1
):

    if variants > 1:
        size = 2**variants
        I = map_x_to_I(x[:-1])
        I_variant = (V @ I @ np.ones((size, 1))).flatten()

        distance_to_threshold = np.min(I_variant) - extinction_threshold
    else:
        distance_to_threshold = 1

    return distance_to_threshold


def extinction_transition_function(
    t,
    x,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.005,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):

    size = 2**variants
    I = map_x_to_I(x[:-1])
    I_variant = (V @ I @ np.ones((size, 1))).flatten()

    extinct_variant = np.argmin(I_variant)

    beta_new = np.delete(old_beta, extinct_variant)
    gamma_new = np.delete(old_gamma, extinct_variant)
    sigma_new = np.delete(
        np.delete(old_sigma, extinct_variant, axis=0), extinct_variant, axis=1
    )
    half_size = size // 2
    V_new = V[: variants - 1, :half_size]
    sets_with_extinct_variant = V[extinct_variant] == 0
    G_new = G[np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)]
    T_new = np.delete(T, extinct_variant, axis=0)
    T_new = np.array(
        [
            np.array(T_i)[
                np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)
            ]
            + np.array(T_i)[
                np.ix_(sets_with_extinct_variant, ~sets_with_extinct_variant)
            ]
            for T_i in T_new
        ]
    )
    S_new = np.array(
        [shift_matrix(2 ** (variants - 1), 2**m) for m in range(variants - 1)]
    )
    flow_args = (
        beta_new,
        gamma_new,
        sigma_new,
        V_new,
        G_new,
        T_new,
        S_new,
        variants - 1,
        old_b,
    )

    I_new = I[sets_with_extinct_variant] + I[~sets_with_extinct_variant]
    I_new = (
        I_new[:, sets_with_extinct_variant]
        + I_new[:, ~sets_with_extinct_variant]
    )
    x_new = map_I_to_x(I_new)
    x_new = np.append(x_new, x[-1])

    jump_set_args = (extinction_threshold, V_new, variants - 1)

    return x_new, flow_args, jump_set_args


def extinction_transition_function_least_transmissible(
    t,
    x,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.005,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):

    size = 2**variants
    I = map_x_to_I(x[:-1])
    I_variant = (V @ I @ np.ones((size, 1))).flatten()

    r0 = np.array(old_beta) / np.array(old_gamma)
    extinct_variant = np.argmin(r0)

    beta_new = np.delete(old_beta, extinct_variant)
    gamma_new = np.delete(old_gamma, extinct_variant)
    sigma_new = np.delete(
        np.delete(old_sigma, extinct_variant, axis=0), extinct_variant, axis=1
    )
    half_size = size // 2
    V_new = V[: variants - 1, :half_size]
    sets_with_extinct_variant = V[extinct_variant] == 0
    G_new = G[np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)]
    T_new = np.delete(T, extinct_variant, axis=0)
    T_new = np.array(
        [
            np.array(T_i)[
                np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)
            ]
            + np.array(T_i)[
                np.ix_(sets_with_extinct_variant, ~sets_with_extinct_variant)
            ]
            for T_i in T_new
        ]
    )
    S_new = np.array(
        [shift_matrix(2 ** (variants - 1), 2**m) for m in range(variants - 1)]
    )
    flow_args = (
        beta_new,
        gamma_new,
        sigma_new,
        V_new,
        G_new,
        T_new,
        S_new,
        variants - 1,
        old_b,
    )

    I_new = I[sets_with_extinct_variant] + I[~sets_with_extinct_variant]
    I_new = (
        I_new[:, sets_with_extinct_variant]
        + I_new[:, ~sets_with_extinct_variant]
    )
    x_new = map_I_to_x(I_new)
    x_new = np.append(x_new, x[-1])

    jump_set_args = (extinction_threshold, V_new, variants - 1)

    return x_new, flow_args, jump_set_args


def total_infections_new(x):

    if isinstance(x[0], (list, np.ndarray)):
        infections = [total_infections_new(y) for y in x]
    else:
        infections = x[-1]

    return infections


def total_infectious(x):
    if isinstance(x[0], (list, np.ndarray)):
        infectious = [total_infectious(y) for y in x]
    else:
        I = map_x_to_I(x)
        infectious = np.sum(I[1:])
    return infectious


new_variant_transition_function_with_total_infections.returns_flow_args = True
new_variant_transition_function_with_total_infections.returns_jump_rate_bound = (
    True
)
new_variant_transition_function.returns_flow_args = True
new_variant_transition_function_with_demographics.returns_flow_args = True
new_variant_transition_function_with_matrices.returns_flow_args = True
new_variant_transition_function_stochastic.returns_flow_args = True

new_variant_transition_function_stochastic.returns_jump_set_args = True

new_variant_transition_function_with_matrices.returns_jump_time_transition_args = (
    True
)
extinction_transition_function.returns_flow_args = True
extinction_transition_function.returns_jump_set_args = True
