# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from numerical_simulation_of_shsvd import solve_shsvd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd
from single_wave_mutation_probability import final_recovered
import os


def sir_model(t, x, beta=0.15, gamma=0.1):
    """
    Computes the derivatives for n separate SIR models.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    beta : float, optional
        Transmission rate. The default is 0.15.
    gamma : float, optional
        Recovery rate. The default is 0.1.

    Returns
    -------
    dx : list
        Derivatives of the state.
    """
    n = int(len(x) / 3)

    dx = []
    for j in range(n):
        s, i, r = x[3 * j : 3 * j + 3]
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        dx.append(ds)
        dx.append(di)
        dx.append(dr)

    return dx


def sir_model_with_seasonality(t, x, beta=lambda x: 0.15, gamma=0.1):
    """
    Computes the derivatives for n separate SIR models with seasonality.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    beta : float, optional
        Transmission rate. The default is 0.15.
    gamma : float, optional
        Recovery rate. The default is 0.1.

    Returns
    -------
    dx : list
        Derivatives of the state.
    """
    n = int(len(x) / 3)

    dx = []
    for j in range(n):
        s, i, r = x[3 * j : 3 * j + 3]
        ds = -beta(t) * s * i
        di = beta(t) * s * i - gamma * i
        dr = gamma * i
        dx.append(ds)
        dx.append(di)
        dx.append(dr)

    return dx


def total_infectious(x):
    """
    Computes the total fraction of infectious individuals for a state x.

    Computes the total fraction of infectious individuals for a state x
    without counting multiply infected multiple times.
    Parameters
    ----------
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    i_ges : float
        Fraction of infectious individuals.
    """
    if isinstance(x[0], (list, np.ndarray)):
        i_ges = [total_infectious(y) for y in x]
        return i_ges

    else:
        n = int(len(x) / 3)
        i_ges = 0
        for j in range(n):
            i_ges += (1 - i_ges) * x[3 * j + 1]
        return i_ges


def total_infectious_added(x):
    """
    Computes the total fraction of infectious individuals for a state x.

    Computes the total fraction of infectious individuals for a state x
    with counting multiply infected multiple times.
    Parameters
    ----------
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    i_ges : float
        Fraction of infectious individuals.
    """
    if isinstance(x[0], (list, np.ndarray)):
        i_ges = [total_infectious(y) for y in x]
        return i_ges

    else:
        i_ges = sum(x[1::3])
        return i_ges


def mutation_probability(t, x, m=0.5):
    """
    Computes the mutation probability density for a state x.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    m : float, optional
        Mutation rate. The default is 0.5.

    Returns
    -------
    mutation_probability : float
        Mutation probability density.
    """
    i_ges = total_infectious(x)
    mutation_probability = m * i_ges

    return mutation_probability


def mutation_probability_added(t, x, m=0.5):
    """
    Computes the mutation probability density for a state x.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    m : float, optional
        Mutation rate. The default is 0.5.

    Returns
    -------
    mutation_probability : float
        Mutation probability density.
    """
    i_ges = total_infectious_added(x)
    mutation_probability = m * i_ges

    return mutation_probability


def transition_function(t, x, i_start=0.001):
    """
    Returns the state after a mutation.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    i_start : float, optional
        Initial infectious fraction with new variant. The default is 0.001.

    Returns
    -------
    x : list
        State of the system after the mutation.
    """
    x = list(x)
    s = 1 - i_start
    i = i_start
    r = 0
    x.append(s)
    x.append(i)
    x.append(r)

    return x


def overall_mutation_probability(values, m=0.5, stepsize=1):
    """
    Computes the cumulative mutation probability.

    Parameters
    ----------
    values : list, or np.ndarray
        State values.
    m : float, optional
        Mutation rate. The default is 0.5.
    stepsize : float, or int, optional
        Stepsize for which values have been computed. The default is 1.

    Returns
    -------
    overall_mutation_probability : np.ndarray
        Cumulative mutation probabilities.
    """
    mutation_probabilities = [
        mutation_probability(0, value, m=m) for value in values
    ]
    overall_mutation_probability = stepsize * np.cumsum(mutation_probabilities)

    return overall_mutation_probability


def active_variants(x):
    """
    Computes the number of 'active' variants, having infected more than .001.

    Parameters
    ----------
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    variants : int
        Number of active variants.
    """
    if isinstance(x[0], (list, np.ndarray)):
        variants = [active_variants(y) for y in x]
        return variants

    else:
        n = int(len(x) / 3)
        variants = 0
        for j in range(n):
            if x[3 * j + 1] > 0.001:
                variants += 1
        return variants


def beta_seasonal(t, beta0=0.15, ampl=0.5, phase_shift=0):
    """
    Returns the seasonal transmission rate.

    Parameters
    ----------
    t : float, list, or np.ndarray
        Time.
    beta0 : float, optional
        Base transmission rate. The default is 0.15.
    ampl : float, optional
        Seasonality amplitude. The default is 0.5.
    phase_shift : float, optional
        Seasonality phase shift. The default is 0.

    Returns
    -------
    beta : float, or np.ndarray
        Seasonal transmission rate.
    """
    beta = beta0 * (1 + ampl * np.sin(2 * np.pi * t / 100) + phase_shift)

    return beta


def extinction_function(t, x, extinction_threshold=0.0009):
    """
    Computes whether the first variant goes extinct.

    Computes the difference between the infectious with first variant and
    extinction threshold.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    extinction_threshold : float, optional
        Extinction threshold under which variant goes extinct. The default is
        0.0009.

    Returns
    -------
    float
        Difference between infectious fraction of first variant and extinction
        threshold.
    """
    i = x[1]

    return i - extinction_threshold


def extinction_transition(t, x):
    """
    Returns the state after extinction of the first variant.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    list
        State after the extinction.
    """
    if len(x) > 3:
        return x[3:]
    else:
        x[1] = 0
        return x


def infectious_with_first_variant(x):
    """
    Returns the infectious fraction with the first variant.

    Parameters
    ----------
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    i_1 : float, or list
        Infectious fraction with the first variant.

    """
    if isinstance(x[0], (list, np.ndarray)):
        i_1 = [infectious_with_first_variant(y) for y in x]
        return i_1
    else:
        i_1 = x[1]
        return i_1


if __name__ == "__main__":
    # Set parameters
    x0 = [0.999, 0.001, 0]
    tf = 1000
    t_span = (0, tf)
    betas = 0.01 * np.arange(11, 26)
    r0s = betas * 10
    r0s = np.round(r0s, 1)
    gamma = 0.1
    mutation_rates = [
        0.01,
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.5,
        0.6,
    ]
    dpi = 500
    tol = 1e-2
    file_name = "Mean_final_i.csv"

    # Read old data and initialise matrix for new one
    if os.path.exists(file_name):
        df_final_i_old = pd.read_csv("Mean_final_i.csv", index_col=0)
    else:
        df_final_i_old = pd.DataFrame()
    df_final_i_old.columns = df_final_i_old.columns.astype(float)
    final_i_values = np.zeros((len(betas), len(mutation_rates)))

    # Compute results by either using old data or computing new one, if no old is
    # available
    for i, r0 in enumerate(r0s):
        for j, mutation_rate in enumerate(mutation_rates):
            if (
                r0 in df_final_i_old.index
                and mutation_rate in df_final_i_old.columns
            ):
                final_i_values[i, j] = df_final_i_old.loc[r0, mutation_rate]
            else:
                sir_mutation_stats = solve_shsvd(
                    sir_model,
                    mutation_probability,
                    transition_function,
                    x0,
                    t_span,
                    mutation_rate,
                    norm_order=[1, 2],
                    collect_jump_times=True,
                    flow_args=(r0 * gamma, gamma),
                    jump_rate_args=(mutation_rate,),
                    tol=tol,
                    min_iterations=50,
                    custom_evaluation_functions=[total_infectious],
                    convergence_estimate=("cef", 0),
                    convergence_estimate_type=(("time", tf),),
                )
                # sir_mutation_stats.plot_cefs()
                final_i_values[i, j] = sir_mutation_stats.cef_means[0][-1]
            print(f"Finished computation for r0={r0} and m={mutation_rate}")

    # Set color map
    ticks = [0, 0.0005, 0.001, 0.05, 0.1, 1]
    lower_section = [0, 0.001]
    middle_section = [0.001, 0.1]
    upper_section = [0.1, 1]

    cmap = plt.colormaps["PiYG"]
    boundaries = (
        list(np.linspace(lower_section[0], lower_section[1], 22))
        + list(np.linspace(middle_section[0], middle_section[1], 16))
        + list(np.linspace(upper_section[0], upper_section[1], 6))
    )
    levels = (
        list(np.linspace(lower_section[0], lower_section[1], 22))
        + list(np.linspace(middle_section[0] * 1.01, middle_section[1], 16))
        + list(np.linspace(upper_section[0] * 1.01, upper_section[1], 6))
    )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Compute critical m for which empv=1
    critical_m = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        r_final = final_recovered(beta / gamma, x0[0])
        critical_m[i] = gamma / r_final

    # Plot results as meshplot and contour plot
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(
        betas,
        mutation_rates,
        final_i_values.T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(
        c,
        label="Mean Final I",
        ticks=ticks,
        boundaries=boundaries,
        format="%.4f",
    )
    plt.ylabel("Mutation Rate")
    plt.xlabel("Beta")
    plt.title("Mean Final I for Different Betas and Mutation Rates")
    plt.yticks(mutation_rates)
    plt.xticks(betas)
    plt.plot(betas, critical_m, "red")
    plt.tight_layout()
    plt.savefig("Mean_final_i_meshplot.jpg", dpi=dpi)
    plt.show()

    betas = betas * 10

    plt.figure()
    c = plt.contourf(
        betas,
        mutation_rates,
        final_i_values.T,
        levels=levels,
        cmap=cmap,
        norm=norm,
    )
    cb = plt.colorbar(
        c,
        label=r"Mean $I_{ges}(1000)$",
        ticks=ticks,
        boundaries=boundaries,
        format="%.4f",
    )
    cb.set_label(label=r"Mean $I_{ges}(1000)$", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    plt.ylabel(r"Mutation rate $m$", fontsize=12)
    plt.xlabel(r"Basic reproduction number $R_0$", fontsize=12)
    # plt.title(
    #     r"Mean $I_{ges}(1000)$ for different $R_0$ and mutation rates $m$"
    # )
    plt.yticks(fontsize=10)  # mutation_rates)
    plt.xticks(fontsize=10)  # betas)
    plt.plot(betas, critical_m, "red", label="$M_0=1$")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("Mean_final_i_countourf.jpg", dpi=dpi)
    plt.show()

    # Save results
    df_final_i = pd.DataFrame(
        final_i_values, index=r0s, columns=mutation_rates
    )
    df_final_i.to_csv("Mean_final_i.csv")

    # Set parameters for example plot with extinction
    t_span = (0, 2000)
    beta = 0.15
    mutation_rate = 0.2

    # Solve example system
    sir_mutation_stats = solve_shsvd(
        sir_model,
        mutation_probability_added,
        transition_function,
        x0,
        t_span,
        mutation_rate * 0.1,  # Speeds up the computation
        norm_order=[],
        collect_dimension=False,
        collect_jump_times=True,
        check_jump_rate=True,
        flow_args=(beta, gamma),
        jump_rate_args=(mutation_rate,),
        iterations=500,
        custom_evaluation_functions=[
            total_infectious,
            # infectious_with_first_variant,
        ],
        # jump_set_function=extinction_function,
        # jump_set_transition_function=extinction_transition,
    )

    # Plot example system
    fig = sir_mutation_stats.plot_cefs()
    plt.figure(fig)
    plt.grid()
    plt.title("Plot of overall infectious fraction")
    plt.ylabel("Overall infectious fraction")
    plt.savefig("Example.jpg", dpi=dpi)
    plt.show()

    sir_mutation_stats.plot_jump_times(
        [1, 2], density=True, separate_plots=True
    )
    sir_mutation_stats.plot_number_of_jumps()
    """
    # Set parameters for example plot with seasonality and extinction
    t_span = (0, 500)
    beta = 0.15
    mutation_rate = 0.2

    # Solve example system
    sir_mutation_stats = solve_shsvd(
        sir_model_with_seasonality,
        mutation_probability,
        transition_function,
        x0,
        t_span,
        mutation_rate,
        norm_order=[1, 2],
        collect_jump_times=True,
        flow_args=(beta_seasonal, gamma),
        jump_rate_args=(mutation_rate,),
        iterations=50,
        custom_evaluation_functions=[
            total_infectious,
            infectious_with_first_variant,
        ],
        jump_set_function=extinction_function,
        jump_set_transition_function=extinction_transition,
    )

    # Plot example system
    fig = sir_mutation_stats.plot_cefs()
    plt.figure(fig)
    plt.grid()
    plt.title("Plot of overall infectious fraction")
    plt.xlabel("Overall infectious fraction")
    plt.savefig("Example_with_seasonality.jpg", dpi=dpi)
    plt.show()

    sir_mutation_stats.plot_jump_times(
        [1, 2], density=True, separate_plots=True
    )
    sir_mutation_stats.plot_number_of_jumps()
    """
