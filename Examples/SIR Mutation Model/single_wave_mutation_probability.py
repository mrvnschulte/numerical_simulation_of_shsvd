# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:45 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd

"""
Computes the number of expected mutations per variant in a multi-variant
mutation model without cross-immunity and a hypothesized 'endemic equilibrium'
"""


def final_recovered_equation(x, r0, s0):
    """
    Defines the equation for the final size of an SIR model.

    Parameters
    ----------
    x : float
        Final size to be determined.
    r0 : float
        Basic reproduction number.
    s0 : float
        Initially susceptible fraction.

    Returns
    -------
    float
        Difference between the two sides of the equation.
    """
    return np.log(s0 / (1 - x)) - r0 * x


def final_recovered(r0, s0):
    """
    Solves the final size equation for given parameters.

    Parameters
    ----------
    r0 : float
        Basic reproduction number.
    s0 : float
        Initially susceptible fraction.

    Returns
    -------
    float
        Solution to the final size equation.
    """
    return brentq(final_recovered_equation, 0, 0.99, args=(r0, s0))


def stable_i(gamma, m, r0, s0):
    """
    Computes the hypothesized endemic equilibrium size.

    Parameters
    ----------
    gamma : float
        Recovery rate.
    m : float
        Mutation rate.
    r0 : float
        Basic reproduction number.
    s0 : float
        Initially susceptible fraction.

    Returns
    -------
    float
        Hypothesized endemic equilibrium size.
    """
    return 1 - gamma / (m * final_recovered(r0, s0))


if __name__ == "__main__":
    # Set parameters
    betas = 0.01 * np.arange(11, 26)
    gamma = 0.1
    mutation_rates = [
        0.01,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    x0 = [0.999, 0.001, 0]
    dpi = 500

    # Initialize result matrices
    empv = np.zeros((len(betas), len(mutation_rates)))
    i_star = np.zeros((len(betas), len(mutation_rates)))
    critical_m = np.zeros(len(betas))

    # Compute results
    for i, beta in enumerate(betas):
        r_final = final_recovered(beta / gamma, x0[0])
        critical_m[i] = gamma / r_final

        for j, mutation_rate in enumerate(mutation_rates):
            expected_mutations_per_wave = mutation_rate / gamma * r_final
            empv[i, j] = expected_mutations_per_wave
            i_star[i, j] = stable_i(gamma, mutation_rate, beta / gamma, x0[0])

    # Set color map
    ticks = [0, 0.5, 1, 1.5, 2, 5]
    lower_section = [0, 1]
    middle_section = [1, 2]
    upper_section = [2, 5.5]

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

    # Plot expected number of mutations per variant
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(
        betas, mutation_rates, empv.T, shading="auto", cmap=cmap, norm=norm
    )
    plt.colorbar(
        c, label="EMPV", ticks=ticks, boundaries=boundaries, format="%.1f"
    )
    plt.ylabel("Mutation Rate")
    plt.xlabel("Beta")
    plt.title("EMPV for Different Betas and Mutation Rates")
    plt.yticks(mutation_rates)
    plt.xticks(betas)
    plt.plot(betas, critical_m, "red")
    plt.tight_layout()
    plt.savefig("EMPV_Meshplot.jpg", dpi=dpi)
    plt.show()

    plt.figure(figsize=(8, 6))
    c = plt.contourf(
        betas, mutation_rates, empv.T, levels=levels, cmap=cmap, norm=norm
    )
    plt.colorbar(
        c, label="EMPV", ticks=ticks, boundaries=boundaries, format="%.1f"
    )
    plt.ylabel("Mutation Rate")
    plt.xlabel("Beta")
    plt.title("EMPV for Different Betas and Mutation Rates")
    plt.yticks(mutation_rates)
    plt.xticks(betas)
    plt.plot(betas, critical_m, "red")
    plt.tight_layout()
    plt.savefig("EMPV_Contourplot.jpg", dpi=dpi)
    plt.show()

    # Save data
    df_empv = pd.DataFrame(empv, index=betas, columns=mutation_rates)
    df_empv.to_csv("EMPV.csv")

    # Set color map
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    lower_section = [0, 0.5]
    middle_section = [0.5, 0.8]
    upper_section = [0.8, 1]

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

    # Plot hypothesized endemic equilibrium
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(
        betas, mutation_rates, i_star.T, shading="auto", cmap=cmap, norm=norm
    )
    plt.colorbar(
        c, label=r"$i^*$", ticks=ticks, boundaries=boundaries, format="%.1f"
    )
    plt.ylabel("Mutation Rate")
    plt.xlabel("Beta")
    plt.title(r"$i^*$ for Different Betas and Mutation Rates")
    plt.yticks(mutation_rates)
    plt.xticks(betas)
    plt.plot(betas, critical_m, "red")
    plt.tight_layout()
    plt.savefig("I_star_Meshplot.jpg", dpi=dpi)
    plt.show()

    plt.figure(figsize=(8, 6))
    c = plt.contourf(
        betas, mutation_rates, i_star.T, levels=levels, cmap=cmap, norm=norm
    )
    plt.colorbar(
        c, label=r"$i^*$", ticks=ticks, boundaries=boundaries, format="%.1f"
    )
    plt.ylabel("Mutation Rate")
    plt.xlabel("Beta")
    plt.title(r"$i^*$ for Different Betas and Mutation Rates")
    plt.yticks(mutation_rates)
    plt.xticks(betas)
    plt.plot(betas, critical_m, "red")
    plt.tight_layout()
    plt.savefig("I_star_Contourplot.jpg", dpi=dpi)
    plt.show()

    # Save data
    df_i_star = pd.DataFrame(i_star, index=betas, columns=mutation_rates)
    df_i_star.to_csv("I_star.csv")
