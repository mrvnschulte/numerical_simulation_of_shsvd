# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:55:49 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Computes the final number of infections for different mutation rates and cross-
immunity factors using the multi-variant model with one possible mutation
analytically using the pre-computed values for final infections with a fixed
delay
"""


def sir_model(t, x, beta, gamma):
    """
    Computes the derivatives of a state of a standard SIR model

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    dx : list
        List of the derivatives of the state variables.

    """
    s, i, r = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    dx = [ds, di, dr]

    return dx


# Initial values
t0 = 0
tf = 500
x0 = [0.999, 0.001, 0]
beta = 0.15
gamma = 0.1
m = 1
t_span = (t0, tf)
t_eval = np.linspace(t0, tf, tf + 1)

# File names of final infections with fixed delay
file1 = "Data/final_infections_1_delayed_mutation.csv"
file2 = "Data/final_infections_2_delayed_mutation.csv"

# Check if files exist
if os.path.exists(file1) and os.path.exists(file2):
    # Read csv's
    final_infections_1 = np.loadtxt(file1, delimiter=",", skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=",", skiprows=1)

    # Extract sigmas (stored in first column)
    cross_immunities = final_infections_1[:, 0]
    final_infections_1 = final_infections_1[:, 1:]
    final_infections_2 = final_infections_2[:, 1:]

    # Extract delays which are the headers
    header = np.loadtxt(file1, delimiter=",", max_rows=1, dtype=str)
    delays = header[1:].astype(float)

# Compute number of final infections of both variants added
final_infections = final_infections_1 + final_infections_2

# Initialise mutation rates and result matrix
mutation_rates = np.linspace(0, 1, 21)
expected_final_infections = np.zeros(
    (len(cross_immunities), len(mutation_rates))
)
expected_final_infections_with_mutation = np.zeros(
    (len(cross_immunities), len(mutation_rates))
)

for j, m in enumerate(mutation_rates):
    # Solve standard SIR model (dynamics of a single variant)
    res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    i = res.y[1]
    r = res.y[2]

    # Compute the probability density function of the mutation time given that
    # a muation happens
    probability_of_no_mutation = np.exp(-m * r[-1] / gamma)
    if probability_of_no_mutation < 1:
        mutation_time_pdf = (m * i * np.exp(-m * r / gamma)) / (
            1 - probability_of_no_mutation
        )
    else:
        mutation_time_pdf = 0

    for i, sigma in enumerate(cross_immunities):
        # Extract the number of final infections for a given sigma and
        # interpolate between them
        final_infections_sigma = final_infections[i]
        interpolated_values = np.interp(t_eval, delays, final_infections_sigma)

        # Compute number of expected final infections given a mutation happens
        # by integration over all delay times with probability density
        # function. The unconditioned expectation is weighted by probability of
        # a mutation happening
        expected_final_infections_with_mutation[i, j] = np.sum(
            mutation_time_pdf * interpolated_values
        )
        expected_final_infections[i, j] = (
            expected_final_infections_with_mutation[i, j]
            * (1 - probability_of_no_mutation)
            + r[-1] * probability_of_no_mutation
        )

# Plot expected number of final infections as color mesh and contour plot
c = plt.pcolormesh(
    mutation_rates[1:],
    cross_immunities,
    expected_final_infections_with_mutation[:, 1:],
)
plt.colorbar(c, label="Final number of infections")
plt.title(
    "Final number of infections given a mutation\n"
    r"($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()

c = plt.contourf(
    mutation_rates[1:],
    cross_immunities,
    expected_final_infections_with_mutation[:, 1:],
)
plt.colorbar(c, label="Final number of infections")
plt.title(
    "Final number of infections given a mutation\n"
    r"($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()

c = plt.pcolormesh(mutation_rates, cross_immunities, expected_final_infections)
plt.colorbar(c, label="Final number of infections")
plt.title(
    r"Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()

c = plt.contourf(mutation_rates, cross_immunities, expected_final_infections)
plt.colorbar(c, label="Final number of infections")
plt.title(
    r"Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()
