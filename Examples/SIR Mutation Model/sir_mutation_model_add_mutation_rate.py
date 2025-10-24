# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from numerical_simulation_of_shsvd import solve_shsvd
import numpy as np
import matplotlib.pyplot as plt
from sir_mutation_model import sir_model, transition_function

"""
Computes the expected behavior for a multi-variant SIR model without
cross-immunity and counting multiply infected multiple times for the mutation
rate.
"""


def total_infectious(x):
    """
    Computes the total fraction of infectious individuals for a state x.

    Computes the total fraction of infectious individuals for a state x with
    counting multiply infected multiple times.
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
            i_ges += x[3 * j + 1]
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


x0 = [0.999, 0.001, 0]
t_span = (0, 500)
gamma = 0.1
beta = 0.15
mutation_rate = 0.2
sir_mutation_stats = solve_shsvd(
    sir_model,
    mutation_probability,
    transition_function,
    x0,
    t_span,
    mutation_rate,
    norm_order=[1, 2],
    collect_jump_times=True,
    flow_args=(beta, gamma),
    jump_rate_args=(mutation_rate,),
    iterations=100,
    custom_evaluation_functions=[total_infectious],
)
fig = sir_mutation_stats.plot_cefs()

plt.figure(fig)
plt.grid()
plt.title("Plot of overall infectious fraction")
plt.xlabel("Overall infectious fraction")
plt.savefig("Example.png")
plt.show()
