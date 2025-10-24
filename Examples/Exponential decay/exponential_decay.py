# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:31:06 2025

@author: schulte
"""

from numerical_simulation_of_shsvd import solve_shsvd, solve_shsvd_single_run
import matplotlib.pyplot as plt

"""
This computes the mean behavior and a sample path of a scaled exponential decay
with constant jump probability. This is a counterexample for exponential
stability in each dimension but not in SHS sense.
"""


def exponential_decay(t, x):
    """
    Computes derivatives of a scaled exponential decay.

    Computes derivatives of an exponential decay scaled by 1 over n in n-th
    coordinate.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    dx : list
        Derivatives of the state.
    """
    n = len(x)

    dx = []
    for i in range(n):
        dx.append(-x[i] / (i + 1))

    return dx


def jump_rate_function(t, x):
    """
    Returns a constant jump rate function of 1.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State (not used).

    Returns
    -------
    int
        1.

    """
    return 1


def transition_function(t, x):
    """
    Computes the state after a jump.

    Shifts the previously last entry of x to a new coordinate and sets the
    previously last (not penultimate) coordinate of x to 0.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system before the jump.

    Returns
    -------
    x : list
        State of the system after the jump.
    """
    x = list(x)
    n = len(x)

    x.append(x[n - 1])
    x[n - 1] = 0

    return x


# Initial values and parameters
initial_value = [1]
t_span = (0, 5)
jump_rate_bound = 1

# Compute and plot mean behavior of the system
sol_stats = solve_shsvd(
    exponential_decay,
    jump_rate_function,
    transition_function,
    initial_value,
    t_span,
    jump_rate_bound,
    stepsize=0.01,
    norm_order=1,
    tol=1e-4,
    min_iterations=50,
    verbose=True,
    convergence_estimate=("norm", 1),
    convergence_estimate_type="max",
)
fig = sol_stats.plot_norm(
    order=1, title="Plot of mean distance to A", ylabel="Distance to A"
)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("Exponential_decay_norm.png")
plt.show()

# Compute a sample path of the system
func = solve_shsvd_single_run(
    exponential_decay,
    jump_rate_function,
    transition_function,
    initial_value,
    t_span,
    jump_rate_bound,
)
fig = func.plot_coordinates(title="Plot of the coordinates of a sample path")
plt.figure(fig)
plt.tight_layout()
plt.savefig("Exponential_decay_sample_solution.png")
