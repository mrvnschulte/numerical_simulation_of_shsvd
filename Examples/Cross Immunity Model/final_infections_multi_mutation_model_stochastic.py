# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:19:53 2025

@author: schulte
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numerical_simulation_of_shsvd import solve_shsvd
from cross_immunity_model import (
    multi_variant_model_with_matrices,
    new_variant_transition_function_stochastic,
    mutation_probability_function_multiple_mutations,
    extinction_function,
    extinction_transition_function,
    total_infections_new,
)

file_name = "expected_infections_multi_mutation.npy"
dpi = 500


t0 = 0
tf = 1000
t_eval = np.arange(tf + 1)
x0 = np.array([0.999, 0, 0.001, 0.001])
beta = [0.15]
gamma = [0.1]
sigma = [[1]]

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
variants = 1

alphas = np.linspace(2, 10, 17)
beta_dist = 2
k = 0.05
mutation_rates = np.linspace(0, 0.1, 11)
extinction_threshold = 0.0001

expected_cross_immunity = np.array(alphas) / (np.array(alphas) + 2)

if os.path.exists:
    expected_infections = np.load(file_name)

else:
    expected_infections = np.zeros((len(mutation_rates), len(alphas)))

    for i, mutation_rate in enumerate(mutation_rates):
        for j, alpha in enumerate(alphas):
            stats = solve_shsvd(
                multi_variant_model_with_matrices,
                mutation_probability_function_multiple_mutations,
                new_variant_transition_function_stochastic,
                x0,
                (t0, tf),
                t_eval=t_eval,
                jump_set_function=extinction_function,
                jump_set_transition_function=extinction_transition_function,
                flow_args=(beta, gamma, sigma, V, G, T, S, variants),
                transition_args=(alpha, beta_dist, k),
                jump_rate_args=(mutation_rate,),
                jump_set_args=(extinction_threshold, V, variants),
                custom_evaluation_functions=[total_infections_new],
                iterations=500,
                # min_iterations=50,
                # convergence_estimate=("cef", 0),
                # convergence_estimate_type=[("time", tf)],
                # tol=1e-4,
            )
            # stats.plot_cefs()
            expected_infections[i, j] = stats.evaluate_cefs(tf)[0]
            print(f"Finished computation for m={mutation_rate}, alpha={alpha}")

    np.save(file_name, expected_infections)

c = plt.pcolormesh(
    mutation_rates, expected_cross_immunity, expected_infections.T
)
cb = plt.colorbar(c, label="Expected number of infections")
cb.set_label(label="Expected number of infections", fontsize=12)
cb.ax.tick_params(labelsize=10)
# plt.title(r"Expected number of infections after time t=1000")
plt.ylabel(r"Expected cross-immunity $E[\sigma]$", fontsize=12)
plt.xlabel(r"Mutation rate $m$", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("Plots/expected_infections_multi_mutation_mesh.jpg", dpi=dpi)
plt.show()

c = plt.contourf(
    mutation_rates, expected_cross_immunity, expected_infections.T
)
plt.colorbar(c, label="Expected number of infections")
plt.title(r"Expected number of infections after time t=1000")
plt.ylabel(r"Expected cross-immunity $E[\sigma]$")
plt.xlabel(r"Mutation rate $m$")
plt.savefig("Plots/expected_infections_multi_mutation_contour.jpg", dpi=dpi)
plt.show()
