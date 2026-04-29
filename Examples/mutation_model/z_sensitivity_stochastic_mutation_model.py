# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:19:53 2025

@author: schulte
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numerical_simulation_of_shsvd import (
    solve_hsvd,
    solve_shsvd,
    solve_shsvd_single_run,
)
from cross_immunity_model import (
    multi_variant_model_with_matrices,
    new_variant_transition_function_stochastic,
    mutation_probability_function_multiple_mutations,
    extinction_function,
    extinction_transition_function,
    total_infections_new,
    total_infectious,
)
import time

start = time.time()


file_name = "expected_infections_multi_mutation.npy"
dpi = 500
iterations = 100

t0 = 0
tf = 1000
t_eval = np.arange(tf + 1)
x0 = np.array([0.999, 0, 0.001, 0.001])
betas = [0.15, 0.2, 0.25]
gamma = [0.09]
sigma = [[1]]

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
b = 0.01
variants = 1

alphas = [2, 4, 8]
beta_dist = 2
ks = [0.025, 0.05, 0.075]
mutation_rates = np.round(np.linspace(0, 0.8, 5), 1)
extinction_threshold = 0.0001

r0 = np.round(np.array(betas) / (gamma[0] + b), 1)
expected_cross_immunity = np.round(
    np.array(alphas) / (np.array(alphas) + 2), 2
)

mean_infectious = []
variance_infectious = []
alpha = np.round(np.mean(alphas))
beta = [np.round(np.mean(betas), 2)]
k = np.mean(ks)

for i, mutation_rate in enumerate(mutation_rates):

    stats = solve_shsvd(
        multi_variant_model_with_matrices,
        mutation_probability_function_multiple_mutations,
        new_variant_transition_function_stochastic,
        x0,
        (t0, tf),
        t_eval=t_eval,
        jump_set_function=extinction_function,
        jump_set_transition_function=extinction_transition_function,
        flow_args=(beta, gamma, sigma, V, G, T, S, variants, b),
        transition_args=(alpha, beta_dist, k),
        jump_rate_args=(mutation_rate,),
        jump_set_args=(extinction_threshold, V, variants),
        custom_evaluation_functions=[total_infectious],
        iterations=iterations,
    )
    # stats.plot_cefs()
    # plt.show()
    mean_infectious.append((stats.cef_means.T).flatten())
    variance_infectious.append((stats.cef_variances.T).flatten())
    print(f"Finished computation for m={mutation_rate}")

plt.plot(t_eval, np.array(mean_infectious).T, label=mutation_rates)
for i in range(len(mean_infectious)):
    plt.fill_between(
        t_eval,
        np.array(mean_infectious[i]) + np.array(variance_infectious[i]),
        np.array(mean_infectious[i]) - np.array(variance_infectious[i]),
        alpha=0.3,
    )
plt.legend(title="Mutation rate", loc="upper left")
plt.xlabel("Time [days]")
plt.ylabel("Infectious fraction")
plt.tight_layout()
plt.savefig("Plots/Sensitivity_mutation_rate.jpg", dpi=dpi)
plt.show()

mean_infectious = []
variance_infectious = []

mutation_rate = np.round(np.mean(mutation_rates), 2)
beta = [np.round(np.mean(betas), 2)]
k = np.mean(ks)

for i, alpha in enumerate(alphas):
    stats = solve_shsvd(
        multi_variant_model_with_matrices,
        mutation_probability_function_multiple_mutations,
        new_variant_transition_function_stochastic,
        x0,
        (t0, tf),
        t_eval=t_eval,
        jump_set_function=extinction_function,
        jump_set_transition_function=extinction_transition_function,
        flow_args=(beta, gamma, sigma, V, G, T, S, variants, b),
        transition_args=(alpha, beta_dist, k),
        jump_rate_args=(mutation_rate,),
        jump_set_args=(extinction_threshold, V, variants),
        custom_evaluation_functions=[total_infectious],
        iterations=iterations,
    )
    # stats.plot_cefs()
    # plt.show()
    mean_infectious.append((stats.cef_means.T).flatten())
    variance_infectious.append((stats.cef_variances.T).flatten())
    print(f"Finished computation for alpha={alpha}")

plt.plot(t_eval, np.array(mean_infectious).T, label=expected_cross_immunity)
for i in range(len(mean_infectious)):
    plt.fill_between(
        t_eval,
        np.array(mean_infectious[i]) + np.array(variance_infectious[i]),
        np.array(mean_infectious[i]) - np.array(variance_infectious[i]),
        alpha=0.3,
    )
plt.legend(title=r"Expected cross-immunity", loc="upper left")
plt.xlabel("Time [days]")
plt.ylabel("Infectious fraction")
plt.tight_layout()
plt.savefig("Plots/Sensitivity_cross_immunity.jpg", dpi=dpi)
plt.show()


mean_infectious = []
variance_infectious = []

mutation_rate = np.round(np.mean(mutation_rates), 2)
alpha = np.round(np.mean(alphas), 2)
k = np.mean(ks)

for i, beta in enumerate(betas):
    beta = [beta]
    stats = solve_shsvd(
        multi_variant_model_with_matrices,
        mutation_probability_function_multiple_mutations,
        new_variant_transition_function_stochastic,
        x0,
        (t0, tf),
        t_eval=t_eval,
        jump_set_function=extinction_function,
        jump_set_transition_function=extinction_transition_function,
        flow_args=(beta, gamma, sigma, V, G, T, S, variants, b),
        transition_args=(alpha, beta_dist, k),
        jump_rate_args=(mutation_rate,),
        jump_set_args=(extinction_threshold, V, variants),
        custom_evaluation_functions=[total_infectious],
        iterations=iterations,
    )
    # stats.plot_cefs()
    # plt.show()
    mean_infectious.append((stats.cef_means.T).flatten())
    variance_infectious.append((stats.cef_variances.T).flatten())
    print(f"Finished computation for beta={beta}")

plt.plot(t_eval, np.array(mean_infectious).T, label=r0)
for i in range(len(mean_infectious)):
    plt.fill_between(
        t_eval,
        np.array(mean_infectious[i]) + np.array(variance_infectious[i]),
        np.array(mean_infectious[i]) - np.array(variance_infectious[i]),
        alpha=0.3,
    )
plt.legend(title=r"Initial $R_0$", loc="upper left")
plt.xlabel("Time [days]")
plt.ylabel("Infectious fraction")
plt.tight_layout()
plt.savefig("Plots/Sensitivity_initial_r0.jpg", dpi=dpi)
plt.show()


mean_infectious = []
variance_infectious = []

mutation_rate = np.round(np.mean(mutation_rates), 2)
alpha = np.round(np.mean(alphas), 2)
beta = [np.round(np.mean(betas), 2)]

for i, k in enumerate(ks):
    stats = solve_shsvd(
        multi_variant_model_with_matrices,
        mutation_probability_function_multiple_mutations,
        new_variant_transition_function_stochastic,
        x0,
        (t0, tf),
        t_eval=t_eval,
        jump_set_function=extinction_function,
        jump_set_transition_function=extinction_transition_function,
        flow_args=(beta, gamma, sigma, V, G, T, S, variants, b),
        transition_args=(alpha, beta_dist, k),
        jump_rate_args=(mutation_rate,),
        jump_set_args=(extinction_threshold, V, variants),
        custom_evaluation_functions=[total_infectious],
        iterations=iterations,
    )
    # stats.plot_cefs()
    # plt.show()
    mean_infectious.append((stats.cef_means.T).flatten())
    variance_infectious.append((stats.cef_variances.T).flatten())
    print(f"Finished computation for k={k}")

plt.plot(
    t_eval,
    np.array(mean_infectious).T,
    label=[rf"{k}(1-$\sigma$)" for k in ks],
)
for i in range(len(mean_infectious)):
    plt.fill_between(
        t_eval,
        np.array(mean_infectious[i]) + np.array(variance_infectious[i]),
        np.array(mean_infectious[i]) - np.array(variance_infectious[i]),
        alpha=0.3,
    )
plt.legend(title=r"Variance of new $R_0$", loc="upper left")
plt.xlabel("Time [days]")
plt.ylabel("Infectious fraction")
plt.tight_layout()
plt.savefig("Plots/Sensitivity_variance_r0.jpg", dpi=dpi)
plt.show()

end = time.time()
elapsed = end - start
print(f"Gesamte Laufzeit: {elapsed:.6f} Sekunden")
