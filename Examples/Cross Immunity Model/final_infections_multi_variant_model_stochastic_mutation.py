# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:48:46 2025

@author: schulte
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from numerical_simulation_of_shsvd import solve_shsvd
from cross_immunity_model import (
    multi_variant_model_with_total_infections,
    new_variant_transition_function_with_total_infections,
    mutation_probability_function,
    total_infections,
)

"""
Computes the final number of infections for different mutation rates and cross-
immunity factors using the multi-variant model with one possible mutation
"""

t0 = 0
tf = 1000
x0 = np.array([0.999, 0, 0.001, 0.001])
beta = [0.15]
gamma = [0.1]
sigma = [[1]]
beta_new = 0.2
gamma_new = 0.1
t_eval = np.arange(tf + 1)
tol = 1e-3

file = f"Data/final_infections_stochastic_mutation_tol_{tol}.csv"

if os.path.exists(file):
    final_infections = np.loadtxt(file, delimiter=",", skiprows=1)

    # Extrahiere sigmas und r02s
    cross_immunities = final_infections[:, 0]  # Die erste Spalte ist sigmas
    final_infections = final_infections[
        :, 1:
    ]  # Der Rest sind die Werte für final_infections_1

    # sigmas extrahieren, indem wir die erste Zeile der CSV Datei lesen und die Kopfzeile ignorieren
    header = np.loadtxt(
        file, delimiter=",", max_rows=1, dtype=str
    )  # Lies die Kopfzeile als String
    mutation_rates = header[1:].astype(
        float
    )  # Wandle die Werte in float um, überspringe das erste Element

else:
    mutation_rates = np.linspace(0, 1, 21)
    cross_immunities = np.linspace(0, 1, 21)

    final_infections = np.zeros((len(cross_immunities), len(mutation_rates)))

    for i, s in enumerate(cross_immunities):
        for j, m in enumerate(mutation_rates):
            sigma_new = [s]

            stats = solve_shsvd(
                multi_variant_model_with_total_infections,
                mutation_probability_function,
                new_variant_transition_function_with_total_infections,
                x0,
                (t0, tf),
                jump_rate_bound=m,
                t_eval=t_eval,
                jump_time_transition_function=new_variant_transition_function_with_total_infections,
                flow_args=(beta, gamma, sigma, 1),
                jump_time_transition_args=(beta_new, gamma_new, sigma_new),
                transition_args=(beta_new, gamma_new, sigma_new),
                jump_rate_args=(m,),
                tol=tol,
                min_iterations=50,
                convergence_estimate=("cef", 0),
                convergence_estimate_type=(("time", tf),),
                custom_evaluation_functions=[total_infections],
            )

            final_infections[i, j] = stats.cef_means[0][-1]
            print(f'Finished computation for sigma={s} and m={m}')

    final_infections_with_sigma = np.column_stack(
        (cross_immunities, final_infections)
    )

    # Speichern der Matrizen als CSV-Dateien
    np.savetxt(
        file,
        final_infections_with_sigma,
        delimiter=",",
        header="sigma," + ",".join(map(str, mutation_rates)),
        comments="",
    )

c = plt.pcolormesh(mutation_rates, cross_immunities, final_infections)
plt.colorbar(c, label="Final number of infections")
plt.title(
    r"Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()

c = plt.contourf(mutation_rates, cross_immunities, final_infections)
plt.colorbar(c, label="Final number of infections")
plt.title(
    r"Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)"
)
plt.ylabel(r"Cross-immunity $\sigma$")
plt.xlabel(r"Mutation rate $m$")
plt.show()
