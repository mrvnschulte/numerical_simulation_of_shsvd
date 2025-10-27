# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:37:51 2025

@author: schulte
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from numerical_simulation_of_shsvd import solve_hsvd
from cross_immunity_model import (
    multi_variant_model_with_total_infections,
    new_variant_transition_function_with_total_infections,
)

"""
Computes the final number of infections with two variants in a cross-immunity
SIR-like model, where the second variant appears with a fixed delay, and
stores it in a numpy array
"""

t0 = 0
tf = 1000
x0 = np.array([0.999, 0, 0.001, 0.001])
beta = [0.15]
gamma = [0.1]
sigma = [[1]]
gamma_new = 0.1
t_eval = np.arange(tf + 1)

betas = np.linspace(0.11, 0.2, 91)
betas = np.round(betas, 3)
cross_immunities = np.linspace(0, 1, 201)
cross_immunities = np.round(cross_immunities, 3)
delays = np.linspace(0, 250, 126)
delays = np.round(delays)

final_infections_1 = np.zeros((len(betas), len(cross_immunities), len(delays)))
final_infections_2 = np.zeros((len(betas), len(cross_immunities), len(delays)))

file_name_1 = "final_infections_1_delayed_mutation.npy"
file_name_2 = "final_infections_2_delayed_mutation.npy"
file_name_betas = "betas_delayed_mutation.npy"
file_name_sigmas = "sigmas_delayed_mutation.npy"
file_name_delays = "delays_delayed_mutation.npy"

compute = True

# Überprüfen, ob die Dateien existieren
if os.path.exists(file_name_1) and os.path.exists(file_name_2):
    betas_old = np.load(file_name_betas)
    sigmas_old = np.load(file_name_sigmas)
    delays_old = np.load(file_name_delays)
    final_infections_1_old = np.load(file_name_1)
    final_infections_2_old = np.load(file_name_2)

    if (
        betas_old == betas
        and sigmas_old == cross_immunities
        and delays_old == delays
    ):
        compute = False
        final_infections_1 = final_infections_1_old
        final_infections_2 = final_infections_2_old

if compute:
    for i, beta_new in enumerate(betas):
        for j, s in enumerate(cross_immunities):
            for k, delay in enumerate(delays):
                sigma_new = [s]

                func = solve_hsvd(
                    multi_variant_model_with_total_infections,
                    x0,
                    (t0, tf),
                    t_eval=t_eval,
                    jump_times=[delay],
                    jump_time_transition_function=new_variant_transition_function_with_total_infections,
                    flow_args=(beta, gamma, sigma, 1),
                    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
                )

                final_infections_1[i, j, k] = func(tf)[-2]
                final_infections_2[i, j, k] = func(tf)[-1]

                print(
                    rf"Computed solution for $\beta={beta_new}, \sigma={s}, \tau={delay}"
                )

    np.save(file_name_1, final_infections_1)
    np.save(file_name_2, final_infections_2)
    np.save(file_name_betas, betas)
    np.save(file_name_sigmas, cross_immunities)
    np.save(file_name_delays, delays)

final_infections = final_infections_1 + final_infections_2

len_betas = len(betas) - 1
middle = len_betas // 2
lower_middle = middle // 2
upper_middle = middle + middle // 2

plot_indices = [0, lower_middle, middle, upper_middle, -1]

for idx in plot_indices:
    max_index = np.argmax(final_infections[idx], axis=1)
    max_delays = delays[max_index]

    c = plt.pcolormesh(delays, cross_immunities, final_infections_1[idx])
    plt.colorbar(c, label="Final number of infections with variant 1")
    plt.title(
        "Final number of infections with variant 1\n"
        rf"($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.pcolormesh(delays, cross_immunities, final_infections_2[idx])
    plt.colorbar(c, label="Final number of infections with variant 2")
    plt.title(
        "Final number of infections with variant 2\n"
        rf"($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.pcolormesh(delays, cross_immunities, final_infections[idx])
    plt.colorbar(c, label="Final number of infections")
    plt.title(
        rf"Final number of infections ($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections_1[idx])
    plt.colorbar(c, label="Final number of infections with variant 1")
    plt.title(
        "Final number of infections with variant 1\n"
        rf"($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections_2[idx])
    plt.colorbar(c, label="Final number of infections with variant 2")
    plt.title(
        "Final number of infections with variant 2\n"
        rf"($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections[idx])
    plt.colorbar(c, label="Final number of infections")
    plt.title(
        rf"Final number of infections ($R_0^1=1.5, R_0^2={betas[idx]/gamma_new}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.scatter(
        max_delays,
        cross_immunities,
        color="red",
        marker="x",
        s=100,
        label="Maximum",
    )
    plt.show()
