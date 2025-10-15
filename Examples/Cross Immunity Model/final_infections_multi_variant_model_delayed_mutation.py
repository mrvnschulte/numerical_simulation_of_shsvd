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
SIR-like model, where the second variant appears with a fixed delay
"""

t0 = 0
tf = 1000
x0 = np.array([0.999, 0, 0.001, 0.001])
beta = [0.15]
gamma = [0.1]
sigma = [[1]]
gamma_new = 0.1
t_eval = np.arange(tf + 1)

r02s = np.linspace(1.1, 2, 10)

for r02 in r02s:
    r02 = round(r02, 1)
    beta_new = r02 * gamma_new

    file1 = f"Data/final_infections_1_delayed_mutation_r02_{r02}.csv"
    file2 = f"Data/final_infections_2_delayed_mutation_r02_{r02}.csv"

    # Überprüfen, ob die Dateien existieren
    if os.path.exists(file1) and os.path.exists(file2):
        # Einlesen der CSV-Dateien
        final_infections_1 = np.loadtxt(file1, delimiter=",", skiprows=1)
        final_infections_2 = np.loadtxt(file2, delimiter=",", skiprows=1)

        # Extrahiere sigmas und r02s
        cross_immunities = final_infections_1[
            :, 0
        ]  # Die erste Spalte ist sigmas
        final_infections_1 = final_infections_1[
            :, 1:
        ]  # Der Rest sind die Werte für final_infections_1
        final_infections_2 = final_infections_2[
            :, 1:
        ]  # Der Rest sind die Werte für final_infections_2

        # sigmas extrahieren, indem wir die erste Zeile der CSV Datei lesen und die Kopfzeile ignorieren
        header = np.loadtxt(
            file1, delimiter=",", max_rows=1, dtype=str
        )  # Lies die Kopfzeile als String
        delays = header[1:].astype(
            float
        )  # Wandle die Werte in float um, überspringe das erste Element

    else:
        cross_immunities = np.linspace(0, 1, 201)
        delays = np.linspace(0, 250, 126)

        final_infections_1 = np.zeros((len(cross_immunities), len(delays)))
        final_infections_2 = np.zeros((len(cross_immunities), len(delays)))

        for i, s in enumerate(cross_immunities):
            for j, delay in enumerate(delays):
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

                final_infections_1[i, j] = func(tf)[-2]
                final_infections_2[i, j] = func(tf)[-1]

        final_infections_1_with_sigma = np.column_stack(
            (cross_immunities, final_infections_1)
        )
        final_infections_2_with_sigma = np.column_stack(
            (cross_immunities, final_infections_2)
        )

        # Speichern der Matrizen als CSV-Dateien
        np.savetxt(
            file1,
            final_infections_1_with_sigma,
            delimiter=",",
            header="sigma," + ",".join(map(str, delays)),
            comments="",
        )
        np.savetxt(
            file2,
            final_infections_2_with_sigma,
            delimiter=",",
            header="sigma," + ",".join(map(str, delays)),
            comments="",
        )

    final_infections = final_infections_1 + final_infections_2
    max_index = np.argmax(final_infections, axis=1)
    max_delays = delays[max_index]
    
    c = plt.pcolormesh(delays, cross_immunities, final_infections_1)
    plt.colorbar(c, label="Final number of infections with variant 1")
    plt.title(
        "Final number of infections with variant 1\n"
        rf"($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.pcolormesh(delays, cross_immunities, final_infections_2)
    plt.colorbar(c, label="Final number of infections with variant 2")
    plt.title(
        "Final number of infections with variant 2\n"
        rf"($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.pcolormesh(delays, cross_immunities, final_infections)
    plt.colorbar(c, label="Final number of infections")
    plt.title(
        rf"Final number of infections ($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections_1)
    plt.colorbar(c, label="Final number of infections with variant 1")
    plt.title(
        "Final number of infections with variant 1\n"
        rf"($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections_2)
    plt.colorbar(c, label="Final number of infections with variant 2")
    plt.title(
        "Final number of infections with variant 2\n"
        rf"($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.show()

    c = plt.contourf(delays, cross_immunities, final_infections)
    plt.colorbar(c, label="Final number of infections")
    plt.title(
        rf"Final number of infections ($R_0^1=1.5, R_0^2={r02}, \gamma_1=\gamma_2=0.1$)"
    )
    plt.ylabel(r"Cross-immunity $\sigma$")
    plt.xlabel(r"Time of mutation")
    plt.scatter(max_delays, cross_immunities, color='red', marker='x', s=100, label='Maximum')
    plt.show()
