# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:55:49 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os


def sir_model(t, x, beta, gamma):

    s, i, r = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i

    return [ds, di, dr]


t0 = 0
tf = 500
x0 = [0.999, 0.001, 0]
beta = 0.15
gamma = 0.1
m = 1

t_span = (t0, tf)

t_eval = np.linspace(t0, tf, tf + 1)

file1 = "Data/final_infections_1_delayed_mutation.csv"
file2 = "Data/final_infections_2_delayed_mutation.csv"

# Überprüfen, ob die Dateien existieren
if os.path.exists(file1) and os.path.exists(file2):
    # Einlesen der CSV-Dateien
    final_infections_1 = np.loadtxt(file1, delimiter=",", skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=",", skiprows=1)

    # Extrahiere sigmas und r02s
    cross_immunities = final_infections_1[:, 0]  # Die erste Spalte ist sigmas
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

final_infections = final_infections_1 + final_infections_2

mutation_rates = np.linspace(0, 1, 21)
expected_final_infections = np.zeros(
    (len(cross_immunities), len(mutation_rates))
)

for j, m in enumerate(mutation_rates):
    res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    i = res.y[1]
    r = res.y[2]

    probability_of_no_mutation = np.exp(-m * r[-1] / gamma)
    if probability_of_no_mutation < 1:
        mutation_time_pdf = (m * i * np.exp(-m * r / gamma)) / (
            1 - probability_of_no_mutation
        )
    else:
        mutation_time_pdf = 0

    for i, sigma in enumerate(cross_immunities):
        final_infections_sigma = final_infections[i]
        interpolated_values = np.interp(t_eval, delays, final_infections_sigma)
        expected_final_infections[i, j] = (
            np.sum(
                mutation_time_pdf
                * interpolated_values
                * (1 - probability_of_no_mutation)
            )
            + r[-1] * probability_of_no_mutation
        )

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
