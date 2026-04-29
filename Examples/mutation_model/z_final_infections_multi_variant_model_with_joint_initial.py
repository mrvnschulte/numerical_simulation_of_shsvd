# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:31:53 2025

@author: schulte
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cross_immunity_model import multi_variant_model_with_total_infections

"""
Compute the number of final infections with each variant and in total for
different values for the cross-immunity and basic reproduction numbers of the
second variant, if both variants are present initially with all initial
infections being with both variants simultaneously
"""
t0 = 0
tf = 1000

r01 = 1.5
r02 = 2
g = 0.1

file1 = "Data/final_infections_1_multi_variant_model_joint_initial.csv"
file2 = "Data/final_infections_2_multi_variant_model_joint_initial.csv"

# Überprüfen, ob die Dateien existieren
if os.path.exists(file1) and os.path.exists(file2):
    # Einlesen der CSV-Dateien
    final_infections_1 = np.loadtxt(file1, delimiter=",", skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=",", skiprows=1)

    # Extrahiere sigmas und r02s
    r02s = final_infections_1[:, 0]  # Die erste Spalte ist sigmas
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
    sigmas = header[1:].astype(
        float
    )  # Wandle die Werte in float um, überspringe das erste Element

else:
    r02s = np.linspace(1, 2, 51)
    sigmas = np.linspace(0, 1, 21)

    final_infections_1 = np.zeros((len(r02s), len(sigmas)))
    final_infections_2 = np.zeros((len(r02s), len(sigmas)))

    for i, r02 in enumerate(r02s):
        for j, s in enumerate(sigmas):

            x0 = np.array([0.999, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.001])
            beta = [r01 * g, r02 * g]
            gamma = [g, g]
            sigma = np.array([[1, s], [s, 1]])

            # print(multi_variant_model(0, x0, beta, gamma, sigma))

            res12 = solve_ivp(
                multi_variant_model_with_total_infections,
                (t0, tf),
                x0,
                t_eval=np.arange(tf),
                args=(beta, gamma, sigma, len(beta)),
            )

            final_infections_1[i, j] = res12.y[-2][-1]
            final_infections_2[i, j] = res12.y[-1][-1]
            # plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12', 'n1', 'n2'])
            # plt.legend()
            # plt.show()

    final_infections_1_with_r02 = np.column_stack((r02s, final_infections_1))
    final_infections_2_with_r02 = np.column_stack((r02s, final_infections_2))

    # Speichern der Matrizen als CSV-Dateien
    np.savetxt(
        file1,
        final_infections_1_with_r02,
        delimiter=",",
        header="r02," + ",".join(map(str, sigmas)),
        comments="",
    )
    np.savetxt(
        file2,
        final_infections_2_with_r02,
        delimiter=",",
        header="r02," + ",".join(map(str, sigmas)),
        comments="",
    )

c = plt.pcolormesh(sigmas, r02s, final_infections_1)
plt.colorbar(c, label="Final number of infections with variant 1")
plt.title(
    r"Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)"
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_2)
plt.colorbar(c, label="Final number of infections with variant 2")
plt.title(
    r"Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)"
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label="Final number of infections")
plt.title(r"Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)")
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1)
plt.colorbar(c, label="Final number of infections with variant 1")
plt.title(
    r"Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)"
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_2)
plt.colorbar(c, label="Final number of infections with variant 2")
plt.title(
    r"Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)"
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label="Final number of infections")
plt.title(r"Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)")
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel(r"$R_0^2$")
plt.show()
