# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:17:45 2026

@author: schulte
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    multi_variant_model_with_demographics,
    variant_in_set_matrix,
    map_x_to_I,
)
from invasion_reproduction_number import invasion_reproduction_number

"""
Investigates the stability of the multi-variant model
"""


def classify_r0(r0_23, r0_32):
    if (r0_23 > 1) and (r0_32 > 1):
        return 2
    elif (r0_23 <= 1) and (r0_32 > 1):
        return 1
    elif (r0_23 > 1) and (r0_32 <= 1):
        return 0
    else:
        return -1


t0 = 0
tf = 1000
beta = [0.2, 0.3]
gamma = [0.1, 0.1]
sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
b = 0.01
x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])
n = 2

sol = solve_ivp(
    multi_variant_model_with_demographics,
    (t0, tf),
    x0,
    args=(beta, gamma, sigma, b),
)

plt.plot(sol.t, sol.y.T)
plt.show()
print(sol.y.T[-1])

x0 = sol.y.T[-1]

beta_1 = np.linspace(0.1, 0.5, 501)
sigma_23 = np.linspace(0, 1, 201)
sigma_12 = np.linspace(0, 1, 11)

for s_12 in sigma_12:
    mask_threshold = 1e-4
    # results_mask = []
    results_flags = []

    for p in beta_1:
        # row_masks = []
        row_flags = []
        for s in sigma_23:

            beta = [0.2, 0.25, p]
            gamma = [0.1, 0.1, 0.1]
            sigma = np.array(
                [[1.0, s_12, s_12 * s], [s_12, 1.0, s], [s_12 * s, s, 1]]
            )
            # b = 0.01
            # x0 = np.zeros(27)
            # n = 3
            # x0[0] = 0.998
            # x0[2] = 0.001
            # x0[4] = 0.001
            # sol = solve_ivp(
            #     multi_variant_model_with_demographics,
            #     (t0, tf),
            #     x0,
            #     args=(beta, gamma, sigma, b),
            # )

            # # plt.plot(sol.t, sol.y.T)
            # # plt.show()
            # # print(sol.y.T[-1])
            # x0 = sol.y.T[-1]
            # x0[0] -= 1e-4
            # x0[10] += 1e-4

            # sol = solve_ivp(
            #     multi_variant_model_with_demographics,
            #     (t0, tf),
            #     x0,
            #     args=(beta, gamma, sigma, b),
            # )

            # # plt.plot(sol.t, sol.y.T)
            # # plt.title(f"{s}, {p}")
            # # plt.show()
            # # print(sol.y.T[-1])

            # I = map_x_to_I(sol.y.T[-1])
            # V = variant_in_set_matrix(3)
            # size = 2**n
            # I_variant = (V @ I @ np.ones((size, 1))).flatten()

            # mask = I_variant > mask_threshold
            r0_12 = invasion_reproduction_number(
                beta[0] / gamma[0], beta[1] / gamma[1], sigma[0, 1]
            )
            r0_21 = invasion_reproduction_number(
                beta[1] / gamma[1], beta[0] / gamma[0], sigma[1, 0]
            )
            r0_13 = invasion_reproduction_number(
                beta[0] / gamma[0], beta[2] / gamma[2], sigma[0, 2]
            )
            r0_31 = invasion_reproduction_number(
                beta[2] / gamma[2], beta[0] / gamma[0], sigma[2, 0]
            )
            r0_23 = invasion_reproduction_number(
                beta[1] / gamma[1], beta[2] / gamma[2], sigma[1, 2]
            )
            r0_32 = invasion_reproduction_number(
                beta[2] / gamma[2], beta[1] / gamma[1], sigma[2, 1]
            )
            classifier_12 = classify_r0(r0_12, r0_21)
            classifier_13 = classify_r0(r0_13, r0_31)
            classifier_23 = classify_r0(r0_23, r0_32)

            classifier = classifier_23 + 3 * classifier_13 + 9 * classifier_12

            # row_masks.append(mask)
            row_flags.append(classifier)

            # print(I_variant)
        # results_mask.append(np.stack(row_masks, axis=0))
        results_flags.append(row_flags)

    # results_mask = np.array(results_mask)

    # for i, p in enumerate(beta_1):
    #     for j, s in enumerate(sigma_23):
    #         mask_ij = results_mask[i, j]
    #         alive_indices = np.where(mask_ij)[0]
    #         print(f"p={p}, s={s}, lebende Varianten-Indizes: {alive_indices}")

    # alive_counts = np.zeros((len(beta_1), len(sigma_23)), dtype=int)

    # for i, p in enumerate(beta_1):
    #     for j, s in enumerate(sigma_23):
    #         mask_ij = results_mask[i, j]  # bool array der Länge 2**n
    #         alive_counts[i, j] = int(np.sum(mask_ij))

    # plt.figure(figsize=(6, 5))
    # plt.imshow(
    #     alive_counts,
    #     origin="lower",
    #     aspect="auto",
    #     extent=[min(sigma_23), max(sigma_23), min(beta_1), max(beta_1)],
    # )
    # plt.colorbar(label="Anzahl lebender Varianten")
    # plt.xlabel("s (sigma_23)")
    # plt.ylabel("p (beta_1)")
    # plt.title("Überlebenskombinationen je (p, s)")
    # plt.tight_layout()
    # plt.show()

    results_flags = np.array(
        results_flags, dtype=float
    )  # dtype je nach Bedarf int/floats

    # Heatmap plotten (wie zuvor, aber mit results_flags)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        results_flags,
        origin="lower",
        aspect="auto",
        extent=[min(sigma_23), max(sigma_23), min(beta_1), max(beta_1)],
        cmap="viridis",  # oder ein anderes Colormap
        vmin=np.min(results_flags),
        vmax=np.max(results_flags),
    )
    plt.colorbar(label="Kennzahl (0, 1, 2, ...)")
    plt.xlabel("s (sigma_23)")
    plt.ylabel("p (beta_1)")
    plt.title("Kennzahlen je (p, s)")
    plt.tight_layout()
    plt.show()
