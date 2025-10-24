# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from numerical_simulation_of_shsvd import solve_shsvd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy.optimize import brentq
from sir_mutation_model import (
    sir_model,
    total_infectious,
    mutation_probability,
)
from single_wave_mutation_probability import final_recovered

"""
Unfinished script for computing multi-variant SIR model with partial immunity
(but only at time of mutation and for one variant). Might not be continued.
"""


def transition_function(t, x, i_start=0.001, partial_immunity=0):

    x = list(x)

    if partial_immunity > 0:

        n = len(x) // 3
        i_all = np.sum(x[1::3])

        p = np.random.random() * i_all
        i_now = np.cumsum(x[1::3])
        index = np.searchsorted(i_now, p)

        if index < n:
            r = partial_immunity * x[3 * index + 1]
            i = i_start
            s = 1 - i_start - r
            x.extend([s, i, r])

    else:
        s = 1 - i_start
        i = i_start
        r = 0
        x.append(s)
        x.append(i)
        x.append(r)

    return x


file_name = "Mean_final_i_partial_immune_escape.csv"

partial_immunity = 0.5
i_start = 0.001

x0 = [0.999, 0.001, 0]
t_span = (0, 1000)
betas = [0.11, 0.15, 0.2]
betas = 0.01 * np.arange(11, 26)
gamma = 0.1
mutation_rates = [0.01, 0.1, 0.2]
mutation_rates = [
    0.01,
    0.1,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
]

try:
    df_empw_old = pd.read_csv(file_name, index_col=0)
    df_empw_old.columns = df_empw_old.columns.astype(float)
    new_data = False
except:
    new_data = True
final_i_values = np.zeros((len(betas), len(mutation_rates)))

for i, beta in enumerate(betas):
    for j, mutation_rate in enumerate(mutation_rates):
        if not new_data:
            if (
                beta in df_empw_old.index
                and mutation_rate in df_empw_old.columns
            ):
                final_i_values[i, j] = df_empw_old.loc[beta, mutation_rate]
            else:
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
                    iterations=50,
                    custom_evaluation_functions=[total_infectious],
                )
                # sir_mutation_stats.plot_cefs()
                final_i_values[i, j] = sir_mutation_stats.cef_means[0][-1]
        else:
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
                iterations=50,
                custom_evaluation_functions=[total_infectious],
            )
            # sir_mutation_stats.plot_cefs()
            final_i_values[i, j] = sir_mutation_stats.cef_means[0][-1]


ticks = [0, 0.0005, 0.001, 0.05, 0.1, 1]
lower_section = [0, 0.001]
middle_section = [0.001, 0.1]
upper_section = [0.1, 1]

cmap = plt.colormaps["PiYG"]
boundaries = (
    list(np.linspace(lower_section[0], lower_section[1], 22))
    + list(np.linspace(middle_section[0], middle_section[1], 16))
    + list(np.linspace(upper_section[0], upper_section[1], 6))
)  # Grenzen für die Farbkodierung
levels = (
    list(np.linspace(lower_section[0], lower_section[1], 22))
    + list(np.linspace(middle_section[0] * 1.01, middle_section[1], 16))
    + list(np.linspace(upper_section[0] * 1.01, upper_section[1], 6))
)  # Grenzen für die Farbkodierung
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

critical_m = np.zeros(len(betas))

for i, beta in enumerate(betas):
    # sol = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    r_final = final_recovered(beta / gamma, x0[0])
    critical_m[i] = gamma / r_final


plt.figure(figsize=(8, 6))
c = plt.pcolormesh(
    betas,
    mutation_rates,
    final_i_values.T,
    shading="auto",
    cmap=cmap,
    norm=norm,
)
plt.colorbar(
    c, label="Mean Final I", ticks=ticks, boundaries=boundaries, format="%.4f"
)
plt.ylabel("Mutation Rate")
plt.xlabel("Beta")
plt.title("Mean Final I for Different Betas and Mutation Rates")
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.show()

plt.figure(figsize=(8, 6))
c = plt.contourf(
    betas,
    mutation_rates,
    final_i_values.T,
    levels=levels,
    cmap=cmap,
    norm=norm,
)
plt.colorbar(
    c, label="Mean Final I", ticks=ticks, boundaries=boundaries, format="%.4f"
)
plt.ylabel("Mutation Rate")
plt.xlabel("Beta")
plt.title("Mean Final I for Different Betas and Mutation Rates")
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, "red")
plt.tight_layout()
plt.savefig("Mean_final_i_countourf.png")
plt.show()


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
    transition_args=(i_start, partial_immunity),
    iterations=500,
    custom_evaluation_functions=[total_infectious],
)
fig = sir_mutation_stats.plot_cefs()

plt.figure(fig)
plt.grid()
plt.title("Plot of overall infectious fraction")
plt.xlabel("Overall infectious fraction")
plt.savefig("Example.png")
plt.show()


df_empw = pd.DataFrame(final_i_values, index=betas, columns=mutation_rates)
df_empw.to_csv(file_name)
