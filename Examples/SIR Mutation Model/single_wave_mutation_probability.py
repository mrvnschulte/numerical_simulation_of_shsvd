# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:45 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd

def sir_model(t, x, beta=.15, gamma=.1):
    s, i, r = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return [ds, di, dr]

def final_recovered_equation(x, r0, s0):
    return np.log(s0/(1 - x)) - r0 * x

def final_recovered(r0, s0):
    return brentq(final_recovered_equation, 0, 0.99, args=(r0, s0))

def stable_i(gamma, m, r0, s0):
    return 1 - gamma / (m * final_recovered(r0, s0))


betas = .01 * np.arange(11, 26)
gamma = .1
mutation_rates = [.01, .1, .12, .14, .16, .18, .2, .3, .4, .5, .6]
t_end = 1000
t_span = (0, t_end)
x0 = [.999, .001, 0]
t_eval = np.arange(0, t_end, 1)

empw = np.zeros((len(betas), len(mutation_rates)))
i_star = np.zeros((len(betas), len(mutation_rates)))
critical_m = np.zeros(len(betas))

for i, beta in enumerate(betas):
    # sol = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    r_final = final_recovered(beta/gamma, x0[0])
    critical_m[i] = gamma / r_final
    
    for j, mutation_rate in enumerate(mutation_rates):
        expected_mutations_per_wave = mutation_rate / gamma * r_final
        empw[i, j] = expected_mutations_per_wave
        i_star[i, j] = stable_i(gamma, mutation_rate, beta/gamma, x0[0])        


ticks = [0, .5, 1, 1.5, 2, 5]
lower_section = [0, 1]
middle_section = [1, 2]
upper_section = [2, 5.5]

cmap = plt.colormaps['PiYG']
boundaries = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0], middle_section[1], 16)) + list(np.linspace(upper_section[0], upper_section[1], 6))  # Grenzen für die Farbkodierung
levels = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0]*1.01, middle_section[1], 16)) + list(np.linspace(upper_section[0]*1.01, upper_section[1], 6))  # Grenzen für die Farbkodierung
norm = BoundaryNorm(boundaries, cmap.N, clip=True)


plt.figure(figsize=(8, 6))
c = plt.pcolormesh(betas, mutation_rates, empw.T, shading='auto', cmap=cmap, norm=norm)
plt.colorbar(c, label='EMPW', ticks=ticks, boundaries=boundaries, format='%.1f')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('EMPW for Different Betas and Mutation Rates')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, 'red')
plt.tight_layout()
plt.savefig('EMPW_Meshplot.png')
plt.show()

plt.figure(figsize=(8, 6))
c = plt.contourf(betas, mutation_rates, empw.T, levels=levels, cmap=cmap, norm=norm)
plt.colorbar(c, label='EMPW', ticks=ticks, boundaries=boundaries, format='%.1f')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('EMPW for Different Betas and Mutation Rates')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, 'red')
plt.tight_layout()
plt.savefig('EMPW_Contourplot.png')
plt.show()

df_empw = pd.DataFrame(empw, index=betas, columns=mutation_rates)
df_empw.to_csv('EMPW.csv')


ticks = [0, .2, .4, .6, .8, 1]
lower_section = [0, .5]
middle_section = [.5, .8]
upper_section = [.8, 1]

cmap = plt.colormaps['PiYG']
boundaries = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0], middle_section[1], 16)) + list(np.linspace(upper_section[0], upper_section[1], 6))  # Grenzen für die Farbkodierung
levels = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0]*1.01, middle_section[1], 16)) + list(np.linspace(upper_section[0]*1.01, upper_section[1], 6))  # Grenzen für die Farbkodierung
norm = BoundaryNorm(boundaries, cmap.N, clip=True)


plt.figure(figsize=(8, 6))
c = plt.pcolormesh(betas, mutation_rates, i_star.T, shading='auto', cmap=cmap, norm=norm)
plt.colorbar(c, label=r'i^*', ticks=ticks, boundaries=boundaries, format='%.1f')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('EMPW for Different Betas and Mutation Rates')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, 'red')
plt.tight_layout()
plt.savefig('EMPW_Meshplot.png')
plt.show()

plt.figure(figsize=(8, 6))
c = plt.contourf(betas, mutation_rates, i_star.T, levels=levels, cmap=cmap, norm=norm)
plt.colorbar(c, label=r'i^*', ticks=ticks, boundaries=boundaries, format='%.1f')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('EMPW for Different Betas and Mutation Rates')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, 'red')
plt.tight_layout()
plt.savefig('EMPW_Contourplot.png')
plt.show()

df_empw = pd.DataFrame(empw, index=betas, columns=mutation_rates)
df_empw.to_csv('EMPW.csv')