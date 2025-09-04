# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from solve_shsvd import solve_shsvd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy.optimize import brentq


def sir_model(t, x, beta=.15, gamma=.1):
    
    n = int(len(x)/3)
    
    dx = []
    for j in range(n):
        s, i, r = x[3*j:3*j+3]
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        dx.append(ds)
        dx.append(di)
        dx.append(dr)
        
    return dx


def total_infectious(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        
        i_ges = [total_infectious(y) for y in x]
        
        return i_ges
        
    else:
        n = int(len(x)/3)
        
        i_ges = 0
        
        for j in range(n):
            i_ges += (1-i_ges) * x[3*j+1]
            
        return i_ges

        

def mutation_probability(t, x, m=.5):
    
    i_ges = total_infectious(x)
    mutation_probability = m * i_ges
    
    return mutation_probability
        

def transition_function(t, x, i_start=.001):
    
    x = list(x)
    s = 1 - i_start
    i = i_start
    r = 0
    x.append(s)
    x.append(i)
    x.append(r)
    
    return x


def overall_mutation_probability(values, m=.5, stepsize=1):
    
    mutation_probabilities = [mutation_probability(0, value, m=mutation_rate) for value in values]
    overall_mutation_probability = stepsize * np.cumsum(mutation_probabilities)
    
    return overall_mutation_probability


def active_variants(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        
        variants = [active_variants(y) for y in x]
        
        return variants
        
    else:
        n = int(len(x)/3)
        
        variants = 0
        
        for j in range(n):
            if x[3*j+1] > .001:
                variants += 1
            
        return variants
    


x0 = [.999, .001, 0]
t_span = (0, 1000)
# betas = [.11, .15, .2]
# betas = .01 * np.arange(11, 26)
gamma = .1
# mutation_rates = [.01, .1, .2]
# mutation_rates = [.01, .1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2, .25, .3, .35, .4, .5, .6]

# df_empw_old = pd.read_csv('Mean_final_i.csv', index_col=0)
# df_empw_old.columns = df_empw_old.columns.astype(float)
# final_i_values = np.zeros((len(betas), len(mutation_rates)))

# for i, beta in enumerate(betas):
#     for j, mutation_rate in enumerate(mutation_rates):
#         if beta in df_empw_old.index and mutation_rate in df_empw_old.columns:
#             final_i_values[i, j] = df_empw_old.loc[beta, mutation_rate]
#         else:
#             sir_mutation_stats = solve_shsvd(sir_model, mutation_probability, transition_function, x0, t_span, mutation_rate, norm_order=[1, 2], collect_jump_times=True, flow_args=(beta, gamma), jump_rate_args=(mutation_rate, ), iterations=50, custom_evaluation_functions=[total_infectious])
#             #sir_mutation_stats.plot_cefs()
#             final_i_values[i, j] = sir_mutation_stats.cef_means[0][-1]
        
# ticks = [0, .0005, .001, .05, .1, 1]
# lower_section = [0, .001]
# middle_section = [.001, .1]
# upper_section = [.1, 1]

# cmap = plt.colormaps['PiYG']
# boundaries = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0], middle_section[1], 16)) + list(np.linspace(upper_section[0], upper_section[1], 6))  # Grenzen für die Farbkodierung
# levels = list(np.linspace(lower_section[0], lower_section[1], 22)) + list(np.linspace(middle_section[0]*1.01, middle_section[1], 16)) + list(np.linspace(upper_section[0]*1.01, upper_section[1], 6))  # Grenzen für die Farbkodierung
# norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# def final_recovered_equation(x, r0, s0):
#     return np.log(s0/(1 - x)) - r0 * x

# def final_recovered(r0, s0):
#     return brentq(final_recovered_equation, 0, 0.99, args=(r0, s0))


# critical_m = np.zeros(len(betas))

# for i, beta in enumerate(betas):
#     # sol = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
#     r_final = final_recovered(beta/gamma, x0[0])
#     critical_m[i] = gamma / r_final


# plt.figure(figsize=(8, 6))
# c = plt.pcolormesh(betas, mutation_rates, final_i_values.T, shading='auto', cmap=cmap, norm=norm)
# plt.colorbar(c, label='Mean Final I', ticks=ticks, boundaries=boundaries, format='%.4f')
# plt.ylabel('Mutation Rate')
# plt.xlabel('Beta')
# plt.title('Mean Final I for Different Betas and Mutation Rates')
# plt.yticks(mutation_rates)
# plt.xticks(betas)
# plt.show()

# plt.figure(figsize=(8, 6))
# c = plt.contourf(betas, mutation_rates, final_i_values.T, levels=levels, cmap=cmap, norm=norm)
# plt.colorbar(c, label=r'Mean $I_{ges}(1000)$', ticks=ticks, boundaries=boundaries, format='%.4f')
# plt.ylabel(r'Mutation rate $m$')
# plt.xlabel(r'$\beta$')
# plt.title(r'Mean $I_{ges}(1000)$ for different $\beta$ and mutation rates $m$')
# plt.yticks(mutation_rates)
# plt.xticks(betas)
# plt.plot(betas, critical_m, 'red')
# plt.tight_layout()
# plt.savefig('Mean_final_i_countourf.jpg', dpi=600)
# plt.show()


t_span = (0, 500)
beta = .15
mutation_rate = .2
sir_mutation_stats = solve_shsvd(sir_model, mutation_probability, transition_function, x0, t_span, mutation_rate, norm_order=[1, 2], collect_jump_times=True, flow_args=(beta, gamma), jump_rate_args=(mutation_rate, ), iterations=10, custom_evaluation_functions=[total_infectious])
fig = sir_mutation_stats.plot_cefs()

plt.figure(fig)
plt.grid()
plt.title('Plot of overall infectious fraction')
plt.xlabel('Overall infectious fraction')
plt.savefig('Example.png')
plt.show()

sir_mutation_stats.plot_inter_jump_times()

# df_empw = pd.DataFrame(final_i_values, index=betas, columns=mutation_rates)
# df_empw.to_csv('Mean_final_i.csv')