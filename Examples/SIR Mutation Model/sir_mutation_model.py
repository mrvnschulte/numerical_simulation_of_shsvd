# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from solve_shsvd import solve_shsvd
import numpy as np
import matplotlib.pyplot as plt


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


x0 = [.999, .001, 0]
t_span = (0, 1000)
betas = .01 * np.arange(11, 26)
gamma = .1
mutation_rates = [.01, .1, .12, .14, .16, .18, .2, .3, .4, .5, .6]

final_i_values = np.zeros((len(betas), len(mutation_rates)))

for i, beta in enumerate(betas):
    for j, mutation_rate in enumerate(mutation_rates):
        sir_mutation_stats = solve_shsvd(sir_model, mutation_probability, transition_function, x0, t_span, mutation_rate, norm_order=[1, 2], collect_jump_times=True, flow_args=(beta, gamma), jump_rate_args=(mutation_rate, ), iterations=50, custom_evaluation_functions=[total_infectious])
        #sir_mutation_stats.plot_cefs()
        final_i_values[i, j] = sir_mutation_stats.cef_means[0][-1]
        
plt.figure(figsize=(8, 6))
c = plt.pcolormesh(betas, mutation_rates, final_i_values.T, shading='auto')
plt.colorbar(c, label='Final I')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('Final I für verschiedene Beta- und Mutationsraten')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.show()