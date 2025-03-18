# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:45 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt

def sir_model(t, x, beta=.15, gamma=.1):
    
    s, i, r = x
    
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    
    dx = [ds, di, dr]
    
    return dx


def final_recovered_equation(x, r0, s0):
    
    return np.log(s0/(1 - x)) - r0 * x

    
def final_recovered(r0, s0):
    
    r_final = brentq(final_recovered_equation, 0, 0.99, args=(r0, s0))
    
    return r_final
    
    
betas = .01 * np.arange(11, 26)
gamma = .1
mutation_rates = [.01, .1, .12, .14, .16, .18, .2, .3, .4, .5, .6]
t_end = 1000
t_span = (0, t_end)
x0 = [.999, .001, 0]
t_eval = np.arange(0, t_end, 1)

swmp = np.zeros((len(betas), len(mutation_rates)))
critical_m = np.zeros(len(betas))

for i, beta in enumerate(betas):

    sol = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    #plt.plot(t_eval, sol.y.T)
    r_final = final_recovered(beta/gamma, x0[0])
    critical_m[i] = gamma / r_final
    
    for j, mutation_rate in enumerate(mutation_rates):

        mutation_probability = mutation_rate * np.cumsum(sol.y[1])
        #plt.plot(t_eval, mutation_probability)
        
        single_wave_mutation_probability = mutation_rate / gamma * r_final
        
        swmp[i, j] = single_wave_mutation_probability
        
        
plt.figure(figsize=(8, 6))
c = plt.pcolormesh(betas, mutation_rates, swmp.T, shading='auto')
plt.colorbar(c, label='SWMP')
plt.ylabel('Mutation Rate')
plt.xlabel('Beta')
plt.title('SWMP für verschiedene Beta- und Mutationsraten')
plt.yticks(mutation_rates)
plt.xticks(betas)
plt.plot(betas, critical_m, 'red')
plt.show()