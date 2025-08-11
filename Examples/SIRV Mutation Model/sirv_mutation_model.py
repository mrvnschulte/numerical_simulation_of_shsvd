# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:18 2025

@author: schulte
"""

from solve_shsvd import solve_shsvd, solve_shsvd_single_run
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


def sirv_model(t, x, betas=[.15, .3], gammas=[.1, .1], alphas=[0, 0]):
    
    n = int(len(x)/4)
    
    dx = []
    for j in range(n):
        s, i, r, v = x[4*j:4*j+4]
        ds = -betas[j] * s * i - alphas[j] * s
        di = betas[j] * s * i - gammas[j] * i
        dr = gammas[j] * i
        dv = alphas[j] * s
        dx.append(ds)
        dx.append(di)
        dx.append(dr)
        dx.append(dv)
        
    return dx


def total_infectious(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        
        i_ges = [total_infectious(y) for y in x]
        
        return i_ges
        
    else:
        n = int(len(x)/4)
        
        i_ges = 0
        
        for j in range(n):
            i_ges += (1-i_ges) * x[4*j+1]
            
        return i_ges
    
    
def total_recovered(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        
        r_ges = [total_recovered(y) for y in x]
        
        return r_ges
        
    else:
        n = int(len(x)/4)
        
        r_ges = 0
        
        for j in range(n):
            r_ges += x[4*j+2]
            
        return r_ges



def infectious_with_first_variant(x):
    
    i_1 = x[1]
    
    return i_1
        

def mutation_probability(t, x, m=.5):
    
    if len(x) == 4:
        i_1 = infectious_with_first_variant(x)
        mutation_probability = m * i_1
    else:
        mutation_probability = 0
    
    return mutation_probability
        

def transition_function(t, x, i_start=.0001):
    
    x = list(x)
    s = 1 - i_start
    i = i_start
    r = 0
    v = 0
    x.append(s)
    x.append(i)
    x.append(r)
    x.append(v)
    
    return x


def overall_mutation_probability(values, m=.5, stepsize=1):
    
    mutation_probabilities = [mutation_probability(0, value, m=m) for value in values]
    overall_mutation_probability = stepsize * np.cumsum(mutation_probabilities)
    
    return overall_mutation_probability


def active_variants(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        
        variants = [active_variants(y) for y in x]
        
        return variants
        
    else:
        n = int(len(x)/4)
        
        variants = 0
        
        for j in range(n):
            if x[4*j+1] > .001:
                variants += 1
            
        return variants
    
    
def final_recovered_equation(x, r0, s0):
    return np.log(s0/(1 - x)) - r0 * x

def final_recovered(r0, s0):
    return brentq(final_recovered_equation, 0, 0.9999, args=(r0, s0))

def number_of_mutations(x):
    
    if isinstance(x[0], (list, np.ndarray)):
        m = [number_of_mutations(y) for y in x]
    else:
        m = int(len(x)/4) - 1
    
    return m



t0 = 0
t_end = 400
x0 = [.999, .001, 0, 0]
t_span = (t0, t_end)
gammas = [.1, .05]
betas = [.15, .15]
alphas = [0, 0]
mutation_rate = .2
t_eval = np.arange(t0, t_end, 1)
min_iterations = 250
max_iterations = 5000
tol = 1e-2

r01 = betas[0]/gammas[0]
r02 = betas[1]/gammas[1]
r1_infty = final_recovered(r01, x0[0])
r2_infty = final_recovered(r02, 1-.0001)

probability_of_mutation = 1 - np.exp(-mutation_rate/gammas[0] * r1_infty)
expected_r_infty = r1_infty + probability_of_mutation * r2_infty


alphas_vaccination1 = [0.001, 0]
vaccination1_result = solve_ivp(sirv_model, t_span, x0,
                                args=(betas, gammas, alphas_vaccination1))
r1_infty_vaccination1 = vaccination1_result.y.T[-1][2]

pm_vacc1 = 1 - np.exp(-mutation_rate/gammas[0] * r1_infty_vaccination1)
er_vacc1 = r1_infty_vaccination1 + pm_vacc1 * r2_infty


alphas_vaccination2 = [0.001, 0.001]
vaccination2_result = solve_ivp(sirv_model, t_span, [1-.0001, .0001, 0, 0],
                                args=([betas[1], betas[1]],
                                      [gammas[1], gammas[1]],
                                      [alphas_vaccination2[1], alphas_vaccination2[1]]))
r2_infty_vaccination2 = vaccination2_result.y.T[-1][2]

pm_vacc2 = 1 - np.exp(-mutation_rate/gammas[0] * r1_infty)
er_vacc2 = r1_infty + pm_vacc2 * r2_infty_vaccination2


print(f'Probability of mutation: {probability_of_mutation}')
print(f'Expected r infty: {expected_r_infty}')

print(f'Probability of mutation (vacc. 1): {pm_vacc1}')
print(f'Expected r infty (vacc. 1): {er_vacc1}')

print(f'Probability of mutation (vacc. 2): {pm_vacc2}')
print(f'Expected r infty (vacc. 2): {er_vacc2}')


'''
sirv_mutation_run = solve_shsvd_single_run(sirv_model, mutation_probability, transition_function, x0, t_span, mutation_rate, flow_args=(betas, gammas, alphas), jump_rate_args=(mutation_rate, ), t_eval=t_eval)
        


sir_mutation_stats = solve_shsvd(sirv_model, mutation_probability,
                                 transition_function, x0, t_span,
                                 mutation_rate, norm_order=[1, 2],
                                 collect_jump_times=True,
                                 flow_args=(betas, gammas, alphas),
                                 jump_rate_args=(mutation_rate, ),
                                 min_iterations=min_iterations,
                                 max_iterations=max_iterations, 
                                 tol=tol, convergence_estimate=('cef', 0),
                                 custom_evaluation_functions=[number_of_mutations])
fig = sir_mutation_stats.plot_cefs()

plt.figure(fig)
plt.grid()
plt.title('Plot of overall infectious fraction')
plt.xlabel('Overall infectious fraction')
plt.savefig('Example.png')
plt.show()
'''