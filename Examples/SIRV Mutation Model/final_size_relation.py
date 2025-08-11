# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:28:28 2025

@author: schulte
"""

from solve_shsvd import solve_shsvd, solve_shsvd_single_run
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


def final_recovered_equation(x, r0, s0):
    return np.log(s0/(1 - x)) - r0 * x

def final_recovered(r0, s0):
    return brentq(final_recovered_equation, 0, 0.9999999999999, args=(r0, s0))

def mutation_probability(mutation_rate, gamma, r_infty):
    return 1 - np.exp(-mutation_rate/gamma * r_infty)

def expected_final_size_vacc1(v0, mutation_rate, gamma1, r1_infty, r2_infty):
    r1_infty_vacc = r1_infty * (1 - v0)
    e = (r1_infty_vacc
         + mutation_probability(mutation_rate, gamma, r1_infty_vacc) 
         * r2_infty)
    return e

def expected_final_size_vacc2(v0, mutation_rate, gamma1, r1_infty, r2_infty):
    r2_infty_vacc = r2_infty * (1 - v0)
    e = (r1_infty
         + mutation_probability(mutation_rate, gamma, r1_infty)
         * r2_infty_vacc)
    return e

r0 = np.linspace(0.5, 5, 101)
s0 = .99
gamma = .1
mutation_rate = .1
r_infty = np.zeros(len(r0))

for i, r in enumerate(r0):
    r_infty[i] = final_recovered(r, s0)
    
plt.plot(r0, r_infty)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$r_\infty$')
plt.title(rf'Final size of an epidemic in relation to $R_0$ with $s_0$={s0}')
#plt.show()

probability_of_mutation = mutation_probability(mutation_rate, gamma, r_infty)

plt.plot(r0, probability_of_mutation)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$P(M)$')
plt.title(rf'Probability of mutation in relation to $R_0$ with $s_0$={s0}')
plt.show()


s0 = .99
beta = np.linspace(.2, 1, 101)
gamma = np.linspace(.1, .5, 101)
mutation_rate = .2

r_infty = np.zeros((len(beta), len(gamma)))
probability_of_mutation = np.zeros((len(beta), len(gamma)))

for i, b in enumerate(beta):
    for j, g in enumerate(gamma):
        r = b/g
        r_infty[i, j] = final_recovered(r, s0)
        probability_of_mutation[i, j] = mutation_probability(mutation_rate, g,
                                                             r_infty[i, j])


c = plt.pcolormesh(beta, gamma, r_infty)
plt.colorbar(c, label=r'$r_\infty$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.title(r'Final size in relation to $\beta$ and $\gamma$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(beta, gamma, r_infty)
plt.colorbar(c, label=r'$r_\infty$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.title(r'Final size in relation to $\beta$ and $\gamma$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()


c = plt.pcolormesh(beta, gamma, probability_of_mutation)
plt.colorbar(c, label=r'$P(M)$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.title(r'Mutation probability in relation to $\beta$ and $\gamma$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(beta, gamma, probability_of_mutation)
plt.colorbar(c, label=r'$P(M)$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.title(r'Mutation probability in relation to $\beta$ and $\gamma$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()