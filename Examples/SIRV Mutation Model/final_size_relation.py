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
from scipy.optimize import brentq, minimize
from scipy.integrate import solve_ivp


def final_recovered_equation(x, r0, s0):
    return np.log(s0/(1 - x)) - r0 * x

def final_recovered(r0, s0):
    try:
        final_r = brentq(final_recovered_equation, 0, 0.9999999999999, args=(r0, s0))
    except:
        final_r = 0.9999999999999
    return final_r

def mutation_probability(mutation_rate_per_gamma, r_infty):
    return 1 - np.exp(-mutation_rate_per_gamma * r_infty)

def expected_final_size_vacc1(v0, mutation_rate_per_gamma1, r1_infty, r2_infty,
                              alpha=1):
    r1_infty_vacc = r1_infty * (1 - v0)
    e = (r1_infty_vacc
         + alpha * mutation_probability(mutation_rate_per_gamma1,
                                        r1_infty_vacc) 
         * r2_infty)
    return e

def expected_final_size_vacc2(v0, mutation_rate_per_gamma1, r1_infty, r2_infty,
                              alpha=1):
    r2_infty_vacc = r2_infty * (1 - v0)
    e = (r1_infty
         + alpha * mutation_probability(mutation_rate_per_gamma1, r1_infty)
         * r2_infty_vacc)
    return e

def vacc2_vs_vacc1(v0, mutation_rate_per_gamma1, r01, r02, alpha=1):
    r1_infty = final_recovered(r01, s0)
    r2_infty = final_recovered(r02, s0)
    efs_vacc1 = expected_final_size_vacc1(v0, mutation_rate_per_gamma1,
                                          r1_infty, r2_infty, alpha)
    efs_vacc2 = expected_final_size_vacc2(v0, mutation_rate_per_gamma1,
                                          r1_infty, r2_infty, alpha)
    utility = efs_vacc2 - efs_vacc1
    
    return utility

def threshold_alpha(final_size_ratio, expected_mutations):
    return -final_size_ratio / (np.exp(-expected_mutations) *
                                (1 + expected_mutations) - 1)

def threshold_vaccination(final_size_ratio, expected_mutations, alpha=1):
    return 0

r0 = np.linspace(0.5, 5, 101)
s0 = .99
mutation_rate_per_gamma = 1
r_infty = np.zeros(len(r0))

for i, r in enumerate(r0):
    r_infty[i] = final_recovered(r, s0)
    
plt.plot(r0, r_infty)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$r_\infty$')
plt.title(rf'Final size of an epidemic in relation to $R_0$ with $s_0$={s0}')
#plt.show()

probability_of_mutation = mutation_probability(mutation_rate_per_gamma,
                                               r_infty)

plt.plot(r0, probability_of_mutation)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$P(M)$')
plt.title(rf'Probability of mutation in relation to $R_0$ with $s_0$={s0}')
plt.show()


s0 = .99
r0 = np.linspace(.5, 5, 101)
mutation_rate_per_gamma = np.linspace(.1, 5, 101)

r_infty = np.zeros((len(r0), len(mutation_rate_per_gamma)))
probability_of_mutation = np.zeros((len(r0), len(mutation_rate_per_gamma)))

for i, r in enumerate(r0):
    for j, m in enumerate(mutation_rate_per_gamma):
        r_infty[i, j] = final_recovered(r, s0)
        probability_of_mutation[i, j] = mutation_probability(m, r_infty[i, j])


c = plt.pcolormesh(r0, mutation_rate_per_gamma, r_infty.T)
plt.colorbar(c, label=r'$r_\infty$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{m}{\gamma}$')
plt.title(r'Final size in relation to $R_0$ and $\frac{m}{\gamma}$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(r0, mutation_rate_per_gamma, r_infty.T)
plt.colorbar(c, label=r'$r_\infty$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{m}{\gamma}$')
plt.title(r'Final size in relation to $R_0$ and $\frac{m}{\gamma}$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()


c = plt.pcolormesh(r0, mutation_rate_per_gamma, probability_of_mutation.T)
plt.colorbar(c, label=r'$P(M)$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{m}{\gamma}$')
plt.title(r'Mutation probability in relation to $R_0$ and $\frac{m}{\gamma}$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(r0, mutation_rate_per_gamma, probability_of_mutation.T)
plt.colorbar(c, label=r'$P(M)$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{m}{\gamma}$')
plt.title(r'Mutation probability in relation to $R_0$ and $\frac{m}{\gamma}$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

# res = minimize(vacc2_vs_vacc1, (.5, .2, .2, .1, 3),
#                bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 10)])

# v0 = np.linspace(0, 1, 101)
# r1_infty = .2
# r2_infty = .9
# m = .2
# gamma1 = .1

# dadv = -r1_infty - m / gamma1 * r1_infty * r2_infty * np.exp(-m / gamma1 * 
#                                                              r1_infty * (1-v0))
# dbdv = -r2_infty + r2_infty * np.exp(-m / gamma1 * r1_infty)
# plt.plot(v0, dadv)
# plt.axhline(dbdv)

r01 = 2
r02 = 3
mutation_rate_per_gamma1 = 1
alpha = np.linspace(1, 10, 91)

r1_infty = final_recovered(r01, s0)
r2_infty = final_recovered(r02, s0)

for a in alpha:
    if a > -r1_infty / (r2_infty * (np.exp(-mutation_rate_per_gamma1 * r1_infty)
                                   * (1 + mutation_rate_per_gamma1 * r1_infty)
                                   - 1)):
        utility = vacc2_vs_vacc1(.001, mutation_rate_per_gamma1, r01, r02, a)
        print(f'{a}: {utility}')
        v_star = brentq(vacc2_vs_vacc1, 1e-8, 1, args=
                        (mutation_rate_per_gamma1, r01, r02, a))
        print(v_star)
        max_utility = -minimize(vacc2_vs_vacc1, .001, args=
                               (mutation_rate_per_gamma1, r01, r02, a)).fun
        print(max_utility)
    else:
        dadv = (-r1_infty - a * mutation_rate_per_gamma1 * r1_infty * r2_infty 
                * np.exp(-mutation_rate_per_gamma1 * r1_infty))
        dbdv = -a * r2_infty + a * r2_infty * np.exp(-mutation_rate_per_gamma1
                                             * r1_infty)
        print(f'{a}: {dadv}-{dbdv}')
        v_star = 0
        print(v_star)
        
        
final_size_ratios = np.linspace(0.3, 1, 101)
expected_mutations = np.linspace(1, 10, 101)

threshold_alphas = np.zeros((len(final_size_ratios), len(expected_mutations)))

for i, fsr in enumerate(final_size_ratios):
    for j, em in enumerate(expected_mutations):
        threshold_alphas[i, j] = threshold_alpha(fsr, em)
        
c = plt.pcolormesh(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r_1}(\infty)}{\bar{r_2}(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r_1}(\infty)}{\bar{r_2}(\infty)}$ and $\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r_1}(\infty)}{\bar{r_2}(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r_1}(\infty)}{\bar{r_2}(\infty)}$ and $\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()