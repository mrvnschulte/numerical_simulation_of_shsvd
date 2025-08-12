# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:28:28 2025

@author: schulte
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize


def final_recovered_equation(x, r0, s0):
    return np.log(s0/(1 - x)) - r0 * x

def final_recovered(r0, s0, final_r_max=0.9999999999999):
    try:
        final_r = brentq(final_recovered_equation, 0, final_r_max,
                         args=(r0, s0))
    except:
        final_r = final_r_max
    return final_r

def mutation_probability(mutation_rate_per_gamma, r_infty):
    return 1 - np.exp(-mutation_rate_per_gamma * r_infty)

def expected_final_size_vacc1(v0, final_size_ratio, expected_mutations, 
                              alpha=1, r2_infty=.9):
    e = (final_size_ratio * r2_infty * (1 - v0)
         + alpha * (1 - np.exp(-expected_mutations * (1 - v0))) * r2_infty)
    return e

def expected_final_size_vacc2(v0, final_size_ratio, expected_mutations,
                              alpha=1, r2_infty=.9):
    e = (final_size_ratio * r2_infty
         + alpha * (1 - np.exp(-expected_mutations)) * r2_infty * (1 - v0))
    return e

def vacc2_vs_vacc1(v0, final_size_ratio, expected_mutations, alpha=1,
                   r2_infty=.9):
    efs_vacc1 = expected_final_size_vacc1(v0, final_size_ratio,
                                          expected_mutations, alpha, r2_infty)
    efs_vacc2 = expected_final_size_vacc2(v0, final_size_ratio,
                                          expected_mutations, alpha, r2_infty)
    utility = efs_vacc2 - efs_vacc1
    
    return utility

def threshold_alpha(final_size_ratio, expected_mutations):
    return -final_size_ratio / (np.exp(-expected_mutations) *
                                (1 + expected_mutations) - 1)

def threshold_vaccination(final_size_ratio, expected_mutations, alpha=1):
    if vacc2_vs_vacc1(1e-8, final_size_ratio, expected_mutations, alpha)<0:
        return brentq(vacc2_vs_vacc1, 1e-8, 1, args=(final_size_ratio,
                                                 expected_mutations, alpha))
    else:
        return 0

def max_utility(final_size_ratio, expected_mutations, alpha=1, r2_infty=.9):
    res = minimize(vacc2_vs_vacc1, .001, bounds=[(0,1)], args=
                     (final_size_ratio, expected_mutations, alpha, r2_infty)
                     )
    value = -res.fun / r2_infty
    v = res.x[0]
    return value, v

r0 = np.linspace(0.5, 5, 101)
s0 = .99
r_infty = np.zeros(len(r0))

for i, r in enumerate(r0):
    r_infty[i] = final_recovered(r, s0)
    
plt.plot(r0, r_infty)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$r_\infty$')
plt.title(rf'Final size of an epidemic in relation to $R_0$ with $s_0$={s0}')
plt.show()

mutation_rate_per_gamma = np.linspace(.1, 5, 101)

probability_of_mutation = np.zeros((len(r0), len(mutation_rate_per_gamma)))

for i, r in enumerate(r0):
    for j, m in enumerate(mutation_rate_per_gamma):
        probability_of_mutation[i, j] = mutation_probability(m, r_infty[i])


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

# r01 = 2
# r02 = 3
# mutation_rate_per_gamma1 = 1
# alpha = np.linspace(1, 10, 91)

# r1_infty = final_recovered(r01, s0)
# r2_infty = final_recovered(r02, s0)

# for a in alpha:
#     if a > -r1_infty / (r2_infty * (np.exp(-mutation_rate_per_gamma1 * r1_infty)
#                                    * (1 + mutation_rate_per_gamma1 * r1_infty)
#                                    - 1)):
#         utility = vacc2_vs_vacc1(.001, mutation_rate_per_gamma1, r01, r02, a)
#         print(f'{a}: {utility}')
#         v_star = brentq(vacc2_vs_vacc1, 1e-8, 1, args=
#                         (mutation_rate_per_gamma1, r01, r02, a))
#         print(v_star)
#         max_utility = -minimize(vacc2_vs_vacc1, .001, args=
#                                (mutation_rate_per_gamma1, r01, r02, a)).fun
#         print(max_utility)
#     else:
#         dadv = (-r1_infty - a * mutation_rate_per_gamma1 * r1_infty * r2_infty 
#                 * np.exp(-mutation_rate_per_gamma1 * r1_infty))
#         dbdv = -a * r2_infty + a * r2_infty * np.exp(-mutation_rate_per_gamma1
#                                              * r1_infty)
#         print(f'{a}: {dadv}-{dbdv}')
#         v_star = 0
#         print(v_star)
        
        
final_size_ratios = np.linspace(0.3, 1, 101)
expected_mutations = np.linspace(1, 10, 101)

threshold_alphas = np.zeros((len(final_size_ratios), len(expected_mutations)))

for i, fsr in enumerate(final_size_ratios):
    for j, em in enumerate(expected_mutations):
        threshold_alphas[i, j] = threshold_alpha(fsr, em)
        
c = plt.pcolormesh(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$ and $\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r_1}(\infty)}{\bar{r_2}(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$ and $\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

alpha = [.5, 1, 2, 4, 8]

for a in alpha:
    threshold_vaccinations = np.zeros((len(final_size_ratios),
                                      len(expected_mutations)))
    max_utilities = np.zeros((len(final_size_ratios),
                              len(expected_mutations)))
    max_utility_v = np.zeros((len(final_size_ratios),
                              len(expected_mutations)))
    
    for i, fsr in enumerate(final_size_ratios):
        for j, em in enumerate(expected_mutations):
            threshold_vaccinations[i, j] = threshold_vaccination(fsr, em)
            value, v = max_utility(fsr, em)
            max_utilities[i, j] = value
            max_utility_v[i, j] = v
            
            
    c = plt.pcolormesh(final_size_ratios, expected_mutations,
                       threshold_vaccinations.T)
    plt.colorbar(c, label=r'$v^*$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Threshold vaccinations $v^*$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()
    
    c = plt.contourf(final_size_ratios, expected_mutations, 
                     threshold_vaccinations.T)
    plt.colorbar(c, label=r'$v^*$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Threshold vaccinations $v^*$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()
    
    c = plt.pcolormesh(final_size_ratios, expected_mutations, max_utilities.T)
    plt.colorbar(c, label=r'$\max A-B$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Max utilities $A-B$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()
    
    c = plt.contourf(final_size_ratios, expected_mutations, max_utilities.T)
    plt.colorbar(c, label=r'$\max A-B$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Max utilities $A-B$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()
    
    c = plt.pcolormesh(final_size_ratios, expected_mutations, max_utility_v.T)
    plt.colorbar(c, label=r'$\max A-B$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Max utility vaccination $v$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()
    
    c = plt.contourf(final_size_ratios, expected_mutations, max_utility_v.T)
    plt.colorbar(c, label=r'$\max A-B$')
    plt.xlabel(r'$\frac{\bar{r}_1(\infty)}{\bar{r}_2(\infty)}$')
    plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
    plt.title(rf'Max utility vaccination $v$ in relation to $\frac{{\bar{{r}}_1(\infty)}}{{\bar{{r}}_2(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$ ($\alpha$={a})')
    plt.tight_layout()
    #plt.savefig('EMPW_Meshplot.png')
    plt.show()