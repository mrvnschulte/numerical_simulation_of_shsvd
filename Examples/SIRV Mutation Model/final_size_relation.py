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

def expected_final_size_vacc1(v0, severe_cases_ratio, expected_mutations,
                              r1_infty=.9):
    e = ((1 - v0) * r1_infty + (1 - np.exp(-expected_mutations* (1 - v0)))
         * severe_cases_ratio * r1_infty)
    return e

def expected_final_size_vacc2(v0, severe_cases_ratio, expected_mutations, 
                              r1_infty=.9):
    e = (r1_infty + (1 - np.exp(-expected_mutations)) * severe_cases_ratio 
         * r1_infty * (1 - v0))
    return e

def vacc2_vs_vacc1(v0, severe_cases_ratio, expected_mutations, r1_infty=.9):
    efs_vacc1 = expected_final_size_vacc1(v0, severe_cases_ratio,
                                          expected_mutations, r1_infty)
    efs_vacc2 = expected_final_size_vacc2(v0, severe_cases_ratio,
                                          expected_mutations, r1_infty)
    utility = efs_vacc2 - efs_vacc1
    
    return utility

def threshold_alpha(final_size_ratio, expected_mutations):
    return -final_size_ratio / (np.exp(-expected_mutations) *
                                (1 + expected_mutations) - 1)

def threshold_vaccination(severe_cases_ratio, expected_mutations):
    if vacc2_vs_vacc1(1e-8, severe_cases_ratio, expected_mutations)<0:
        return brentq(vacc2_vs_vacc1, 1e-8, 1, args=(severe_cases_ratio,
                                                 expected_mutations))
    else:
        return 0

def max_utility(severe_cases_ratio, expected_mutations, r1_infty=.9):
    res = minimize(vacc2_vs_vacc1, .001, bounds=[(0,1)], args=
                     (severe_cases_ratio, expected_mutations, r1_infty)
                     )
    value = -res.fun / r1_infty
    v = res.x[0]
    return value, v

def expected_severe_infections(v1, final_size_ratio, expected_mutations,
                               alpha, total_vaccinations):
    value = (final_size_ratio * (1 - v1)
             + alpha * (1 - np.exp(-expected_mutations * (1 - v1)))
             * (1 - total_vaccinations + v1))
    return value


dpi = 500

# Setting parameters and compute the final size of an epidemic with respect to
# these parameters
r0 = np.linspace(0.5, 5, 101)
s0 = .99
r_infty = np.zeros(len(r0))

for i, r in enumerate(r0):
    r_infty[i] = final_recovered(r, s0)
    
plt.plot(r0, r_infty)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$r(\infty)$')
plt.title(rf'Final size of an epidemic in relation to $R_0$ with $s(0)$={s0}')
plt.grid()
plt.savefig('Final_size_relation.jpg', dpi=dpi)
plt.show()


# Setting parameters and compute the mutation probability
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
plt.savefig('Mutation_probability_mesh.jpg', dpi=dpi)
plt.show()

c = plt.contourf(r0, mutation_rate_per_gamma, probability_of_mutation.T)
plt.colorbar(c, label=r'$P(M)$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{m}{\gamma}$')
plt.title(r'Mutation probability in relation to $R_0$ and $\frac{m}{\gamma}$')
plt.tight_layout()
plt.savefig('Mutation_probability_contour.jpg', dpi=dpi)
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
        
        
final_size_ratios = np.linspace(1, 5, 101)
expected_mutations = np.linspace(1, 5, 101)

threshold_alphas = np.zeros((len(final_size_ratios), len(expected_mutations)))

for i, fsr in enumerate(final_size_ratios):
    for j, em in enumerate(expected_mutations):
        threshold_alphas[i, j] = threshold_alpha(1/fsr, em)
        
c = plt.pcolormesh(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$ and $\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(final_size_ratios, expected_mutations, threshold_alphas.T)
plt.colorbar(c, label=r'$\alpha^*$')
plt.xlabel(r'$\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r_1}(\infty)$')
plt.title(r'Threshold severity ratio $\alpha^*$ in relation to $\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$ and $\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

threshold_severe_cases_ratio = threshold_alpha(1, expected_mutations)

plt.plot(expected_mutations, threshold_severe_cases_ratio)
plt.show()

severe_cases_ratios = np.linspace(1, 10, 101)

threshold_vaccinations = np.zeros((len(severe_cases_ratios),
                                  len(expected_mutations)))
max_utilities = np.zeros((len(severe_cases_ratios),
                          len(expected_mutations)))
max_utility_v = np.zeros((len(severe_cases_ratios),
                          len(expected_mutations)))

for i, scr in enumerate(severe_cases_ratios):
    for j, em in enumerate(expected_mutations):
        threshold_vaccinations[i, j] = threshold_vaccination(scr, em)
        value, v = max_utility(scr, em)
        max_utilities[i, j] = value
        max_utility_v[i, j] = v
        
        
c = plt.pcolormesh(severe_cases_ratios, expected_mutations,
                   threshold_vaccinations.T)
plt.colorbar(c, label=r'$v^*$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Threshold vaccinations $v^*$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(severe_cases_ratios, expected_mutations, 
                 threshold_vaccinations.T)
plt.colorbar(c, label=r'$v^*$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Threshold vaccinations $v^*$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.pcolormesh(final_size_ratios, expected_mutations, max_utilities.T)
plt.colorbar(c, label=r'$\max A-B$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Max utilities $A-B$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(final_size_ratios, expected_mutations, max_utilities.T)
plt.colorbar(c, label=r'$\max A-B$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Max utilities $A-B$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.pcolormesh(final_size_ratios, expected_mutations, max_utility_v.T)
plt.colorbar(c, label=r'$\max A-B$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Max utility vaccination $v$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()

c = plt.contourf(final_size_ratios, expected_mutations, max_utility_v.T)
plt.colorbar(c, label=r'$\max A-B$')
plt.xlabel(r'$\alpha\frac{\bar{r}_2(\infty)}{\bar{r}_1(\infty)}$')
plt.ylabel(r'$\frac{m}{\gamma}\bar{r}_1(\infty)$')
plt.title(r'Max utility vaccination $v$ in relation to $\alpha\frac{{\bar{{r}}_2(\infty)}}{{\bar{{r}}_1(\infty)}}$ and $\frac{{m}}{{\gamma}}\bar{{r}}_1(\infty)$')
plt.tight_layout()
#plt.savefig('EMPW_Meshplot.png')
plt.show()
    
    
final_size_ratio = .5
expected_mutations = 2
alpha = 2
total_vaccinations = .6

v1 = np.linspace(0, total_vaccinations, 101)
severe_infections = expected_severe_infections(v1, final_size_ratio,
                                               expected_mutations, alpha,
                                               total_vaccinations)

plt.plot(v1, severe_infections)
plt.xlabel('Vaccinations against variant 1')
plt.ylabel('Expected severe infections')
plt.title('Expected severe infections as function of vaccinations against variant 1')