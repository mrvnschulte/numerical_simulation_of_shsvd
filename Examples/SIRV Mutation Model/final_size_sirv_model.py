# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:03:26 2025

@author: schulte
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def sirv_model(t, x, beta=.2, gamma=.1, alpha=0):
    
    s, i, r, v = x
    ds = -beta * s * i - alpha * s
    di = beta * s * i - gamma * i
    dr = gamma * i
    dv = alpha * s
    
    dx = [ds, di, dr, dv]
        
    return dx

def final_size_sirv_model(beta, gamma, alpha, x_initial):
    
    res = solve_ivp(sirv_model, (0, 500), x_initial, args=(beta, gamma, alpha))
    r_infty = res.y[2][-1]

    return r_infty

def expected_severe_cases_vacc1(alpha, r01, r02, severity_ratio,
                                mutation_rate_per_gamma, gamma1=.1, gamma2=.1):
    
    beta1 = r01 * gamma1
    beta2 = r02 * gamma2
    
    r1_infty = final_size_sirv_model(beta1, gamma1, alpha, x_initial)
    r2_infty = final_size_sirv_model(beta2, gamma2, 0, x_initial)
    
    esc = r1_infty + severity_ratio * (1 - np.exp(-mutation_rate_per_gamma
                                                  * r1_infty)) * r2_infty
    
    return esc

def expected_severe_cases_vacc2(alpha, r01, r02, severity_ratio,
                                mutation_rate_per_gamma, gamma1=.1, gamma2=.1):
    
    beta1 = r01 * gamma1
    beta2 = r02 * gamma2
    
    r1_infty = final_size_sirv_model(beta1, gamma1, 0, x_initial)
    r2_infty = final_size_sirv_model(beta2, gamma2, alpha, x_initial)
    
    esc = r1_infty + severity_ratio * (1 - np.exp(-mutation_rate_per_gamma
                                                  * r1_infty)) * r2_infty
    
    return esc

def vacc2_vs_vacc1(alpha, r01, r02, severity_ratio, 
                   mutation_rate_per_gamma, gamma1=.1, gamma2=.1):
    esc_vacc1 = expected_severe_cases_vacc1(alpha, r01, r02, severity_ratio,
                                            mutation_rate_per_gamma, gamma1, 
                                            gamma2)
    esc_vacc2 = expected_severe_cases_vacc2(alpha, r01, r02, severity_ratio,
                                            mutation_rate_per_gamma, gamma1, 
                                            gamma2)
    utility = (esc_vacc1 - esc_vacc2) / esc_vacc1
    
    return utility


dpi = 500

x_initial = [.99, .01, 0, 0]
t0 = 0
tf = 500
beta = .2
gamma = .1
alpha = gamma / 2.5

t_eval = np.linspace(t0, tf, tf - t0 + 1)

res = solve_ivp(sirv_model, (t0, tf), x_initial, t_eval=t_eval,
                args=(beta, gamma, alpha))

plt.plot(res.t, res.y.T, label=['Susceptibles', 'Infectious', 'Recovered',
                                'Vaccinated'])
plt.legend()
plt.title('Dynamics of an SIRV model')
plt.xlabel('Time')
plt.ylabel('Population fraction')
plt.show()
'''
r0 = np.linspace(.5, 5, 101)
alpha_per_gamma = np.linspace(0, .5, 51)

alphas = alpha_per_gamma * gamma
betas = r0 * gamma
r_infty = np.zeros((len(betas), len(alphas)))


for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        r_infty[i, j] = final_size_sirv_model(b, gamma, a, x_initial)
        
c = plt.pcolormesh(r0, alpha_per_gamma, r_infty.T)
plt.colorbar(c, label=r'$r(\infty)$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{\alpha}{\gamma}$')
plt.title(r'Final size $r(\infty)$ in relation to $R_0$ and $\alpha$')
plt.tight_layout()
#plt.savefig('Mutation_probability_mesh.jpg', dpi=dpi)
plt.show()

c = plt.contourf(r0, alpha_per_gamma, r_infty.T)
plt.colorbar(c, label=r'$r(\infty)$')
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\frac{\alpha}{\gamma}$')
plt.title(r'Final size $r(\infty)$ in relation to $R_0$ and $\alpha$')
plt.tight_layout()
#plt.savefig('Mutation_probability_mesh.jpg', dpi=dpi)
plt.show()
'''
alphas = np.linspace(0, 1, 101)
utilities = np.zeros((len(alphas)))
r0 = 8

for i, a in enumerate(alphas):
    utilities[i] = vacc2_vs_vacc1(a, r0, 1/2*r0, 5, 2)

plt.plot(alphas, utilities)