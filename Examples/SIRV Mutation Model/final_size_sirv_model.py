# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:03:26 2025

@author: schulte
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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
                                mutation_rate_per_gamma, gamma1=.1, gamma2=.1,
                                x_initial=[.99, .01, 0, 0]):
    
    beta1 = r01 * gamma1
    beta2 = r02 * gamma2
    
    r1_infty = final_size_sirv_model(beta1, gamma1, alpha, x_initial)
    r2_infty = final_size_sirv_model(beta2, gamma2, 0, x_initial)
    
    esc = r1_infty + severity_ratio * (1 - np.exp(-mutation_rate_per_gamma
                                                  * r1_infty)) * r2_infty
    
    return esc

def expected_severe_cases_vacc2(alpha, r01, r02, severity_ratio,
                                mutation_rate_per_gamma, gamma1=.1, gamma2=.1,
                                x_initial=[.99, .01, 0, 0]):
    
    beta1 = r01 * gamma1
    beta2 = r02 * gamma2
    
    r1_infty = final_size_sirv_model(beta1, gamma1, 0, x_initial)
    r2_infty = final_size_sirv_model(beta2, gamma2, alpha, x_initial)
    
    esc = r1_infty + severity_ratio * (1 - np.exp(-mutation_rate_per_gamma
                                                  * r1_infty)) * r2_infty
    
    return esc

def vacc2_vs_vacc1(alpha, r01, r02, severity_ratio, 
                   mutation_rate_per_gamma, gamma1=.1, gamma2=.1,
                   x_initial=[.99, .01, 0, 0]):
    esc_vacc1 = expected_severe_cases_vacc1(alpha, r01, r02, severity_ratio,
                                            mutation_rate_per_gamma, gamma1, 
                                            gamma2, x_initial)
    esc_vacc2 = expected_severe_cases_vacc2(alpha, r01, r02, severity_ratio,
                                            mutation_rate_per_gamma, gamma1, 
                                            gamma2, x_initial)
    utility = (esc_vacc1 - esc_vacc2)
    
    return utility

def find_threshold_rate(r01, r02, severity_ratio, mutation_rate_per_gamma,
                        gamma1=.1, gamma2=.1, x_initial=[.99, .01, 0, 0]):
    
    if vacc2_vs_vacc1(.0001, r01, r02, severity_ratio, 
                       mutation_rate_per_gamma, gamma1, gamma2, x_initial)>0:
        if vacc2_vs_vacc1(1, r01, r02, severity_ratio, 
                           mutation_rate_per_gamma, gamma1, gamma2,
                           x_initial)<0:
            threshold = brentq(vacc2_vs_vacc1, .0001, 1,
                               args=(r01, r02, severity_ratio, 
                                     mutation_rate_per_gamma, gamma1, gamma2, 
                                     x_initial))
        else:
            threshold = 1
    else:
        if vacc2_vs_vacc1(1, r01, r02, severity_ratio, 
                           mutation_rate_per_gamma, gamma1, gamma2,
                           x_initial)>0:
            threshold = -brentq(vacc2_vs_vacc1, .0001, 1,
                               args=(r01, r02, severity_ratio, 
                                     mutation_rate_per_gamma, gamma1, gamma2, 
                                     x_initial))
        threshold = 0
    
    return threshold


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
plt.grid()
plt.tight_layout()
plt.savefig('Dynamics_SIRV_model.jpg', dpi=dpi)
plt.show()

r0 = np.linspace(.5, 5, 101)
alpha_per_gamma = np.linspace(0, .5, 51)

alphas = alpha_per_gamma * gamma
betas = r0 * gamma
r_infty = np.zeros((len(betas), len(alphas)))


# for i, b in enumerate(betas):
#     for j, a in enumerate(alphas):
#         r_infty[i, j] = final_size_sirv_model(b, gamma, a, x_initial)
        
# c = plt.pcolormesh(r0, alpha_per_gamma, r_infty.T)
# plt.colorbar(c, label=r'$r(\infty)$')
# plt.xlabel(r'$R_0$')
# plt.ylabel(r'$\frac{\alpha}{\gamma}$')
# plt.title(r'Final size $r(\infty)$ in relation to $R_0$ and $\alpha$')
# plt.tight_layout()
# plt.savefig('Final_size_mesh.jpg', dpi=dpi)
# plt.show()

# c = plt.contourf(r0, alpha_per_gamma, r_infty.T)
# plt.colorbar(c, label=r'$r(\infty)$')
# plt.xlabel(r'$R_0$')
# plt.ylabel(r'$\frac{\alpha}{\gamma}$')
# plt.title(r'Final size $r(\infty)$ in relation to $R_0$ and $\alpha$')
# plt.tight_layout()
# plt.savefig('Final_size_contour.jpg', dpi=dpi)
# plt.show()


r0 = [2, 4, 8]
severity_ratio = np.linspace(1, 5, 51)
mutation_rate_per_gamma = np.linspace(.1, 5, 51)

# for r in r0:
#     threshold_alpha = np.zeros((len(severity_ratio), len(mutation_rate_per_gamma)))
    
#     for i, s in enumerate(severity_ratio):
#         for j, m in enumerate(mutation_rate_per_gamma):
#             threshold_alpha[i, j] = find_threshold_rate(r, r, s, m, 
#                                                         x_initial=x_initial)
            
#     c = plt.pcolormesh(severity_ratio, mutation_rate_per_gamma, threshold_alpha.T)
#     plt.colorbar(c, label=r'$\alpha^*$')
#     plt.xlabel(r'$\frac{\varphi_2}{\varphi_1}$')
#     plt.ylabel(r'$\frac{m}{\gamma}$')
#     plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $\frac{{\varphi_2}}{{\varphi_1}}$ and $\frac{{m}}{{\gamma}}$ ($R_0={r}$)')
#     plt.tight_layout()
#     plt.savefig(f'Threshold_vaccination_rate_R0={r}_mesh.jpg', dpi=dpi)
#     plt.show()
    
#     c = plt.contourf(severity_ratio, mutation_rate_per_gamma, threshold_alpha.T)
#     plt.colorbar(c, label=r'$\alpha^*$')
#     plt.xlabel(r'$\frac{\varphi_2}{\varphi_1}$')
#     plt.ylabel(r'$\frac{m}{\gamma}$')
#     plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $\frac{{\varphi_2}}{{\varphi_1}}$ and $\frac{{m}}{{\gamma}}$ ($R_0={r}$)')
#     plt.tight_layout()
#     plt.savefig(f'Threshold_vaccination_rate_R0={r}_contour.jpg', dpi=dpi)
#     plt.show()

severity_ratio = [1.5, 3, 4.5]
r0 = np.linspace(1, 10, 51)

# for s in severity_ratio:
#     threshold_alpha = np.zeros((len(r0), len(mutation_rate_per_gamma)))

#     for i, r in enumerate(r0):
#         for j, m in enumerate(mutation_rate_per_gamma):
#             threshold_alpha[i, j] = find_threshold_rate(r, r, s, m, 
#                                                         x_initial=x_initial)
            
#     c = plt.pcolormesh(r0, mutation_rate_per_gamma, threshold_alpha.T)
#     plt.colorbar(c, label=r'$\alpha^*$')
#     plt.xlabel(r'$R_0$')
#     plt.ylabel(r'$\frac{m}{\gamma}$')
#     plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $R_0$ and $\frac{{m}}{{\gamma}}$ ($\frac{{\varphi_2}}{{\varphi_1}}={s}$)')
#     plt.tight_layout()
#     plt.savefig(f'Threshold_vaccination_rate_s={s}_mesh.jpg', dpi=dpi)
#     plt.show()
    
#     c = plt.contourf(r0, mutation_rate_per_gamma, threshold_alpha.T)
#     plt.colorbar(c, label=r'$\alpha^*$')
#     plt.xlabel(r'$R_0$')
#     plt.ylabel(r'$\frac{m}{\gamma}$')
#     plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $R_0$ and $\frac{{m}}{{\gamma}}$ ($\frac{{\varphi_2}}{{\varphi_1}}={s}$)')
#     plt.tight_layout()
#     plt.savefig(f'Threshold_vaccination_rate_s={s}_contour.jpg', dpi=dpi)
#     plt.show()
    
    
mutation_rate_per_gamma = [1, 3, 5]
severity_ratio = np.linspace(1, 5, 51)

for m in mutation_rate_per_gamma:
    threshold_alpha = np.zeros((len(r0), len(severity_ratio)))

    for i, r in enumerate(r0):
        for j, s in enumerate(severity_ratio):
            threshold_alpha[i, j] = find_threshold_rate(r, r, s, m, 
                                                        x_initial=x_initial)
            
    c = plt.pcolormesh(r0, severity_ratio, threshold_alpha.T)
    plt.colorbar(c, label=r'$\alpha^*$')
    plt.xlabel(r'$R_0$')
    plt.ylabel(r'$\frac{\varphi_2}{\varphi_1}$')
    plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $R_0$ and $\frac{{\varphi_2}}{{\varphi_1}}$ ($\frac{{m}}{{\gamma}}={m}$)')
    plt.tight_layout()
    plt.savefig(f'Threshold_vaccination_rate_m={m}_mesh.jpg', dpi=dpi)
    plt.show()
    
    c = plt.contourf(r0, severity_ratio, threshold_alpha.T)
    plt.colorbar(c, label=r'$\alpha^*$')
    plt.xlabel(r'$R_0$')
    plt.ylabel(r'$\frac{\varphi_2}{\varphi_1}$')
    plt.title(rf'Threshold vaccination rate $\alpha^*$ in relation to $R_0$ and $\frac{{\varphi_2}}{{\varphi_1}}$ ($\frac{{m}}{{\gamma}}={m}$)')
    plt.tight_layout()
    plt.savefig(f'Threshold_vaccination_rate_m={m}_contour.jpg', dpi=dpi)
    plt.show()


x_initial_new = [.999, .001, 0, 0]
alphas = np.linspace(0, 1, 1001)
utilities = np.zeros((len(alphas)))
r0 = 2

for i, a in enumerate(alphas):
    utilities[i] = vacc2_vs_vacc1(a, r0, 3*r0, 5, 100, x_initial=x_initial_new)

plt.plot(alphas, utilities)
plt.xlabel(r'Vaccination rate $\alpha$')
plt.ylabel(r'Utility $g(\alpha)$')
plt.grid()
plt.title('Utility of vaccinating against strain 1')
plt.savefig('Utility_of_vaccination.jpg', dpi=dpi)
plt.show()