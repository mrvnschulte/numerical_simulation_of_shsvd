# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:47:50 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def sir_model(t, x, beta, gamma):
    
    s, i, r = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i

    return [ds, di, dr]


t0 = 0
tf = 500
x0 = [.999, .001, 0]
beta = .15
gamma = .1
m = .2

t_span = (t0, tf)

t_eval = np.linspace(t0, tf, tf + 1)

res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))

i = res.y[1]
r = res.y[2]
mutation_time_pdf = (m * i * np.exp(-m * r / gamma)) / (1 - np.exp(-m * r[-1] / gamma))
mutation_time_cdf = np.cumsum((m * i * np.exp(-m * r / gamma)) / (1 - np.exp(-m * r[-1] / gamma)))
expected_mutation_time = np.cumsum(t_eval * (m * i * np.exp(-m * r / gamma)) / (1 - np.exp(-m * r[-1] / gamma)))
plt.plot(t_eval, i)
plt.axvline(40)
plt.show()
plt.plot(t_eval, expected_mutation_time)
plt.show()
plt.plot(t_eval, mutation_time_pdf)
plt.show()
plt.plot(t_eval, mutation_time_cdf)
plt.show()


def iterated_sum_poisson(max_iter, lambda_param, n):
    
    sum_poisson_random_variable = np.random.poisson(lambda_param, n)
    
    for i in range(max_iter-1):
        sum_poisson_random_variable += np.where(sum_poisson_random_variable > i,
                                                np.random.poisson(lambda_param, n),
                                                0)
        
    return sum_poisson_random_variable

lambda_param = 2  # Der Durchschnitt (λ)

# Anzahl der Zufallsvariablen, die generiert werden sollen
n = 100000

# Generiere Poisson-Zufallsvariablen
sum_poisson = iterated_sum_poisson(10, lambda_param, n)

# Optional: Histogramm der generierten Zufallsvariablen
plt.hist(sum_poisson, bins=np.arange(0, max(sum_poisson) + 1), alpha=0.75, color='blue', edgecolor='black', density=True)
plt.title('Histogramm der Poisson-Zufallsvariablen')
plt.xlabel('Wert')
plt.ylabel('Häufigkeit')
plt.grid()
plt.show()
