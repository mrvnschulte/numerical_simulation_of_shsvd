# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:03:51 2025

@author: schulte
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cross_immunity_model import two_variant_model, multi_variant_model

# Solve the two-variant with .001 infectious with each variant at the beginning
# and plot it
t0 = 0
tf = 500

x0 = np.array([0.001, 0, 0.001, 0, 0.998, 0, 0, 0])

res12 = solve_ivp(
    two_variant_model,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(0.15, 0.2, 0.8),
)

plt.plot(
    res12.t, res12.y.T, label=["i1", "i12", "i2", "i21", "s", "r1", "r2", "r"]
)
plt.legend()
plt.show()

# Solve the same system using multi-variant model
t0 = 0
tf = 500

x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])
beta = [0.15, 0.2]
gamma = [0.1, 0.1]
sigma = np.array([[1, 0.8], [0.8, 1]])

res12 = solve_ivp(
    multi_variant_model,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma),
)

plt.plot(
    res12.t,
    res12.y.T,
    label=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12"],
)
plt.legend()
plt.show()

# Solve a 1-variant SIR model using multi-variant model
x0 = np.array([0.999, 0, 0.001])
beta = [0.2]
gamma = [0.1]
sigma = np.array([[1]])

res1 = solve_ivp(
    multi_variant_model,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma),
)

plt.plot(res1.t, res1.y.T, label=["s", "r1", "i1"])
plt.legend()
plt.show()
