# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:07:54 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def two_variant_model(t, x, beta1, beta2, sigma, gamma1=.1, gamma2=.1):
    
    i1, i12, i2, i21, s, r1, r2, r = x
    
    i1_ges = i1 + i12
    i2_ges = i2 + i21
    
    ds = -beta1 * s * i1_ges - beta2 * s * i2_ges
    di1 = (1 - sigma) * beta1 * s * i1_ges - gamma1 * i1
    dr1 = gamma1 * i1 - beta2 * r1 * i2_ges
    di21 = sigma * beta2 * s * i2_ges + beta2 * r1 * i2_ges - gamma2 * i21
    di2 = (1 - sigma) * beta2 * s * i2_ges - gamma2 * i2
    dr2 = gamma2 * i2 - beta1 * r2 * i1_ges
    di12 = sigma * beta1 * s * i1_ges + beta1 * r2 * i1_ges - gamma1 * i12
    dr = gamma2 * i21 + gamma1 * i12
    
    dx = [di1, di12, di2, di21, ds, dr1, dr2, dr]
    
    return dx


t0 = 0
tf = 500

x0 = np.array([.001, 0, .001, 0, .998, 0, 0, 0])

res = solve_ivp(two_variant_model, (t0, tf), x0, args=(.12, .15, .5))

plt.plot(res.t, res.y.T)