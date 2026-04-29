# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:35:12 2026

@author: schulte
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numerical_simulation_of_shsvd import (
    solve_hsvd,
    solve_shsvd,
    solve_shsvd_single_run,
)
from cross_immunity_model import (
    multi_variant_model_with_matrices,
    new_variant_transition_function_stochastic,
    mutation_probability_function,
    mutation_probability_function_multiple_mutations,
    extinction_function,
    extinction_transition_function,
    total_infections_new,
    total_infectious,
)
import time

start = time.time()


file_name = "expected_infections_multi_mutation.npy"
dpi = 500
iterations = 2

t0 = 0
tf = 1000
t_eval = np.arange(tf + 1)
x0 = np.array([0.999, 0, 0.001, 0.001])
betas = [0.15, 0.2, 0.25]
gamma = [0.09]
sigma = [[1]]

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
b = 0
variants = 1

alpha = 0.0001
beta_dist = 2
ks = [0.025, 0.05, 0.075]
mutation_rate = 1
extinction_threshold = 0.0001

r0 = np.round(np.array(betas) / (gamma[0] + b), 1)

mean_infectious = []
variance_infectious = []
beta = [np.round(np.mean(betas), 2)]
k = np.mean(ks)

stats = solve_shsvd(
    multi_variant_model_with_matrices,
    mutation_probability_function,
    new_variant_transition_function_stochastic,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_set_function=extinction_function,
    jump_set_transition_function=extinction_transition_function,
    flow_args=(beta, gamma, sigma, V, G, T, S, variants, b),
    transition_args=(alpha, beta_dist, k),
    jump_rate_args=(mutation_rate,),
    jump_set_args=(extinction_threshold, V, variants),
    custom_evaluation_functions=[total_infectious],
    iterations=iterations,
)
