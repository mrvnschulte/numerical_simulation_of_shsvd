# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:42:05 2025

@author: schulte
"""
import numpy as np
from numerical_simulation_of_shsvd import solve_hsvd, solve_shsvd_single_run
from cross_immunity_model import (
    multi_variant_model_with_total_infections,
    new_variant_transition_function_with_total_infections,
    mutation_probability_function,
    multi_variant_model_with_demographics,
    new_variant_transition_function,
    new_variant_transition_function_with_demographics,
    multi_variant_model_with_matrices,
    new_variant_transition_function_with_matrices,
    new_variant_transition_function_stochastic,
    mutation_probability_function_multiple_mutations,
    extinction_function,
    extinction_transition_function,
)

t0 = 0
tf = 300
x0 = np.array([0.999, 0, 0.001, 0.001])
beta = [0.15]
gamma = [0.1]
sigma = [[1]]
beta_new = 0.2
gamma_new = 0.1
t_eval = np.arange(tf + 1)

sigma_new = [0.5]

mutation_rate = 0.1

func = solve_hsvd(
    multi_variant_model_with_total_infections,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[100],
    jump_time_transition_function=new_variant_transition_function_with_total_infections,
    flow_args=(beta, gamma, sigma, 1),
    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
)
func.plot_coordinates(
    labels=[
        "s",
        "r1",
        "i1",
        "r2",
        "i2",
        "r12",
        "i1,12",
        "i2,12",
        "i12",
        "n1",
        "n2",
    ]
)

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
n = 1
x0 = np.array([0.999, 0, 0.001, 0.001])


func = solve_hsvd(
    multi_variant_model_with_matrices,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[100],
    jump_time_transition_function=new_variant_transition_function_with_matrices,
    flow_args=(beta, gamma, sigma, V, G, T, S, n),
    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
)
func.plot_coordinates(
    labels=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12", "n"],
    title="Compartment plot",
    ylabel="Fraction of the population",
)

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
n = 1
x0 = np.array([0.999, 0, 0.001, 0.001])

alpha = 10
beta_dist = 2
k = 0.05


func = solve_hsvd(
    multi_variant_model_with_matrices,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[100],
    jump_time_transition_function=new_variant_transition_function_stochastic,
    flow_args=(beta, gamma, sigma, V, G, T, S, n),
    jump_time_transition_args=(alpha, beta_dist, k),
    jump_set_args=(1,),
)
func.plot_coordinates(
    labels=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12", "n"]
)

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
n = 1
x0 = np.array([0.999, 0, 0.001, 0.001])

beta_new = 0.2
gamma_new = 0.1
sigma_new = [0.5]


func = solve_hsvd(
    multi_variant_model_with_matrices,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[100, 200],
    jump_time_transition_function=new_variant_transition_function_with_matrices,
    flow_args=(beta, gamma, sigma, V, G, T, S, n),
    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
)
func.plot_coordinates(
    labels=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12", "n"]
)

V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
n = 1
x0 = np.array([0.999, 0, 0.001, 0.001])

alpha = 10
beta_dist = 2
k = 0.1

func = solve_hsvd(
    multi_variant_model_with_matrices,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[100, 200],
    jump_time_transition_function=new_variant_transition_function_stochastic,
    jump_set_function=extinction_function,
    jump_set_transition_function=extinction_transition_function,
    flow_args=(beta, gamma, sigma, V, G, T, S, n),
    jump_time_transition_args=(alpha, beta_dist, k),
    jump_set_args=(0.0001, V, n),
)
func.plot_coordinates(
    labels=[
        "s",
        "r1",
        "i1",
        "r2",
        "i2",
        "r12",
        "i1,12",
        "i2,12",
        "i12",
        "n",
    ]
)


V = np.array([[0, 1]])
G = np.array([[0, gamma[0]], [0, 0]])
T = np.array([[[0, 1], [0, 0]]])
S = np.array([[[0, 0], [1, 0]]])
n = 1
x0 = np.array([0.999, 0, 0.001, 0.001])

alpha = 5
beta_dist = 2
k = 0.05
mutation_rate = 0.2

func = solve_shsvd_single_run(
    multi_variant_model_with_matrices,
    mutation_probability_function_multiple_mutations,
    new_variant_transition_function_stochastic,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_set_function=extinction_function,
    jump_set_transition_function=extinction_transition_function,
    flow_args=(beta, gamma, sigma, V, G, T, S, n),
    transition_args=(alpha, beta_dist, k),
    jump_rate_args=(mutation_rate,),
    jump_set_args=(0.0001, V, n),
)
func.plot_coordinates(
    labels=[
        "s",
        "r1",
        "i1",
        "r2",
        "i2",
        "r12",
        "i1,12",
        "i2,12",
        "i12",
        "n",
    ]
)


x0 = np.array([0.999, 0, 0.001, 0.001])

func = solve_shsvd_single_run(
    multi_variant_model_with_total_infections,
    mutation_probability_function,
    new_variant_transition_function_with_total_infections,
    x0,
    (t0, tf),
    jump_rate_bound=mutation_rate,
    t_eval=t_eval,
    flow_args=(beta, gamma, sigma, 1),
    transition_args=(beta_new, gamma_new, sigma_new),
    jump_rate_args=(mutation_rate,),
)
func.plot_coordinates(
    labels=[
        "s",
        "r1",
        "i1",
        "r2",
        "i2",
        "r12",
        "i1,12",
        "i2,12",
        "i12",
        "n1",
        "n2",
    ]
)

x0 = np.array([0.999, 0, 0.001])
b = 0.01
tf = 2000
t_eval = np.arange(tf + 1)
beta_new = 0.16
sigma_new = [0.8]

func = solve_hsvd(
    multi_variant_model_with_demographics,
    x0,
    (t0, tf),
    t_eval=t_eval,
    jump_times=[1000],
    jump_time_transition_function=new_variant_transition_function_with_demographics,
    flow_args=(beta, gamma, sigma, b),
    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
)
func.plot_coordinates(
    labels=[
        "s",
        "r1",
        "i1",
        "r2",
        "i2",
        "r12",
        "i1,12",
        "i2,12",
        "i12",
        "n1",
        "n2",
    ]
)
