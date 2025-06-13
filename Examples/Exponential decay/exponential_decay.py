# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:31:06 2025

@author: schulte
"""

from solve_shsvd import solve_shsvd, solve_shsvd_single_run
import matplotlib.pyplot as plt

def exponential_decay(t, x):
    
    n = len(x)
    
    dx = []
    for i in range(n):
        dx.append(-x[i]/(i+1))
        
    return dx


def jump_rate_function(t, x):
    
    return 1

def transition_function(t, x):
    
    x = list(x)
    n = len(x)
    
    x.append(x[n-1])
    x[n-1] = 0
    
    return x

initial_value = [1]
t_span = (0, 5)
jump_rate_bound = 1

# sol_stats = solve_shsvd(exponential_decay, jump_rate_function, transition_function, initial_value, t_span, jump_rate_bound, stepsize=.01, norm_order=1)

# fig = sol_stats.plot_norm(order=1, title='Plot of mean distance to A', ylabel='Distance to A')
# plt.ylim(-.05, 1.05)
# plt.tight_layout()
# plt.savefig('Exponential_decay_norm.png')
# plt.show()

func = solve_shsvd_single_run(exponential_decay, jump_rate_function, transition_function, initial_value, t_span, jump_rate_bound)
fig = func.plot_coordinates(title='Plot of the coordinates of a sample path')
plt.figure(fig)
plt.tight_layout()
plt.savefig('Exponential_decay_sample_solution.png')