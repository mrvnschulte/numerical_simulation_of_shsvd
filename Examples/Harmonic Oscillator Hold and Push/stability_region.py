# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:26:49 2025

@author: schulte
"""

import matplotlib.pyplot as plt
import numpy as np


def cot(t):
    
    y = -np.tan(t+np.pi/2)
    
    return y

plot_legend_outside = False

t = np.linspace(0.01, 3, 601)
t *= np.pi

upper_c = - cot(t) + np.sqrt(cot(t)**2 + 1)
lower_c = - cot(t) - np.sqrt(cot(t)**2 + 1)

lower_limit = -20
upper_limit = 20

ticks = int(np.round(t[-1] / np.pi * 2)) + 1
xticks = [n*np.pi/2 for n in range(ticks)]
x_tick_labels = [f'{int(n/2)}π' if n%2 == 0 else f'{n}π/2' for n in range(ticks)]
x_tick_labels[0] = '0'
x_tick_labels[1] = 'π/2'
x_tick_labels[2] = 'π'

plt.plot(t, upper_c, 'red', label='Stable')
plt.plot(t, lower_c, 'red')
plt.fill_between(t, lower_c, upper_c, color='blue', alpha=.2, label='Asymptotically stable')
plt.fill_between(t, lower_limit, lower_c, color='red', alpha=.2, label='Instable')
plt.fill_between(t, upper_c, upper_limit, color='red', alpha=.2)
if plot_legend_outside:
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.15))
else:
    plt.legend()
plt.title('Stability region of hold and push harmonic oscillator')
plt.xlabel('Flow time T')
plt.ylabel('Push constant c')
plt.xticks(xticks, x_tick_labels)
plt.ylim(lower_limit, upper_limit)
plt.xlim(0, t[-1])
plt.tight_layout()
plt.savefig('Stability_region.png')