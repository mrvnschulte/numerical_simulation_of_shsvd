# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:26:49 2025

@author: schulte
"""

import matplotlib.pyplot as plt
import numpy as np

"""
This shows the stable and unstable regions of the deterministic harmonic
oscillator which is held and pushed, depending on the parameters for the flow
time T and the push constant c.
"""


def cot(t):
    """
    This function computes the cotangent of t.

    Parameters
    ----------
    t : float, or np.ndarray
        Time.

    Returns
    -------
    y : float, or np.ndarray
        Cotangent of t.
    """
    y = -np.tan(t + np.pi / 2)
    return y


# Plot settings
plot_legend_outside = False
lower_limit = -20
upper_limit = 20

# Values for flow time T
t = np.linspace(0.01, 3, 601)
t *= np.pi

# Compute upper and lower limit of stable region
upper_c = -cot(t) + np.sqrt(cot(t) ** 2 + 1)
lower_c = -cot(t) - np.sqrt(cot(t) ** 2 + 1)

# Set ticks of plot
ticks = int(np.round(t[-1] / np.pi * 2)) + 1
xticks = [n * np.pi / 2 for n in range(ticks)]
x_tick_labels = [
    f"{int(n/2)}π" if n % 2 == 0 else f"{n}π/2" for n in range(ticks)
]
x_tick_labels[0] = "0"
x_tick_labels[1] = "π/2"
x_tick_labels[2] = "π"

# Plot stability region
plt.plot(t, upper_c, "red", label="Stable")
plt.plot(t, lower_c, "red")
plt.fill_between(
    t, lower_c, upper_c, color="blue", alpha=0.2, label="Asymptotically stable"
)
plt.fill_between(
    t, lower_limit, lower_c, color="red", alpha=0.2, label="Instable"
)
plt.fill_between(t, upper_c, upper_limit, color="red", alpha=0.2)
if plot_legend_outside:
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
else:
    plt.legend()
plt.title("Stability region of hold and push harmonic oscillator")
plt.xlabel("Flow time T")
plt.ylabel("Push constant c")
plt.xticks(xticks, x_tick_labels)
plt.ylim(lower_limit, upper_limit)
plt.xlim(0, t[-1])
plt.tight_layout()
plt.savefig("Stability_region.png")
