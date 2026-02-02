# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:20:27 2025

@author: schulte
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Computes the invasion reproduction number and plots the invasion regions
"""


def invasion_reproduction_number(r01, r02, sigma):
    """
    Computes the invasion reproduction number of variant 1 invading variant 2
    equilibrium.

    Parameters
    ----------
    r01 : float
        Basic reproduction number of variant 1.
    r02 : float
        Basic reproduction number of variant 2.
    sigma : float
        Cross-immunity between the two variants.

    Returns
    -------
    invasion_r0 : float
        Invasion reproduction number of variant 1 invading the variant 2
        equilibrium.
    """
    invasion_r0 = r01 - sigma * (r01 / r02) * (r02 - 1)
    return invasion_r0


def critical_sigma(r01, r02):
    """
    Computes the critical sigma, under which variant 1 can invade variant 2.

    Parameters
    ----------
    r01 : float
        Basic reproduction number of variant 1.
    r02 : float, or np.ndarray
        Basic reproduction number of variant 2.

    Returns
    -------
    sigma : float, or np.ndarray
        Critical sigma, under which variant 1 can invade variant 2.
    """
    r02 = np.asarray(r02, dtype=float)
    sigma = np.ones_like(r02)
    mask = r02 > r01
    sigma[mask] = r02[mask] * (r01 - 1) / (r01 * (r02[mask] - 1))
    if sigma.ndim == 0:
        sigma = float(sigma)
    return sigma


def critical_r01(r02, sigma):
    """
    Computes the critical r01, over which variant 1 can invade variant 2.

    Parameters
    ----------
    r02 : float, or np.ndarray
        Basic reproduction number of variant 2.
    sigma : float
        Cross-immunity between the two variants.

    Returns
    -------
    r01 : float, or np.ndarray
        Critical basic reproduction number of variant 1, over which variant 1
        can invade variant 2.
    """
    r02 = np.asarray(r02, dtype=float)
    r01 = np.ones_like(r02)
    mask = r02 > 1
    r01[mask] = 1.0 / (1.0 - sigma / r02[mask] * (r02[mask] - 1.0))
    if r01.ndim == 0:
        r01 = float(r01)
    return r01


if __name__ == "__main__":
    # Set parameters
    r02 = np.linspace(1, 4, 201)
    sigmas = np.linspace(0, 1, 11)

    alpha = 0.4

    for sigma in sigmas:
        # Compute and plot invasion regions
        r01 = critical_r01(r02, sigma)
        if sigma < 1:
            plt.plot(r02, r01, "red")
            plt.plot(r01, r02, "red")
        plt.vlines(1, 0, 1, color="blue")
        plt.hlines(1, 0, 1, color="blue")
        plt.vlines(1, 1, r02[-1], color="purple")
        plt.hlines(1, 1, r02[-1], color="purple")
        plt.fill_between(
            r02,
            r02,
            r01,
            color="red",
            alpha=alpha,
            edgecolor="none",
            label="Both invasions possible",
        )
        plt.fill_betweenx(
            r02, r02, r01, color="red", alpha=alpha, edgecolor="none"
        )
        plt.fill_between(
            r02,
            r01,
            np.ones_like(r02),
            color="purple",
            alpha=alpha,
            edgecolor="none",
            label="One invasion possible",
        )
        plt.fill_betweenx(
            r02,
            r01,
            np.ones_like(r02),
            color="purple",
            alpha=alpha,
            edgecolor="none",
        )
        plt.fill_between(r02, 1, 0, color="blue", alpha=alpha)
        plt.fill_betweenx(r02, 1, 0, color="blue", alpha=alpha)
        plt.fill_between([0, 1], 0, 1, color="black", alpha=alpha)
        plt.xlim(0, r02[-1])
        plt.ylim(0, r02[-1])
        plt.title(rf"Invasion regions for $\sigma$={round(sigma, 1)}")
        plt.xlabel(r"$R_0^1$")
        plt.ylabel(r"$R_0^2$")
        plt.legend()
        plt.show()

        # Compute and plot invasion reproduction number
        invasion_r0 = np.zeros((len(r02), len(r02)))
        for i, r01 in enumerate(r02):
            invasion_r0[i] = invasion_reproduction_number(r01, r02, sigma)

        ir = plt.contourf(r02, r02, invasion_r0.T)
        plt.xlabel(r"$R_0^1$")
        plt.ylabel(r"$R_0^2$")
        plt.title(
            rf"Invasion reproduction number $R_0^{{1\to 2}}$ "
            rf"for $\sigma={round(sigma, 2)}$"
        )
        cbar = plt.colorbar(ir)
        cbar.set_label(r"$R_0^{1\to 2}$")
        plt.show()

    # Compute and plot critical sigmas
    critical_sigmas = np.zeros((len(r02), len(r02)))
    for i, r01 in enumerate(r02):
        critical_sigmas[i] = critical_sigma(r01, r02)

    cs = plt.contourf(r02, r02, critical_sigmas.T)
    plt.xlabel(r"$R_0^1$")
    plt.ylabel(r"$R_0^2$")
    plt.title("Critical Sigma")
    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\sigma^*$")
    plt.show()

    # Compute and plot critical R01
    sigmas = np.linspace(0, 1, 101)
    critical_r01s = np.zeros((len(sigmas), len(r02)))
    for i, sigma in enumerate(sigmas):
        critical_r01s[i] = critical_r01(r02, sigma)

    cr = plt.contourf(sigmas, r02, critical_r01s.T)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$R_0^2$")
    plt.title(r"Critical $R_0^1$")
    cbar = plt.colorbar(cr)
    cbar.set_label(r"$R_0^{1*}$")
    plt.show()

    cr = plt.contourf(sigmas, r02, (critical_r01s / r02).T)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$R_0^2$")
    plt.title(r"Critical $\frac{R_0^{1*}}{R_0^2}$")
    cbar = plt.colorbar(cr)
    cbar.set_label(r"$\frac{R_0^{1*}}{R_0^2}$")
    plt.show()
