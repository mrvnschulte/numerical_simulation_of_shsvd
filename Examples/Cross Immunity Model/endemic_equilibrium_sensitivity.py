import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from invasion_reproduction_number import critical_sigma


def x_6(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_6.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_6 at two-strain endemic equilibrium.
    """
    return b / (beta_1 * i_1 + beta_2 * i_2 + b)


def x_1(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_1.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_1 at two-strain endemic equilibrium.
    """
    return (
        (1 - sigma)
        * beta_1
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
    ) / (gamma_1 + beta_2 * i_2 + b)


def x_4(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_4.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_4 at two-strain endemic equilibrium.
    """
    return (
        (1 - sigma)
        * beta_2
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
    ) / (gamma_2 + beta_1 * i_1 + b)


def x_7(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_7.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_7 at two-strain endemic equilibrium.
    """
    return (
        gamma_1 * x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (beta_2 * i_2 + b)


def x_8(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_8.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_8 at two-strain endemic equilibrium.
    """
    return (
        gamma_2 * x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (beta_1 * i_1 + b)


def x_3(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_3.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_3 at two-strain endemic equilibrium.
    """
    return (
        beta_1
        * x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + beta_2
        * x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
    ) / (gamma_1 + gamma_2 + b)


def x_2(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_2.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_2 at two-strain endemic equilibrium.
    """
    return (
        sigma
        * beta_1
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + beta_1
        * x_8(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_1
        + gamma_2 * x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (gamma_1 + b)


def x_5(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_5.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_5 at two-strain endemic equilibrium.
    """
    return (
        sigma
        * beta_2
        * x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
        + beta_2
        * x_7(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        * i_2
        + gamma_1 * x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / (gamma_2 + b)


def x_9(
    i_1,
    i_2,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Endemic equilibrium of compartment x_9.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    float
        Value of x_9 at two-strain endemic equilibrium.
    """
    return (
        gamma_1 * x_2(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + gamma_2 * x_5(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    ) / b


def equation(
    i,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Defines the characteristic equation of the endemic equilibrium for i.

    Parameters
    ----------
    i : list, tuple, or np.ndarray
        Infectious fraction with each variant as list.
    sigma : float, optional
        Cross-immunity factor. The default is 0.7.
    beta_1 : float, optional
        Transmission rate for variant 1. The default is 0.3.
    beta_2 : float, optional
        Transmission rate for variant 2. The default is 0.35.
    gamma_1 : float, optional
        Reocvery rate for variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate for variant 2. The default is 0.1.
    b : float, optional
        Birth and death rate. The default is 0.01.

    Returns
    -------
    list
        Difference between infectious fraction given and as computed for the
        two-variant endemic equilibrium.
    """
    i_1, i_2 = i
    return [
        x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_2(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        - i_1,
        x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        + x_5(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        - i_2,
    ]


sigmas = np.linspace(0, 0.8, 5)
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01


np.random.seed(42)
N_initial_values = 1000

plt.figure(figsize=(4, 4))

sol = root(
    equation,
    [1, 0],
    args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "o", label="Variant 1 equilibrium")
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

sol = root(
    equation,
    [0, 1],
    args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "o", label="Variant 2 equilibrium")
plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)


for s in sigmas:
    sol = root(
        equation,
        [1, 1],
        args=(s, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"$\sigma={round(s, 1)}$")

sigmas_line = np.linspace(0, 1, 101)
i_solutions = []
for s in sigmas_line:
    sol = root(
        equation,
        [1, 1],
        args=(s, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    i_solutions.append(sol.x)

i_solutions = np.array(i_solutions).T
plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

plt.xlim((-0.002, 0.065))
plt.ylim((-0.002, 0.065))
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_sensitivity.jpg", dpi=500)
plt.show()
