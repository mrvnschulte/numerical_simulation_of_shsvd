import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from invasion_reproduction_number import (
    critical_r01,
    critical_r02,
    critical_sigma,
)


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


def x_bar(
    i,
    sigma=0.7,
    beta_1=0.3,
    beta_2=0.35,
    gamma_1=0.1,
    gamma_2=0.1,
    b=0.01,
):
    """
    Computes the two-variant endemic equilibrium for given i.

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
        Two-strain endemic equilibrium in original compartments.
    """

    i_1, i_2 = i
    return [
        x_1(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_2(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_3(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_4(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_5(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_6(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_7(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_8(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        x_9(i_1, i_2, sigma, beta_1, beta_2, gamma_1, gamma_2, b),
    ]


def susceptibles(x):
    return x[5]


def protected_1(x):
    return x[0] + x[6]


def protected_2(x):
    return x[3] + x[7]


def protected_12(x):
    return x[1] + x[2] + x[4] + x[8]


### Plot sigma sensitivity for one beta_2

sigmas = np.linspace(0, 0.9, 10)
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01

plt.figure(figsize=(5, 5))

sol = root(
    equation,
    [1, 0],
    args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s", label="Variant 1 equilibrium")
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

sol = root(
    equation,
    [0, 1],
    args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s", label="Variant 2 equilibrium")
plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)
plt.gca().set_prop_cycle(None)


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
x_solutions = []
for s in sigmas_line:
    sol = root(
        equation,
        [1, 1],
        args=(s, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    i_solutions.append(sol.x)
    x_solutions.append(x_bar(sol.x, s, beta_1, beta_2, gamma_1, gamma_2, b))

i_solutions = np.array(i_solutions).T
x_solutions = np.array(x_solutions)
plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

plt.xlim((-0.002, 0.065))
plt.ylim((-0.002, 0.065))
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_sensitivity_sigma.jpg", dpi=500)
plt.show()

threshold_sigma = critical_sigma(
    beta_1 / (gamma_1 + b), beta_2 / (gamma_2 + b)
)

plt.semilogy(sigmas_line, x_solutions, label=[rf"$x_{i+1}$" for i in range(9)])
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.ylim(bottom=1e-5, top=1)
plt.legend(loc="lower left")
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_x_sensitivity_sigma.jpg", dpi=500)
plt.show()

plt.plot(
    sigmas_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["s", r"$p_1$", r"$p_2$", r"$p_{12}$"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_p_sensitivity_sigma.jpg", dpi=500)
plt.show()

plt.plot(
    sigmas_line,
    i_solutions.T,
    label=[r"$i_1$", r"$i_2$"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_i_sensitivity_sigma.jpg", dpi=500)
plt.show()


### Plot beta_2 sensitivity for one sigma
sigma = 0.8
beta_1 = 0.3
beta_2s = np.linspace(0.2, 0.55, 8)
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01

plt.figure(figsize=(5, 5))

for beta_2 in beta_2s:
    sol = root(
        equation,
        [1, 1],
        args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"$\beta_2={round(beta_2, 2)}$")

beta_2s_line = np.linspace(0.2, 0.55, 101)
i_solutions = []
x_solutions = []
for beta_2 in beta_2s_line:
    sol = root(
        equation,
        [1, 1],
        args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    i_solutions.append(sol.x)
    x_solutions.append(
        x_bar(sol.x, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
    )

i_solutions = np.array(i_solutions).T
x_solutions = np.array(x_solutions)
plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_sensitivity_r02.jpg", dpi=500)
plt.show()

lower_threshold_beta2 = (gamma_2 + b) * critical_r01(
    beta_1 / (gamma_1 + b), sigma
)
upper_threshold_beta2 = (gamma_2 + b) * critical_r02(
    beta_1 / (gamma_1 + b), sigma
)

plt.semilogy(
    beta_2s_line, x_solutions, label=[rf"$x_{i+1}$" for i in range(9)]
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.ylim(bottom=1e-5, top=1)
plt.legend(loc=(1.02, 1e-3))
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_x_sensitivity_r02.jpg", dpi=500)
plt.show()

plt.plot(
    beta_2s_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["s", r"$p_1$", r"$p_2$", r"$p_{12}$"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_p_sensitivity_r02.jpg", dpi=500)
plt.show()

plt.plot(
    beta_2s_line,
    i_solutions.T,
    label=[r"$i_1$", r"$i_2$"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_i_sensitivity_r02.jpg", dpi=500)
plt.show()


# Plot sigma sensitivity for multiple beta_2

sigmas = np.linspace(0, 1, 11)
beta_1 = 0.3
beta_2s = [0.2, 0.25, 0.3, 0.35, 0.45, 0.55]
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01


plt.figure(figsize=(5, 5))

sol = root(
    equation,
    [1, 0],
    args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

i1_line = np.linspace(0, sol.x[0], 101)
i1_i2_line = sol.x[0] - i1_line
plt.plot(i1_line, i1_i2_line, linestyle="dotted", color="black", alpha=0.4)

for beta_2 in beta_2s:

    sol = root(
        equation,
        [0, 1],
        args=(1, beta_1, beta_2, gamma_1, gamma_2, b),
        method="hybr",
    )
    plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)
    plt.gca().set_prop_cycle(None)

    for s in sigmas:
        sol = root(
            equation,
            [1, 1],
            args=(s, beta_1, beta_2, gamma_1, gamma_2, b),
            method="hybr",
        )
        plt.plot(
            sol.x[0],
            sol.x[1],
            "o",
            label=(rf"$\sigma={round(s, 1)}$" if beta_2 == 0.2 else None),
        )
        plt.text(
            sol.x[0] + 0.004,
            sol.x[1] - 0.0007,
            rf"$\beta_2={round(beta_2, 2)}$" if s == 0 else "",
        )

    sigmas_line = np.linspace(0, 1, 101)
    i_solutions = []
    x_solutions = []
    for s in sigmas_line:
        sol = root(
            equation,
            [1, 1],
            args=(s, beta_1, beta_2, gamma_1, gamma_2, b),
            method="hybr",
        )
        i_solutions.append(sol.x)
        x_solutions.append(
            x_bar(sol.x, s, beta_1, beta_2, gamma_1, gamma_2, b)
        )

    i_solutions = np.array(i_solutions).T
    x_solutions = np.array(x_solutions)
    plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

plt.ylim(bottom=-0.002)
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_multisensitivity_sigma.jpg", dpi=500)
plt.show()


### Plot beta_2 sensitivity for multiple sigma

sigmas = [0.7, 0.8, 0.9]
beta_1 = 0.3
beta_2s = np.linspace(0.2, 0.6, 9)
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01

plt.figure(figsize=(5, 5))

for sigma in sigmas:
    upper_threshold_beta2 = (gamma_2 + b) * critical_r02(
        beta_1 / (gamma_1 + b), sigma
    )

    plt.gca().set_prop_cycle(None)
    for beta_2 in beta_2s:
        if beta_2 > upper_threshold_beta2:
            continue
        sol = root(
            equation,
            [1, 1],
            args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
            method="hybr",
        )
        plt.plot(
            sol.x[0],
            sol.x[1],
            "o",
            label=(rf"$\beta_2={round(beta_2, 2)}$" if sigma == 0.7 else None),
        )

    beta_2s_line = np.linspace(0.2, min(0.6, upper_threshold_beta2), 101)
    i_solutions = []
    x_solutions = []
    for beta_2 in beta_2s_line:
        sol = root(
            equation,
            [1, 1],
            args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
            method="hybr",
        )
        i_solutions.append(sol.x)
        x_solutions.append(
            x_bar(sol.x, sigma, beta_1, beta_2, gamma_1, gamma_2, b)
        )

    i_solutions = np.array(i_solutions).T
    x_solutions = np.array(x_solutions)
    plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

plt.text(0.001, 0.0525, r"$\sigma=0.9$")
plt.text(-0.005, 0.0725, r"$\sigma=0.8$")
plt.text(0.025, 0.0525, r"$\sigma=0.7$")

plt.plot(i1_line, i1_line, color="black", linestyle="dashed", alpha=0.4)
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.savefig("Plots/Endemic_equilibrium_multisensitivity_r02.jpg", dpi=500)
plt.show()
