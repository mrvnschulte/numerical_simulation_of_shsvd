# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:47:50 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

"""
Computes the probability functions and the influence of parameters on it
"""


def sir_model(t, x, beta, gamma):
    """
    Computes the derivatives of a state of a standard SIR model.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    dx : list
        List of the derivatives of the state variables.
    """
    s, i, r = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    dx = [ds, di, dr]

    return dx


def compute_mutation_time_pdf(i, r, m, gamma):
    """
    Computes the mutation pdf, given a mutation happens.

    Computes the mutation probability density function, under the condition
    that a mutation happens.

    Parameters
    ----------
    i : list, or np.ndarray
        Fraction of infectious individuals.
    r : list, or np.ndarray
        Fraction of recovered individuals. Must be the same length as i.
    m : float
        Mutation rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    pdf : np.ndarray
        The probability density function. Has the same length as i.
    """
    pdf = (m * i * np.exp(-m * r / gamma)) / (1 - np.exp(-m * r[-1] / gamma))

    return pdf


def compute_mutation_time_cdf(i, r, m, gamma):
    """
    Computes the mutation cdf, given a mutation happens.

    Computes the mutation cumulative density function, under the condition that
    a mutation happens.

    Parameters
    ----------
    i : list, or np.ndarray
        Fraction of infectious individuals.
    r : list, or np.ndarray
        Fraction of recovered individuals. Must be the same length as i.
    m : float
        Mutation rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    cdf : np.ndarray
        The cumulative density function. Has the same length as i.
    """
    cdf = np.cumsum(compute_mutation_time_pdf(i, r, m, gamma))

    return cdf


def compute_unconditioned_mutation_time_pdf(i, r, m, gamma):
    """
    Computes the mutation pdf.

    Compute the mutation probability density function.

    Parameters
    ----------
    i : list, or np.ndarray
        Fraction of infectious individuals.
    r : list, or np.ndarray
        Fraction of recovered individuals. Must be the same length as i.
    m : float
        Mutation rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    pdf : np.ndarray
        The probability density function. Has the same length as i.
    """
    pdf = m * i * np.exp(-m * r / gamma)

    return pdf


def compute_unconditioned_mutation_time_cdf(i, r, m, gamma):
    """
    Computes the mutation cumulative density function.

    Parameters
    ----------
    i : list, or np.ndarray
        Fraction of infectious individuals.
    r : list, or np.ndarray
        Fraction of recovered individuals. Must be the same length as i.
    m : float
        Mutation rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    cdf : np.ndarray
        The cumulative density function. Has the same length as i.
    """
    cdf = np.cumsum(compute_unconditioned_mutation_time_pdf(i, r, m, gamma))

    return cdf


def compute_expected_mutation_time(t, i, r, m, gamma):
    """
    Computes the expected mutation time, given a mutation happens.

    Parameters
    ----------
    t : list, or np.ndarray
        Time points at which i and r were computed.
    i : list, or np.ndarray
        Fraction of infectious individuals. Must be the same length as t.
    r : list, or np.ndarray
        Fraction of recovered individuals. Must be the same length as t.
    m : float
        Mutation rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    expected_mutation_time : float
        The cumulative density function. Has the same length as t.
    """
    expected_mutation_time = np.sum(
        t * compute_mutation_time_pdf(i, r, m, gamma)
    )

    return expected_mutation_time


def lambda_2(t, tau, i, m):
    """
    Computes the mutation probability for the 2-variant model.

    Parameters
    ----------
    t : int, list, or np.ndarray
        Time.
    tau : int, list, or np.ndarray
        Time after which second strain emerged.
    i : list, or np.ndarray
        Fraction of infectious individuals in SIR model with time step 1.
    m : float
        Mutation rate.

    Returns
    -------
    lambda2 : float, list, or np.ndarray
        Mutation probability of the 2-variant model.
    """
    if isinstance(t, (list, np.ndarray)):
        lambda2 = np.array([lambda_2(t, tau, i, m) for t in t])
        return lambda2

    if isinstance(tau, (list, np.ndarray)):
        lambda2 = np.array([lambda_2(t, tau, i, m) for tau in tau])
        return lambda2

    if t >= tau:
        if t >= len(i):
            lambda2 = m * i[-1] + m * (1 - i[-1]) * i[t - tau]
        else:
            lambda2 = m * i[t] + m * (1 - i[t]) * i[t - tau]
    else:
        lambda2 = 0

    return lambda2


def iterated_sum_poisson(max_iter, lambda_param, n):
    """
    Computes n realizations of the sum of iterated Poisson variables.

    Computes n realizations of the sum of iterated Poisson variables with
    Poisson parameter lambda_param. Another realization of a
    Poisson-distributed random variable is added to the respective sum if the
    sum of the first m variables is greater or equal to m.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    lambda_param : float
        Poisson parameter of the random variables.
    n : int
        Number of realizations to be computed.

    Returns
    -------
    sum_poisson_random_variable : np.ndarray
        Array of n realizations of the iterated Poisson variables.
    """
    sum_poisson_random_variable = np.random.poisson(lambda_param, n)

    for i in range(max_iter - 1):
        sum_poisson_random_variable += np.where(
            sum_poisson_random_variable > i,
            np.random.poisson(lambda_param, n),
            0,
        )

    return sum_poisson_random_variable


# Initial values and parameters
t0 = 0
tf = 500
x0 = [0.999, 0.001, 0]
beta_values = [0.15, 0.2, 0.125]
gamma_values = [0.1, 0.125, 0.075]
mutation_rates = [0.1, 0.5, 0.02]

t_span = (t0, tf)
t_eval = np.linspace(t0, tf, tf + 1)

"""
Standard case
"""
gamma = gamma_values[0]
beta = beta_values[0]
m = mutation_rates[0]

# Solve the standard SIR model
res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
i = res.y[1]
r = res.y[2]

# Compute the pdf and cdf of a mutation happening and the expected mutation
# time, given that a mutation happens
mutation_time_pdf = compute_mutation_time_pdf(i, r, m, gamma)
mutation_time_cdf = compute_mutation_time_cdf(i, r, m, gamma)
unconditioned_mutation_time_pdf = compute_unconditioned_mutation_time_pdf(
    i, r, m, gamma
)
unconditioned_mutation_time_cdf = compute_unconditioned_mutation_time_cdf(
    i, r, m, gamma
)
expected_mutation_time = compute_expected_mutation_time(t_eval, i, r, m, gamma)

# Compute the conditioned pdf of the second mutation
integration_value = np.zeros((len(t_eval), len(t_eval)))
t_eval = t_eval.astype(int)

for j, tau in enumerate(t_eval):
    integration_value[j] = (
        lambda_2(t_eval, tau, i, m)
        * np.exp(-np.cumsum(lambda_2(t_eval, tau, i, m)))
        / (1 - np.exp(-np.sum(lambda_2(t_eval + tau, tau, i, m))))
        * mutation_time_pdf[j]
    )
second_mutation_time_pdf = np.sum(integration_value, axis=0)

# Plot the computed quantities
fig1 = plt.figure(1)
plt.plot(t_eval, i)
plt.axvline(
    expected_mutation_time, color="#ff7f0e", label="Expected Mutation Time"
)

fig2 = plt.figure(2)
plt.plot(t_eval, mutation_time_pdf)
plt.axvline(
    expected_mutation_time, color="#ff7f0e", label="Expected Mutation Time"
)

fig3 = plt.figure(3)
plt.plot(t_eval, mutation_time_cdf)
plt.axvline(
    expected_mutation_time, color="#ff7f0e", label="Expected Mutation Time"
)

fig4 = plt.figure(4)
plt.plot(t_eval, unconditioned_mutation_time_pdf)
plt.axvline(
    expected_mutation_time, color="#ff7f0e", label="Expected Mutation Time"
)

fig5 = plt.figure(5)
plt.plot(t_eval, unconditioned_mutation_time_cdf)
plt.axvline(
    expected_mutation_time, color="#ff7f0e", label="Expected Mutation Time"
)

print("Probability of no mutation:")
print(np.exp(-m * r[-1] / gamma))

fig6 = plt.figure(6)
plt.plot(t_eval, second_mutation_time_pdf)

fig7 = plt.figure(7)
first_mutation_time = 200
plt.plot(
    t_eval,
    lambda_2(t_eval, first_mutation_time, i, m)
    * np.exp(-np.cumsum(lambda_2(t_eval, first_mutation_time, i, m))),
    # * mutation_time_pdf[first_mutation_time],
    label=rf"$\beta$={beta}",
)

plt.figure(fig1)
plt.title(
    "Infectious fraction in an SIR model\n"
    rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Infectious fraction")
plt.legend()

plt.figure(fig2)
plt.title(
    "Pdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig3)
plt.title(
    "Cdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig4)
plt.title(
    "Pdf of the first mutation\n" rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig5)
plt.title(
    "Cdf of the first mutation\n" rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig6)
plt.title(
    "Pdf of the second mutation, given that it happens\n"
    rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")

plt.figure(fig7)
plt.title(
    "Probability density of second mutation, given that first happens at"
    f" {first_mutation_time}\n"
    rf"($\beta={beta}, \gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")

plt.show()

"""
Parameter study for beta
"""
for beta in beta_values:
    gamma = gamma_values[0]
    m = mutation_rates[0]

    # Solve the standard SIR model
    res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    i = res.y[1]
    r = res.y[2]

    # Compute the pdf and cdf of a mutation happening and the expected mutation
    # time, given that a mutation happens
    mutation_time_pdf = compute_mutation_time_pdf(i, r, m, gamma)
    mutation_time_cdf = compute_mutation_time_cdf(i, r, m, gamma)
    unconditioned_mutation_time_pdf = compute_unconditioned_mutation_time_pdf(
        i, r, m, gamma
    )
    unconditioned_mutation_time_cdf = compute_unconditioned_mutation_time_cdf(
        i, r, m, gamma
    )
    expected_mutation_time = compute_expected_mutation_time(
        t_eval, i, r, m, gamma
    )

    # Compute the conditioned pdf of the second mutation
    integration_value = np.zeros((len(t_eval), len(t_eval)))
    t_eval = t_eval.astype(int)

    for j, tau in enumerate(t_eval):
        integration_value[j] = (
            lambda_2(t_eval, tau, i, m)
            * np.exp(-np.cumsum(lambda_2(t_eval, tau, i, m)))
            / (1 - np.exp(-np.sum(lambda_2(t_eval + tau, tau, i, m))))
            * mutation_time_pdf[j]
        )
    second_mutation_time_pdf = np.sum(integration_value, axis=0)

    # Plot the computed quantities
    fig1 = plt.figure(1)
    plt.plot(t_eval, i, label=rf"$\beta$={beta}")

    fig2 = plt.figure(2)
    plt.plot(t_eval, mutation_time_pdf, label=rf"$\beta$={beta}")

    fig3 = plt.figure(3)
    plt.plot(t_eval, mutation_time_cdf, label=rf"$\beta$={beta}")

    fig4 = plt.figure(4)
    plt.plot(t_eval, unconditioned_mutation_time_pdf, label=rf"$\beta$={beta}")

    fig5 = plt.figure(5)
    plt.plot(t_eval, unconditioned_mutation_time_cdf, label=rf"$\beta$={beta}")

    fig6 = plt.figure(6)
    plt.plot(t_eval, second_mutation_time_pdf, label=rf"$\beta$={beta}")

    fig7 = plt.figure(7)
    first_mutation_time = 200
    plt.plot(
        t_eval,
        lambda_2(t_eval, first_mutation_time, i, m)
        * np.exp(-np.cumsum(lambda_2(t_eval, first_mutation_time, i, m))),
        # * mutation_time_pdf[first_mutation_time],
        label=rf"$\beta$={beta}",
    )

plt.figure(fig1)
plt.title(
    "Infectious fraction in an SIR model\n" rf"($\gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Infectious fraction")
plt.legend()

plt.figure(fig2)
plt.title(
    "Pdf of the first mutation given that a mutation happens\n"
    rf"($\gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig3)
plt.title(
    "Cdf of the first mutation given that a mutation happens\n"
    rf"($\gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig4)
plt.title("Pdf of the first mutation\n" rf"($\gamma={gamma}, m={m}$)")
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig5)
plt.title("Cdf of the first mutation\n" rf"($\gamma={gamma}, m={m}$)")
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig6)
plt.title(
    "Pdf of the second mutation, given that it happens\n"
    rf"($\gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig7)
plt.title(
    "Probability density of second mutation, given that first happens at"
    f" {first_mutation_time}\n"
    rf"($\gamma={gamma}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.show()

"""
Parameter study for gamma
"""
for gamma in gamma_values:
    beta = beta_values[0]
    m = mutation_rates[0]

    # Solve the standard SIR model
    res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    i = res.y[1]
    r = res.y[2]

    # Compute the pdf and cdf of a mutation happening and the expected mutation
    # time, given that a mutation happens
    mutation_time_pdf = compute_mutation_time_pdf(i, r, m, gamma)
    mutation_time_cdf = compute_mutation_time_cdf(i, r, m, gamma)
    unconditioned_mutation_time_pdf = compute_unconditioned_mutation_time_pdf(
        i, r, m, gamma
    )
    unconditioned_mutation_time_cdf = compute_unconditioned_mutation_time_cdf(
        i, r, m, gamma
    )
    expected_mutation_time = compute_expected_mutation_time(
        t_eval, i, r, m, gamma
    )

    # Compute the conditioned pdf of the second mutation
    integration_value = np.zeros((len(t_eval), len(t_eval)))
    t_eval = t_eval.astype(int)

    for j, tau in enumerate(t_eval):
        integration_value[j] = (
            lambda_2(t_eval, tau, i, m)
            * np.exp(-np.cumsum(lambda_2(t_eval, tau, i, m)))
            / (1 - np.exp(-np.sum(lambda_2(t_eval + tau, tau, i, m))))
            * mutation_time_pdf[j]
        )
    second_mutation_time_pdf = np.sum(integration_value, axis=0)

    # Plot the computed quantities
    fig1 = plt.figure(1)
    plt.plot(t_eval, i, label=rf"$\gamma$={gamma}")

    fig2 = plt.figure(2)
    plt.plot(t_eval, mutation_time_pdf, label=rf"$\gamma$={gamma}")

    fig3 = plt.figure(3)
    plt.plot(t_eval, mutation_time_cdf, label=rf"$\gamma$={gamma}")

    fig4 = plt.figure(4)
    plt.plot(
        t_eval, unconditioned_mutation_time_pdf, label=rf"$\gamma$={gamma}"
    )

    fig5 = plt.figure(5)
    plt.plot(
        t_eval, unconditioned_mutation_time_cdf, label=rf"$\gamma$={gamma}"
    )

    fig6 = plt.figure(6)
    plt.plot(t_eval, second_mutation_time_pdf, label=rf"$\gamma$={gamma}")

    fig7 = plt.figure(7)
    first_mutation_time = 200
    plt.plot(
        t_eval,
        lambda_2(t_eval, first_mutation_time, i, m)
        * np.exp(-np.cumsum(lambda_2(t_eval, first_mutation_time, i, m))),
        # * mutation_time_pdf[first_mutation_time],
        label=rf"$\gamma$={gamma}",
    )

plt.figure(fig1)
plt.title("Infectious fraction in an SIR model\n" rf"($\beta={beta}, m={m}$)")
plt.xlabel("Time")
plt.ylabel("Infectious fraction")
plt.legend()

plt.figure(fig2)
plt.title(
    "Pdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig3)
plt.title(
    "Cdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig4)
plt.title("Pdf of the first mutation\n" rf"($\beta={beta}, m={m}$)")
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig5)
plt.title("Cdf of the first mutation\n" rf"($\beta={beta}, m={m}$)")
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig6)
plt.title(
    "Pdf of the second mutation, given that it happens\n"
    rf"($\beta={beta}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig7)
plt.title(
    "Probability density of second mutation, given that first happens at"
    f" {first_mutation_time}\n"
    rf"($\beta={beta}, m={m}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.show()

"""
Parameter study for m
"""
for m in mutation_rates:
    beta = beta_values[0]
    gamma = gamma_values[0]

    # Solve the standard SIR model
    res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
    i = res.y[1]
    r = res.y[2]

    # Compute the pdf and cdf of a mutation happening and the expected mutation
    # time, given that a mutation happens
    mutation_time_pdf = compute_mutation_time_pdf(i, r, m, gamma)
    mutation_time_cdf = compute_mutation_time_cdf(i, r, m, gamma)
    unconditioned_mutation_time_pdf = compute_unconditioned_mutation_time_pdf(
        i, r, m, gamma
    )
    unconditioned_mutation_time_cdf = compute_unconditioned_mutation_time_cdf(
        i, r, m, gamma
    )
    expected_mutation_time = compute_expected_mutation_time(
        t_eval, i, r, m, gamma
    )

    # Compute the conditioned pdf of the second mutation
    integration_value = np.zeros((len(t_eval), len(t_eval)))
    t_eval = t_eval.astype(int)

    for j, tau in enumerate(t_eval):
        integration_value[j] = (
            lambda_2(t_eval, tau, i, m)
            * np.exp(-np.cumsum(lambda_2(t_eval, tau, i, m)))
            / (1 - np.exp(-np.sum(lambda_2(t_eval + tau, tau, i, m))))
            * mutation_time_pdf[j]
        )
    second_mutation_time_pdf = np.sum(integration_value, axis=0)

    # Plot the computed quantities
    fig1 = plt.figure(1)
    plt.plot(t_eval, i, label=rf"$m$={m}")

    fig2 = plt.figure(2)
    plt.plot(t_eval, mutation_time_pdf, label=rf"$m$={m}")

    fig3 = plt.figure(3)
    plt.plot(t_eval, mutation_time_cdf, label=rf"$m$={m}")

    fig4 = plt.figure(4)
    plt.plot(t_eval, unconditioned_mutation_time_pdf, label=rf"$m$={m}")

    fig5 = plt.figure(5)
    plt.plot(t_eval, unconditioned_mutation_time_cdf, label=rf"$m$={m}")

    fig6 = plt.figure(6)
    plt.plot(t_eval, second_mutation_time_pdf, label=rf"$m$={m}")

    fig7 = plt.figure(7)
    first_mutation_time = 200
    plt.plot(
        t_eval,
        lambda_2(t_eval, first_mutation_time, i, m)
        * np.exp(-np.cumsum(lambda_2(t_eval, first_mutation_time, i, m))),
        # * mutation_time_pdf[first_mutation_time],
        label=rf"$m$={m}",
    )

plt.figure(fig1)
plt.title(
    "Infectious fraction in an SIR model\n"
    rf"($\beta={beta}, \gamma={gamma}$)"
)
plt.xlabel("Time")
plt.ylabel("Infectious fraction")
plt.legend()

plt.figure(fig2)
plt.title(
    "Pdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, gamma={gamma}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig3)
plt.title(
    "Cdf of the first mutation given that a mutation happens\n"
    rf"($\beta={beta}, \gamma={gamma}$)"
)
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig4)
plt.title("Pdf of the first mutation\n" rf"($\beta={beta}, \gamma={gamma}$)")
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig5)
plt.title("Cdf of the first mutation\n" rf"($\beta={beta}, \gamma={gamma}$)")
plt.xlabel("Time")
plt.ylabel("Cumulative probability")
plt.legend()

plt.figure(fig6)
plt.title(
    "Pdf of the second mutation, given that it happens\n"
    rf"($\beta={beta}, \gamma={gamma}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.figure(fig7)
plt.title(
    "Probability density of second mutation, given that first happens at"
    f" {first_mutation_time}\n"
    rf"($\beta={beta}, \gamma={gamma}$)"
)
plt.xlabel("Time")
plt.ylabel("Probability density")
plt.legend()

plt.show()

# Set standard parameters
gamma = gamma_values[0]
beta = beta_values[0]
m = mutation_rates[0]

# Solve the standard SIR model
res = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(beta, gamma))
i = res.y[1]
r = res.y[2]

# Set parameters
lambda_param = m * r[-1] / gamma
n = 100000
max_iter = 10

# Generate random variables
sum_poisson = iterated_sum_poisson(max_iter, lambda_param, n)

# Plot random variables as histogram
counts, bins = np.histogram(
    sum_poisson, bins=np.arange(0, max(sum_poisson) + 1), density=True
)
mask = bins[:-1] < max_iter
plt.bar(
    bins[:-1][mask],
    counts[mask],
    width=1,
    alpha=0.75,
    color="blue",
    edgecolor="black",
    label="Values <= max_iter",
)
mask = bins[:-1] >= max_iter
plt.bar(
    bins[:-1][mask],
    counts[mask],
    width=1,
    alpha=0.75,
    color="red",
    edgecolor="black",
    label="Values > max_iter",
)
plt.title("Histogram of iterated Poisson variables after 10 iterations")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid()
plt.show()
