import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import sympy as sp

from invasion_reproduction_number import invasion_reproduction_number


def two_variant_model(t, x, beta1, beta2, gamma1, gamma2, sigma, b):
    """
    Full two-variant cross-immmunity model with demographics.

    Parameters
    ----------
    t : float
        Time (not used).
    x : list, or np.ndarray
        State of the system.
    beta1 : float
        Transmission rate of variant 1.
    beta2 : float
        Transmission rate of variant 2.
    gamma1 : float
        Recovery rate of variant 1.
    gamma2 : float
        Recovery rate of variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    f : list
        Derivatives of the state.
    """

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x

    f1 = (
        -gamma1 * x1
        - beta2 * x1 * (x3 + x4 + x5)
        + (1 - sigma) * beta1 * x6 * (x1 + x2 + x3)
        - b * x1
    )
    f2 = (
        -gamma1 * x2
        + sigma * beta1 * x6 * (x1 + x2 + x3)
        + beta1 * x8 * (x1 + x2 + x3)
        + gamma2 * x3
        - b * x2
    )
    f3 = (
        -gamma1 * x3
        - gamma2 * x3
        + beta1 * x4 * (x1 + x2 + x3)
        + beta2 * x1 * (x3 + x4 + x5)
        - b * x3
    )
    f4 = (
        -gamma2 * x4
        - beta1 * x4 * (x1 + x2 + x3)
        + (1 - sigma) * beta2 * x6 * (x3 + x4 + x5)
        - b * x4
    )
    f5 = (
        -gamma2 * x5
        + sigma * beta2 * x6 * (x3 + x4 + x5)
        + beta2 * x7 * (x3 + x4 + x5)
        + gamma1 * x3
        - b * x5
    )
    f6 = (
        -beta1 * x6 * (x1 + x2 + x3) - beta2 * x6 * (x3 + x4 + x5) + b - b * x6
    )
    f7 = -beta2 * x7 * (x3 + x4 + x5) + gamma1 * x1 - b * x7
    f8 = -beta1 * x8 * (x1 + x2 + x3) + gamma2 * x4 - b * x8
    f9 = gamma1 * x2 + gamma2 * x5 - b * x9

    f = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    return f


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


def check_validity(x):
    """
    Checks biological feasibility of a state x.

    Parameters
    ----------
    x : list, or np.ndarray
        State of the system.

    Returns
    -------
    bool
        True, if x is biologically feasible. False, otherwise.
    """
    x = np.array(x)
    return np.all((0 <= x) & (x <= 1))


def random_simplex_points(n, dim=9):
    """
    Generate random points on the standard simplex according to Dirichlet
    distribution.

    Parameters
    ----------
    n : int
        Number of points to generate.
    dim : int, optional
        Dimension of x. The default is 9.

    Returns
    -------
    np.ndarray
        Array of n random points on the simplex.
    """
    alpha = np.ones(dim)
    mats = np.random.dirichlet(alpha, size=n)
    return mats


def sample_parameters(rng):
    """
    Sample random parameters for the model.

    Parameters
    ----------
    rng : TYPE
        Random number generator.

    Returns
    -------
    dict
        Dictionary of model parameters.
    """
    sigma = rng.uniform(0.0, 1.0)
    beta1 = rng.uniform(0.1, 2.0)
    beta2 = rng.uniform(0.1, 2.0)
    gamma1 = rng.uniform(0.1, beta1)
    gamma2 = rng.uniform(0.1, beta2)
    b = rng.uniform(0.01, 0.1)
    return dict(
        sigma=sigma,
        beta1=beta1,
        beta2=beta2,
        gamma1=gamma1,
        gamma2=gamma2,
        b=b,
    )


def check_parameter_sets(
    N_target_sets, N_startpoints, threshold_r0, filename, verbose=False
):
    rng = np.random.default_rng(seed=42)

    tf = 5000

    valid_stats = []

    # Symbolic variables

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols(
        "x1 x2 x3 x4 x5 x6 x7 x8 x9"
    )
    gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
        "gamma1 gamma2 beta1 beta2 sigma b", positive=True
    )

    xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

    F = two_variant_model(0, xs, beta1, beta2, gamma1, gamma2, sigma, b)

    J = sp.Matrix(F).jacobian(xs)

    cycles = 0
    while len(valid_stats) < N_target_sets:
        rng = np.random.default_rng(seed=cycles)
        cycles += 1
        params = sample_parameters(rng)
        sigma_val = params["sigma"]
        beta_1 = params["beta1"]
        beta_2 = params["beta2"]
        gamma_1 = params["gamma1"]
        gamma_2 = params["gamma2"]
        b_val = params["b"]

        # Reproduction numbers
        r_01 = beta_1 / (gamma_1 + b_val)
        r_02 = beta_2 / (gamma_2 + b_val)
        r_012 = invasion_reproduction_number(
            beta_1 / (gamma_1 + b_val), beta_2 / (gamma_2 + b_val), sigma_val
        )
        r_021 = invasion_reproduction_number(
            beta_2 / (gamma_2 + b_val), beta_1 / (gamma_1 + b_val), sigma_val
        )

        if not (r_012 > threshold_r0 and r_021 > threshold_r0):
            continue

        sol = root(
            equation,
            [1, 1],
            args=(sigma_val, beta_1, beta_2, gamma_1, gamma_2, b_val),
            method="hybr",
        )
        if min(sol.x) < 1e-3:
            continue
        x_star = np.array(
            x_bar(sol.x, sigma_val, beta_1, beta_2, gamma_1, gamma_2, b_val)
        )

        x_star_dict = {
            x1: x_star[0],
            x2: x_star[1],
            x3: x_star[2],
            x4: x_star[3],
            x5: x_star[4],
            x6: x_star[5],
            x7: x_star[6],
            x8: x_star[7],
            x9: x_star[8],
            b: b_val,
            beta1: beta_1,
            beta2: beta_2,
            gamma1: gamma_1,
            gamma2: gamma_2,
            sigma: sigma_val,
        }

        J_num = J.subs(x_star_dict)
        J_num = np.array(J_num.evalf().tolist(), dtype=float)

        eigvals = np.linalg.eigvals(J_num)
        max_real_part = np.max(eigvals.real)
        # if max_real_part > 0:
        #     print(rf"\nR_{{0,1\to2}}={r_012}")
        #     print(rf"R_{{0,2\to1}}={r_021}")
        #     print(max_real_part)
        #     print(sol.x)
        # valid_stats.append(max_real_part)

        x0_list = rng.dirichlet(alpha=np.ones(9), size=N_startpoints)

        final_states = []
        dist_to_xstar = []
        for x0 in x0_list:
            sol_ode = solve_ivp(
                two_variant_model,
                (0, tf),
                x0,
                args=(beta_1, beta_2, gamma_1, gamma_2, sigma_val, b_val),
                vectorized=False,
            )
            final_x = sol_ode.y.T[-1]
            if not check_validity(final_x):
                continue
            final_states.append(final_x)
            dist_to_xstar.append(np.linalg.norm(final_x - x_star))
            if np.linalg.norm(final_x - x_star) > 1e-3:
                plt.plot(sol_ode.t, sol_ode.y.T)
                for xi in x_star:
                    plt.axhline(xi, color="blue")
                plt.show()

        #     if len(valid_stats) == 2:
        #         print(f"Final state: {final_x}")
        #         print(
        #             f"With derivative: {two_variant_model(0, final_x, beta_1, beta_2, gamma_1, gamma_2, sigma, b)}"
        #         )
        #         plt.plot(sol_ode.t, sol_ode.y.T)
        #         for xi in x_star:
        #             plt.axhline(xi, color="blue")
        #         plt.show()

        # if len(valid_stats) == 2:
        #     print(f"Final state: {x_star}")
        #     print(
        #         f"With derivative: {two_variant_model(0, x_star, beta_1, beta_2, gamma_1, gamma_2, sigma, b)}"
        #     )
        #     print(f"Parameters: {params}")

        if len(final_states) == 0:
            continue

        final_states = np.array(final_states)
        dist_to_xstar = np.array(dist_to_xstar)

        stats = {
            "set": len(valid_stats),
            "params": params,
            "mean_dist": dist_to_xstar.mean(),
            "std_dist": dist_to_xstar.std(),
            "max_dist": dist_to_xstar.max(),
            "min_dist": dist_to_xstar.min(),
            "mean_final": final_states.mean(axis=0),
            "r_01": r_01,
            "r_02": r_02,
            "r_012": r_012,
            "r_021": r_021,
            "max_eigval": max_real_part,
        }
        if stats["max_dist"] > 1e-3:
            print(sol.x)
            plt.plot()
        valid_stats.append(stats)

        if verbose:
            print(
                f"Valid Set {stats['set']}: "
                f"R_01={r_01:.3f}, R_02={r_02:.3f}, R_012={r_012:.3f}, R_021={r_021:.3f}, "
                f"mean_dist={stats['mean_dist']:.4e}, max_dist={stats['max_dist']:.4e}, cycles={cycles}"
            )

    plt.scatter(
        [s["set"] for s in valid_stats], [s["max_dist"] for s in valid_stats]
    )
    plt.xlabel("Index of parameter set")
    plt.ylabel(r"Maximal distance $||x(5000) - \bar x||$")
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.show()

    plt.scatter(
        [s["set"] for s in valid_stats], [s["max_eigval"] for s in valid_stats]
    )
    plt.axhline(color="grey", linestyle="dotted")
    plt.xlabel("Index of parameter set")
    plt.ylabel(r"Maximal real part of eigenvalues$")
    plt.tight_layout()
    plt.savefig(f"{filename.replace('.jpg','')}_eigval.jpg", dpi=500)
    plt.show()

    max_val = max([min(s["r_012"], s["r_021"]) for s in valid_stats])
    bins = np.arange(1.0, max_val + 0.1, 0.1)
    plt.hist([min(s["r_012"], s["r_021"]) for s in valid_stats], bins=bins)
    plt.xlabel("Lower invasion reproduction number")
    plt.ylabel("Number of parameter sets")
    plt.tight_layout()
    plt.savefig(f"{filename.replace('.jpg','')}_stats.jpg", dpi=500)
    plt.show()

    print(max([s["max_eigval"] for s in valid_stats]))


### STEP 1: Check for given parameters and multiple initial values

sigma = 0.8
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01


np.random.seed(42)
N_initial_values = 100000
x0_list = random_simplex_points(N_initial_values, dim=9)

sol = root(
    equation,
    [1, 1],
    args=(sigma, beta_1, beta_2, gamma_1, gamma_2, b),
    method="hybr",
)
x_star = np.array(x_bar(sol.x, sigma, beta_1, beta_2, gamma_1, gamma_2, b))

tf = 1000
results_final = []
distances = []
for i in range(N_initial_values):
    x0 = x0_list[i]
    sol_ode = solve_ivp(
        two_variant_model,
        (0, tf),
        x0,
        args=(beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        vectorized=False,
    )
    final_x = sol_ode.y.T[-1]
    results_final.append(final_x)

    dist = np.linalg.norm(final_x - x_star)
    distances.append(dist)

distances = np.array(distances)

print("Mean distance to x_star:", distances.mean())
print("Maximal distance to x_star:", max(distances))

results_final = np.array(results_final)
plt.scatter(range(N_initial_values), distances)
plt.xlabel("Index of initial value")
plt.ylabel(r"Distance $||x(1000) - \bar x||$")
plt.tight_layout()
plt.savefig("Plots/Distance_different_initial_values.jpg", dpi=500)
plt.show()


### STEP 2: Check for mutliple parameter sets and initial values

N_target_sets = 1000
N_startpoints = 100
threshold_r0 = 1
filename = f"Distance_different_parameters_{threshold_r0}_test.jpg"

check_parameter_sets(
    N_target_sets, N_startpoints, threshold_r0, filename, verbose=True
)
