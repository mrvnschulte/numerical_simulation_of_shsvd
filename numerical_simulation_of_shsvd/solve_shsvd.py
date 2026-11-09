"""
Provides a solver for multiple trajectories of a stochastic hybrid system
of variable dimension.

The solver collects statistics and can stop after a prescribed number of
iterations or when changes in selected estimates fall below a specified
tolerance.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .solve_shsvd_single_run import solve_shsvd_single_run
from .stats_functions_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)


Args = tuple[Any, ...] | None
Estimate = str | tuple[str, int | float]
EstimateType = str | tuple[str, float]


def solve_shsvd(
    flow_function: Callable,
    jump_rate_function: Callable,
    transition_function: Callable,
    initial_value: ArrayLike,
    t_span: tuple[float, float],
    jump_rate_bound: float | None = None,
    jump_times: ArrayLike | None = None,
    jump_time_transition_function: Callable | None = None,
    jump_set_function: Callable | None = None,
    jump_set_transition_function: Callable | None = None,
    custom_evaluation_functions: Callable | list[Callable] | None = None,
    stepsize: float = 1,
    t_eval: ArrayLike | None = None,
    norm_order: int | float | ArrayLike = 2,
    collect_dimension: bool = True,
    collect_jump_times: bool = False,
    collect_dimension_jump_times: bool = False,
    flow_args: Args = None,
    jump_rate_args: Args = None,
    transition_args: Args = None,
    jump_time_transition_args: Args = None,
    jump_set_args: Args = None,
    jump_set_transition_args: Args = None,
    cef_args: tuple[Args, ...] | None = None,
    check_jump_rate: bool = False,
    iterations: int = 500,
    tol: float | ArrayLike = 1e-5,
    min_iterations: int = 10,
    max_iterations: int = 1000,
    convergence_estimate: Estimate | list[Estimate] | None = None,
    convergence_estimate_type: EstimateType | list[EstimateType] = "max",
    estimation_mode: str = "all",
    estimation_weights: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    solve_ivp_method: str = "RK45",
    verbose: bool = False,
    verbose_step: int = 1,
    **solve_ivp_options: Any,
) -> StatsFunctionsOfVariableDimension:
    """
    Computes statistics for a stochastic hybrid system of variable dimension.

    Simulates multiple trajectories with solve_shsvd_single_run and collects
    statistics of their norms, dimensions, and optional custom evaluations.
    The number of trajectories can be fixed or determined by a convergence
    criterion based on changes in the estimated means.

    Parameters
    ----------
    flow_function : callable
        Function defining the continuous dynamics. It is called as
        flow_function(t, y, *flow_args) and returns the state derivatives.
    jump_rate_function : callable
        Function defining the stochastic jump rate. It is called as
        jump_rate_function(t, y, *jump_rate_args) and returns a rate.
    transition_function : callable
        Transition function applied at accepted stochastic jumps.
    initial_value : array-like
        Initial state of the system.
    t_span : tuple of float
        Start and end times of the integration.
    jump_rate_bound : float or None, optional
        Upper bound for the stochastic jump rate. If None or 0, the bound is
        initialized and checked in each trajectory. The default is None.
    jump_times : list or np.ndarray or None, optional
        Prescribed jump times. A nonempty list or array must be provided when
        using this option. The default is None.
    jump_time_transition_function : callable or None, optional
        Transition function applied at prescribed jump times. If None,
        transition_function is used when jump_times is provided. The default
        is None.
    jump_set_function : callable or None, optional
        Event function defining the jump set. The default is None.
    jump_set_transition_function : callable or None, optional
        Transition function applied when the jump set is reached. If None,
        transition_function is used when jump_set_function is provided. The
        default is None.
    custom_evaluation_functions : callable, list of callable, or None, optional
        Functions evaluated on each trajectory at the statistics evaluation
        times. Each function receives the list of state values and should
        return one value per evaluation time. The default is None.
    stepsize : float, optional
        Step size used to construct the statistics evaluation grid when
        t_eval is None. It is also used in the 'int' convergence estimate.
        The default is 1.
    t_eval : array-like or None, optional
        Evaluation times for the trajectory solver and statistics collector.
        The interval endpoints are included as needed. The default is None.
    norm_order : int, float, or array-like, optional
        Norm order or orders for which statistics are collected. The default
        is 2.
    collect_dimension : bool, optional
        If True, collect statistics of the state dimension. The default is
        True.
    collect_jump_times : bool, optional
        If True, store the jump times of each trajectory. The default is
        False.
    collect_dimension_jump_times : bool, optional
        If True, store the dimension jump times of each trajectory. The
        default is False.
    flow_args : tuple or None, optional
        Additional arguments passed to flow_function. The default is None.
    jump_rate_args : tuple or None, optional
        Additional arguments passed to jump_rate_function. The default is
        None.
    transition_args : tuple or None, optional
        Additional arguments for transition_function. The default is None.
    jump_time_transition_args : tuple or None, optional
        Additional arguments for jump_time_transition_function. The default
        is None.
    jump_set_args : tuple or None, optional
        Additional arguments passed to jump_set_function. The default is None.
    jump_set_transition_args : tuple or None, optional
        Additional arguments for jump_set_transition_function. The default is
        None.
    cef_args : tuple of tuple or None, optional
        Additional arguments for the custom evaluation functions. When
        custom_evaluation_functions is a list, provide one tuple or None per
        function. The default is None.
    check_jump_rate : bool, optional
        If True, check the jump rate at stored values of each trajectory
        segment. The single-run solver also enables this check if the bound
        is None or 0. The default is False.
    iterations : int, optional
        Number of trajectories to simulate when convergence_estimate is None.
        The default is 500.
    tol : float or array-like, optional
        Tolerance for the convergence criterion. A scalar applies to all
        estimates; an array can specify individual tolerances for the 'all'
        and 'any' modes. The default is 1e-5.
    min_iterations : int, optional
        Minimum number of trajectories before convergence is checked. At
        least two are used. The default is 10.
    max_iterations : int, optional
        Maximum number of trajectories when convergence_estimate is provided.
        The default is 1000.
    convergence_estimate : str, tuple, list, or None, optional
        Mean or means whose changes are checked for convergence. Each
        specification is 'dimension', ('norm', order), or ('cef', index).
        The CEF index is zero-based. Dimension statistics and the requested
        norms or CEFs must be collected. If None, a fixed number of
        iterations is used. The default is None.
    convergence_estimate_type : str or list, optional
        Reduction applied to the absolute change in each selected mean:
        'max', 'mean', 'int', or ('time', t). The 'int' estimate is the sum
        of changes multiplied by stepsize. For ('time', t), t must be in the
        statistics evaluation grid. A string applies to all estimates;
        otherwise provide a list with one entry per estimate. The default is
        'max'.
    estimation_mode : str, optional
        Rule for comparing convergence estimates with tol: 'all', 'any',
        'sum', or 'weighted_sum'. The default is 'all'.
    estimation_weights : array-like or None, optional
        Factors and exponents for the 'weighted_sum' mode. Each row contains
        the factor and exponent for one convergence estimate. The default is
        None.
    rng : np.random.Generator or None, optional
        Random number generator shared by the simulated trajectories.
        If None, a new generator is created with np.random.default_rng().
        The default is None.
    solve_ivp_method : str, optional
        Integration method passed to solve_ivp. The default is 'RK45'.
    verbose : bool, optional
        If True, print progress and convergence information. The default is
        False.
    verbose_step : int, optional
        Interval, in trajectories, between progress messages. The default is
        1.
    **solve_ivp_options : dict
        Additional keyword arguments passed to solve_ivp.

    Returns
    -------
    stats : StatsFunctionsOfVariableDimension
        Statistics collected from the simulated trajectories.

    Raises
    ------
    ValueError
        If the number of CEF argument tuples does not match the number of
        CEFs, or if a convergence specification, estimate type, mode, or
        requested norm is invalid.

    Notes
    -----
    Convergence is assessed from changes between consecutive estimates of
    the means. It is not a confidence-interval or sampling-error criterion.

    For estimation_mode='sum' or 'weighted_sum', tol should be a scalar.
    The 'int' estimate uses stepsize even when t_eval is provided or the
    evaluation grid is nonuniform.

    The state of a provided generator advances across trajectories.
    """
    if verbose:
        print("Initialising")

    # Initialise random number generator if none is provided
    if rng is None:
        rng = np.random.default_rng()

    # Initialise stats object
    stats = StatsFunctionsOfVariableDimension(
        t_span,
        stepsize,
        t_eval=t_eval,
        order=norm_order,
        collect_dimension=collect_dimension,
        collect_jump_times=collect_jump_times,
        collect_dimension_jump_times=collect_dimension_jump_times,
    )

    # Add custom evaluation functions with additional arguments, if any, to
    # stats object
    if custom_evaluation_functions is not None:
        if isinstance(custom_evaluation_functions, list):
            if cef_args is not None:
                if len(custom_evaluation_functions) != len(cef_args):
                    raise ValueError(
                        "Number of cef functions does not match "
                        "number of cef arguments"
                    )
                for cef, cef_arg in zip(custom_evaluation_functions, cef_args):
                    if cef_arg is not None:

                        def cef_with_args(
                            values: Any,
                            fun: Callable = cef,
                            args: tuple[Any, ...] = cef_arg,
                        ):
                            return fun(values, *args)

                        stats.add_custom_evaluation_function(cef_with_args)
                    else:
                        stats.add_custom_evaluation_function(cef)
            else:
                for cef in custom_evaluation_functions:
                    stats.add_custom_evaluation_function(cef)
        else:
            stats.add_custom_evaluation_function(custom_evaluation_functions)

    if convergence_estimate is not None:
        # Check if estimation_mode is set to reasonable value
        if estimation_mode not in ["all", "any", "sum", "weighted_sum"]:
            raise ValueError(
                f"{estimation_mode} is not a suitable estimation\
                             mode. These are 'all', 'any', 'sum' and\
                             'weighted_sum'."
            )

        # Set number of iterations
        min_iterations = max(min_iterations, 2)
        iterations = max_iterations

        # If only one convergence_estimate is provided, make it a list of one
        # element
        if not isinstance(convergence_estimate, list):
            convergence_estimate = [convergence_estimate]

        # Check whether one convergence_estimate_type is provided for all
        # convergence estimates or a list of the same length as
        # convergence_estimate
        if not isinstance(convergence_estimate_type, str):
            if not len(convergence_estimate_type) == len(convergence_estimate):
                raise ValueError(
                    "Either a string or a list of strings with \
                                 length equal to the number of convergence \
                                 estimates must be provided for convergence \
                                 estimate type. Possible options are 'max', \
                                 'mean', 'int' and 'time' together with a \
                                 specific time."
                )
        else:
            convergence_estimate_type = [convergence_estimate_type] * len(
                convergence_estimate
            )

        # Check whether convergence estimate types are set to possible options
        # and list indices of the options in a dictionary
        estimate_type_dict = {"max": [], "mean": [], "int": [], "time": []}
        for i, cet in enumerate(convergence_estimate_type):
            if cet not in ["max", "mean", "int"]:
                if cet[0] != "time":
                    raise ValueError(
                        f"Possible options for convergence estimate \
                            type are 'max', 'mean', 'int' and 'time'. {cet} is \
                            no known option."
                    )
                if cet[1] not in stats.times:
                    raise ValueError(
                        f"If specific time is used as convergence estimate, \
                            it must be in t_eval, but {cet[1]} is not."
                    )
                t_eval_list = list(stats.times)
                index = t_eval_list.index(cet[1])
                estimate_type_dict["time"].append([i, index])
            else:
                estimate_type_dict[cet].append(i)

        # If estimation_mode is weighted_sum, check whether estimation_weights
        # of the same length as convergence_estimate are provided and set
        # estimation_factors and estimation_powers
        if estimation_mode == "weighted_sum":
            if not len(convergence_estimate) == len(estimation_weights):
                raise ValueError(
                    "Number of estimation weigths must equal\
                                 number of convergence estimates."
                )
            estimation_factors = np.array(estimation_weights)[:, 0]
            estimation_powers = np.array(estimation_weights)[:, 1]

        # Check whether convergence_esimates are possible options and find
        # corresponding index to norm order
        for i, ce in enumerate(convergence_estimate):
            if not isinstance(ce, tuple):
                if not ce.lower() == "dimension":
                    raise ValueError(
                        "Convergence estimate has to be a tuple \
                                     consisting of the keyword 'norm' plus its\
                                     order, the keyword 'cef' plus its \
                                     position or be equal to the keyword \
                                     'dimension'"
                    )
                else:
                    convergence_estimate[i] = ("dimension", -1)
            else:
                if not (ce[0].lower() == "norm" or ce[0].lower() == "cef"):
                    raise ValueError(
                        "Convergence estimate has to be a tuple \
                                     consisting of the keyword 'norm' plus its\
                                     order, the keyword 'cef' plus its \
                                     position or be equal to the keyword \
                                     'dimension'"
                    )
                elif ce[0].lower() == "norm":
                    if not ce[1] in stats.order:
                        raise ValueError(
                            f"Norm of order {ce[1]} is not stored\
                                         and hence cannot be evaluated. Please\
                                         provide it to norm_order."
                        )
                    else:
                        index = stats.order.index(ce[1])
                        convergence_estimate[i] = (ce[0], index)
                elif ce[0].lower() == "cef" and not 0 <= ce[1] < len(
                    stats.cef_means
                ):
                    raise ValueError(
                        f"Only {len(stats.cef_means)} custom \
                                     evaluation functions have been provided,\
                                     position {ce[1]} cannot be accessed."
                    )

        if verbose:
            successful_convergence = False

    if verbose:
        print("Computing sample paths of the system")
    for it in range(iterations):
        if verbose and (it + 1) % verbose_step == 0:
            print(f"Sample path {it+1}")
        # Get previous values for convergence check
        if convergence_estimate is not None and (it + 1) >= min_iterations:
            old_convergence_values = []
            for i, ce in enumerate(convergence_estimate):
                old_convergence_values.append(
                    deepcopy(getattr(stats, ce[0] + "_means"))
                )
                if ce[1] >= 0:
                    old_convergence_values[i] = old_convergence_values[i][
                        ce[1]
                    ]

        # Compute a sample path of the solution and add it to the stats
        func = solve_shsvd_single_run(
            flow_function,
            jump_rate_function,
            transition_function,
            initial_value,
            t_span,
            jump_rate_bound,
            jump_times=jump_times,
            jump_time_transition_function=jump_time_transition_function,
            jump_set_function=jump_set_function,
            jump_set_transition_function=jump_set_transition_function,
            t_eval=t_eval,
            flow_args=flow_args,
            jump_rate_args=jump_rate_args,
            transition_args=transition_args,
            jump_time_transition_args=jump_time_transition_args,
            jump_set_args=jump_set_args,
            jump_set_transition_args=jump_set_transition_args,
            check_jump_rate=check_jump_rate,
            rng=rng,
            solve_ivp_method=solve_ivp_method,
            **solve_ivp_options,
        )
        stats.add_function(func)

        if convergence_estimate is not None and (it + 1) >= min_iterations:
            # Get new values for convergence checks
            new_convergence_values = []
            for i, ce in enumerate(convergence_estimate):
                new_convergence_values.append(getattr(stats, ce[0] + "_means"))
                if ce[1] >= 0:
                    new_convergence_values[i] = new_convergence_values[i][
                        ce[1]
                    ]

            old_convergence_values = np.array(old_convergence_values)
            new_convergence_values = np.array(new_convergence_values)

            # Compute differences between previous and new convergence values
            # and evaluate their maximum, mean or integral according to the
            # estimation_type
            difference = np.zeros(old_convergence_values.shape[0])
            if len(estimate_type_dict["max"]) > 0:
                difference[estimate_type_dict["max"]] = np.max(
                    abs(old_convergence_values - new_convergence_values)[
                        estimate_type_dict["max"]
                    ],
                    axis=1,
                )
            if len(estimate_type_dict["mean"]) > 0:
                difference[estimate_type_dict["mean"]] = np.mean(
                    abs(old_convergence_values - new_convergence_values)[
                        estimate_type_dict["mean"]
                    ],
                    axis=1,
                )
            if len(estimate_type_dict["int"]) > 0:
                difference[estimate_type_dict["int"]] = (
                    np.sum(
                        abs(old_convergence_values - new_convergence_values)[
                            estimate_type_dict["int"]
                        ],
                        axis=1,
                    )
                    * stepsize
                )
            if len(estimate_type_dict["time"]) > 0:
                for i in estimate_type_dict["time"]:
                    difference[i[0]] = abs(
                        old_convergence_values[i[0], i[1]]
                        - new_convergence_values[i[0], i[1]]
                    )

            if verbose and (it + 1) % verbose_step == 0:
                print(f"Convergence estimates: {difference}")

            # Check convergence according to the estimation_mode
            if estimation_mode == "all":
                if np.all(difference < tol):
                    if verbose:
                        print(
                            (
                                "Convergence estimate below tolerance with "
                                f"{it+1} sample paths"
                            )
                        )
                        successful_convergence = True
                    break
            elif estimation_mode == "any":
                if np.any(difference < tol):
                    if verbose:
                        print(
                            (
                                "Convergence estimate below tolerance with "
                                f"{it+1} sample paths"
                            )
                        )
                        successful_convergence = True
                    break
            elif estimation_mode == "sum":
                if np.sum(difference) < tol:
                    if verbose:
                        print(
                            (
                                "Convergence estimate below tolerance with "
                                f"{it+1} sample paths"
                            )
                        )
                        successful_convergence = True
                    break
            elif estimation_mode == "weighted_sum":
                if (
                    np.sum(
                        estimation_factors
                        * np.power(difference, estimation_powers)
                    )
                    < tol
                ):
                    if verbose:
                        print(
                            (
                                "Convergence estimate below tolerance with "
                                f"{it+1} sample paths"
                            )
                        )
                        successful_convergence = True
                    break

    if verbose:
        if convergence_estimate is not None:
            if not successful_convergence:
                print(
                    "Maximum number of sample paths reached without \
                      convergence"
                )
        else:
            print("Given number of sample paths reached")

    return stats
