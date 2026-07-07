from scipy.integrate import solve_ivp
import numpy as np
from .function_of_variable_dimension import FunctionOfVariableDimension
from .stats_functions_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import wraps


def solve_hsvd(
    flow_function,
    initial_value,
    t_span,
    jump_times=None,
    jump_time_transition_function=None,
    jump_set_function=None,
    jump_set_transition_function=None,
    t_eval=None,
    flow_args=None,
    jump_time_transition_args=None,
    jump_set_args=None,
    jump_set_transition_args=None,
    jump_rate_args=None,
    transition_args=None,
    jump_rate_bound=None,
    return_args=False,
    solve_ivp_method="RK45",
    **solve_ivp_options,
):
    """
    Solves a hybrid system of variable dimension.

    This function integrates the flow of a system defined by the flow_function
    over a specified time span, accounting for possible transitions defined by
    the transition_function and jump events defined by jump_set_function as
    well as preset jump_times.

    Parameters
    ----------
        flow_function : callable
            The function that defines the flow of the system.
        initial_value : array-like
            The initial state of the system.
        t_span : tuple
            A tuple specifying the start and end times for integration.
        jump_times : list, optional
            A list of times at which jumps occur.
        jump_time_transition_function : callable, optional
            The function that defines the transition behavior at jump times.
        jump_set_function : callable, optional
            A function that defines the conditions for a jump to occur.
        jump_set_transition_function : callable, optional
            A function that defines the transition at jump sets. It is set to
            jump_time_transition_function if none is provided.
        t_eval : array-like, optional
            Specific times at which to store the computed solution.
        flow_args : tuple, optional
            Additional arguments to pass to the flow_function.
        jump_time_transition_args : tuple, optional
            Additional arguments to pass to the jump_time_transition_function.
        jump_set_args : tuple, optional
            Additional arguments to pass to the jump_set_function.
        jump_set_transition_args : tuple, optional
            Additional arguments to pass to the jump_set_transition_function.
        jump_rate_args : tuple, optional
            Additional arguments to pass to the jump_rate_function in
            solve_shsvd_single_run, which might change during jumps. Should
            not be provided when calling solve_hsvd directly.
        transition_args : tuple, optional
            Additional arguments to pass to the transition_function in
            solve_shsvd_single_run, which might change during jumps. Should
            not be provided when calling solve_hsvd directly.
        jump_rate_bound : float, optional
            The upper bound of the jump rate in solve_shsvd_single_run, which
            might change during jumps. Should not be provided when calling
            solve_hsvd directly.
        return_args : bool, optional
            If True, all additional arguments are returned. This option is
            mainly for calls of the method inside solve_shsvd_single_run.
        solve_ivp_method : str, optional
            The method to use for solving the initial value problem
            (default is 'RK45').
        solve_ivp_options : keyword arguments, optional
            Additional options for the solve_ivp function.

    Returns
    -------
        func : FunctionOfVariableDimension
            An object containing the time and value data of the system after
            integration.
        flow_args : tuple, optional
            Additional arguments to pass to the flow_function. Only returned if
            return_args is True.
        jump_rate_args : tuple, optional
            Additional arguments to pass to the jump_rate_function. Only
            returned if return_args is True.
        transition_args : tuple, optional
            Additional arguments to pass to the transition_function. Only
            returned if return_args is True.
        jump_time_transition_args : tuple, optional
            Additional arguments to pass to the jump_time_transition_function.
            Only returned if return_args is True.
        jump_set_args : tuple, optional
            Additional arguments to pass to the jump_set_function. Only
            returned if return_args is True.
        jump_set_transition_args : tuple, optional
            Additional arguments to pass to the jump_set_transition_function.
            Only returned if return_args is True.

    Raises
    ------
        ValueError
            If jump time/set transition function is not given although jump
            times/set are given.
    """
    # Set additional arguments in functions
    if flow_args is not None:

        @wraps(flow_function)
        def flow_function(t, y, fun=flow_function):
            return fun(t, y, *flow_args)

    if jump_time_transition_function is not None:
        _add_attributes(jump_time_transition_function)

        @wraps(jump_time_transition_function)
        def jump_time_transition_function(
            t, y, fun=jump_time_transition_function
        ):
            args = _stack_args(
                "time",
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
                fun.returns_flow_args,
                fun.returns_jump_rate_args,
                fun.returns_transition_args,
                fun.returns_jump_time_transition_args,
                fun.returns_jump_set_args,
                fun.returns_jump_set_transition_args,
                fun.returns_jump_rate_bound,
            )
            return fun(t, y, *args)

    elif jump_times is not None:
        raise ValueError(
            "Jump time transition function must be given if jump times are \
                not None"
        )

    if jump_set_function is not None:
        if jump_set_args is not None:

            @wraps(jump_set_function)
            def jump_set_function(t, y, fun=jump_set_function):
                return fun(t, y, *jump_set_args)

        jump_set_function.terminal = True
        if not hasattr(jump_set_function, "direction"):
            jump_set_function.direction = 0
        if jump_set_transition_function is None:
            if jump_time_transition_function is not None:
                jump_set_transition_function = jump_time_transition_function
            else:
                raise ValueError(
                    "Jump set transition function must be given \
                                 if jump set is not None"
                )
        else:
            _add_attributes(jump_set_transition_function)

            @wraps(jump_set_transition_function)
            def jump_set_transition_function(
                t, y, fun=jump_set_transition_function
            ):
                args = _stack_args(
                    "set",
                    flow_args,
                    jump_rate_args,
                    transition_args,
                    jump_time_transition_args,
                    jump_set_args,
                    jump_set_transition_args,
                    jump_rate_bound,
                    fun.returns_flow_args,
                    fun.returns_jump_rate_args,
                    fun.returns_transition_args,
                    fun.returns_jump_time_transition_args,
                    fun.returns_jump_set_args,
                    fun.returns_jump_set_transition_args,
                    fun.returns_jump_rate_bound,
                )
                return fun(t, y, *args)

    # Initialise solution function
    t0, end_time = t_span

    if isinstance(jump_times, (list, np.ndarray)):
        jump_times = list(jump_times)
        jump_times.sort()
        if jump_times[-1] < t0:
            jump_times = [end_time]
        else:
            while jump_times[0] < t0:
                del jump_times[0]
        tf = min(end_time, jump_times[0])
    else:
        tf = end_time

    func = FunctionOfVariableDimension()

    # Treat special case of jump at beginning
    if t0 == tf:
        values_after_jump = jump_time_transition_function(t0, initial_value)
        (
            initial_value,
            flow_args,
            jump_rate_args,
            transition_args,
            jump_time_transition_args,
            jump_set_args,
            jump_set_transition_args,
            jump_rate_bound,
        ) = _assign_args(
            values_after_jump,
            flow_args,
            jump_rate_args,
            transition_args,
            jump_time_transition_args,
            jump_set_args,
            jump_set_transition_args,
            jump_rate_bound,
            jump_time_transition_function.returns_flow_args,
            jump_time_transition_function.returns_jump_rate_args,
            jump_time_transition_function.returns_transition_args,
            jump_time_transition_function.returns_jump_time_transition_args,
            jump_time_transition_function.returns_jump_set_args,
            jump_time_transition_function.returns_jump_set_transition_args,
            jump_time_transition_function.returns_jump_rate_bound,
        )
        del jump_times[0]
        if len(jump_times) == 0:
            tf = end_time
        else:
            tf = min(end_time, jump_times[0])

    while tf < end_time:
        # Solve IVP with current initial value until next jump time or the jump
        # set is reached
        t_eval_partial = _partial_t_eval(t_eval, t0, tf)
        y = solve_ivp(
            flow_function,
            (t0, tf),
            initial_value,
            t_eval=t_eval_partial,
            events=jump_set_function,
            method=solve_ivp_method,
            **solve_ivp_options,
        )
        func.add_values(y.t, y.y.T, checks=False)

        # Set new initial value to value after jump, specified by the
        # corresponding transition function
        if jump_set_function is not None:
            if y.t_events[0].size > 0:
                t0 = y.t_events[0][0]
                y_jump = y.y_events[0][0]
                values_after_jump = jump_set_transition_function(t0, y_jump)
                (
                    initial_value,
                    flow_args,
                    jump_rate_args,
                    transition_args,
                    jump_time_transition_args,
                    jump_set_args,
                    jump_set_transition_args,
                    jump_rate_bound,
                ) = _assign_args(
                    values_after_jump,
                    flow_args,
                    jump_rate_args,
                    transition_args,
                    jump_time_transition_args,
                    jump_set_args,
                    jump_set_transition_args,
                    jump_rate_bound,
                    jump_set_transition_function.returns_flow_args,
                    jump_set_transition_function.returns_jump_rate_args,
                    jump_set_transition_function.returns_transition_args,
                    jump_set_transition_function.returns_jump_time_transition_args,
                    jump_set_transition_function.returns_jump_set_args,
                    jump_set_transition_function.returns_jump_set_transition_args,
                    jump_set_transition_function.returns_jump_rate_bound,
                )
                if t0 not in y.t:
                    func.add_values(t0, y_jump, checks=False)
            else:
                values_after_jump = jump_time_transition_function(
                    y.t[-1], y.y.T[-1]
                )
                (
                    initial_value,
                    flow_args,
                    jump_rate_args,
                    transition_args,
                    jump_time_transition_args,
                    jump_set_args,
                    jump_set_transition_args,
                    jump_rate_bound,
                ) = _assign_args(
                    values_after_jump,
                    flow_args,
                    jump_rate_args,
                    transition_args,
                    jump_time_transition_args,
                    jump_set_args,
                    jump_set_transition_args,
                    jump_rate_bound,
                    jump_time_transition_function.returns_flow_args,
                    jump_time_transition_function.returns_jump_rate_args,
                    jump_time_transition_function.returns_transition_args,
                    jump_time_transition_function.returns_jump_time_transition_args,
                    jump_time_transition_function.returns_jump_set_args,
                    jump_time_transition_function.returns_jump_set_transition_args,
                    jump_time_transition_function.returns_jump_rate_bound,
                )

                t0 = tf
                del jump_times[0]
                if len(jump_times) == 0:
                    tf = end_time
                else:
                    tf = min(end_time, jump_times[0])

        else:
            values_after_jump = jump_time_transition_function(
                y.t[-1], y.y.T[-1]
            )
            (
                initial_value,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
            ) = _assign_args(
                values_after_jump,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
                jump_time_transition_function.returns_flow_args,
                jump_time_transition_function.returns_jump_rate_args,
                jump_time_transition_function.returns_transition_args,
                jump_time_transition_function.returns_jump_time_transition_args,
                jump_time_transition_function.returns_jump_set_args,
                jump_time_transition_function.returns_jump_set_transition_args,
                jump_time_transition_function.returns_jump_rate_bound,
            )
            t0 = tf
            del jump_times[0]
            if len(jump_times) == 0:
                tf = end_time
            else:
                tf = min(end_time, jump_times[0])

    # Solve IVP until final time
    t_eval_partial = _partial_t_eval(t_eval, t0, tf)
    y = solve_ivp(
        flow_function,
        (t0, tf),
        initial_value,
        t_eval=t_eval_partial,
        events=jump_set_function,
        method=solve_ivp_method,
        **solve_ivp_options,
    )
    func.add_values(y.t, y.y.T, checks=False)

    # If jump set is given, compute solution iteratively until the next time
    # the jump set is reached. Then, set the new value according to the
    # transition function.
    if jump_set_function is not None:
        while y.t_events[0].size > 0:
            t0 = y.t_events[0][0]
            y_jump = y.y_events[0][0]
            values_after_jump = jump_set_transition_function(t0, y_jump)
            (
                initial_value,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
            ) = _assign_args(
                values_after_jump,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
                jump_set_transition_function.returns_flow_args,
                jump_set_transition_function.returns_jump_rate_args,
                jump_set_transition_function.returns_transition_args,
                jump_set_transition_function.returns_jump_time_transition_args,
                jump_set_transition_function.returns_jump_set_args,
                jump_set_transition_function.returns_jump_set_transition_args,
                jump_set_transition_function.returns_jump_rate_bound,
            )
            if t0 not in y.t:
                func.add_values(t0, y_jump, checks=False)

            t_eval_partial = _partial_t_eval(t_eval, t0, tf)
            y = solve_ivp(
                flow_function,
                (t0, tf),
                initial_value,
                t_eval=t_eval_partial,
                events=jump_set_function,
                method=solve_ivp_method,
                **solve_ivp_options,
            )
            func.add_values(y.t, y.y.T, checks=False)

    if return_args:
        return (
            func,
            flow_args,
            jump_rate_args,
            transition_args,
            jump_time_transition_args,
            jump_set_args,
            jump_set_transition_args,
            jump_rate_bound,
        )
    else:
        return func


def solve_shsvd_single_run(
    flow_function,
    jump_rate_function,
    transition_function,
    initial_value,
    t_span,
    jump_rate_bound=None,
    jump_times=None,
    jump_time_transition_function=None,
    jump_set_function=None,
    jump_set_transition_function=None,
    t_eval=None,
    flow_args=None,
    jump_rate_args=None,
    transition_args=None,
    jump_time_transition_args=None,
    jump_set_args=None,
    jump_set_transition_args=None,
    check_jump_rate=False,
    solve_ivp_method="RK45",
    **solve_ivp_options,
):
    """
    Executes a single run of the stochastic hybrid system dynamics.

    This function runs the integration of the system defined by the
    flow_function as well as forced transitions while computing an execution
    of the random variable given by the jump_rate_function and transitions
    specified by transition_function.

    Parameters
    ----------
        flow_function : callable
            The function that defines the flow of the system.
        jump_rate_function : callable
            The function that defines the jump rate of the system.
        transition_function : callable
            The function that defines the transition behavior.
        initial_value : array-like
            The initial state of the system.
        t_span : tuple
            A tuple specifying the start and end times for integration.
        jump_rate_bound : float, optional
            The upper bound of the jump rate. If none is provided, it will be
            estimated during simulation.
        jump_times : list, optional
            A list of times at which jumps occur.
        jump_time_transition_function : callable, optional
            A function that defines the transition at jump occurrences. It is
            set to transition_function if jump_times are given, but no
            jump_time_transition_function.
        jump_set_function : callable, optional
            A function that defines the conditions for a jump to occur.
        jump_set_transition_function : callable, optional
            A function that defines the transition behavior for jump sets. It
            isbset to transition_function if a jump_set_function is given, but
            no jump_time_transition_function.
        t_eval : array-like, optional
            Specific times at which to store the computed solution.
        flow_args : tuple, optional
            Additional arguments to pass to the flow_function.
        jump_rate_args : tuple, optional
            Additional arguments to pass to the jump_rate_function.
        transition_args : tuple, optional
            Additional arguments to pass to the transition_function.
        jump_time_transition_args : tuple, optional
            Additional arguments to pass to the jump_time_transition_function.
        jump_set_args : tuple, optional
            Additional arguments to pass to the jump_set_function.
        jump_set_transition_args : tuple, optional
            Additional arguments to pass to the jump_set_transition_function.
        check_jump_rate : bool, optional
            If True, check if jump_rate_bound was violated in each step.
            Automatically set to True, if no jump_rate_bound is provided.
        solve_ivp_method : str, optional
            The method to use for solving the initial value problem (default is
            'RK45').
        solve_ivp_options : keyword arguments, optional
            Additional options for the solve_ivp function.

    Returns:
        func : FunctionOfVariableDimension
            An object containing the time and value data of the system after
            integration during this single run.
    """
    # Set additional arguments in functions
    if jump_rate_args is not None:

        @wraps(jump_rate_function)
        def jump_rate_function(t, y, fun=jump_rate_function):
            return fun(t, y, *jump_rate_args)

    _add_attributes(transition_function)

    if transition_args is not None:

        @wraps(transition_function)
        def transition_function(t, y, fun=transition_function):
            args = _stack_args(
                "stochastic",
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
                fun.returns_flow_args,
                fun.returns_jump_rate_args,
                fun.returns_transition_args,
                fun.returns_jump_time_transition_args,
                fun.returns_jump_set_args,
                fun.returns_jump_set_transition_args,
                fun.returns_jump_rate_bound,
            )
            return fun(t, y, *args)

    # Set transition function if none are provided
    if jump_times is not None and jump_time_transition_function is None:
        jump_time_transition_function = transition_function

    if jump_set_function is not None and jump_set_transition_function is None:
        jump_set_transition_function = transition_function

    # Initialise solution
    t0, end_time = t_span

    func = FunctionOfVariableDimension()

    # Compute next jump time of homogeneous Poisson process with rate
    # jump_rate_bound
    if jump_rate_bound is None or jump_rate_bound == 0:
        jump_rate_bound = max(jump_rate_function(t0, initial_value), 1e-10)
        check_jump_rate = True
    r = random.random()
    t = min(t0 - np.log(r) / jump_rate_bound, end_time)

    store_t0 = True

    while t <= end_time:
        # Solve solution until next possible jump
        (
            func_partial,
            new_flow_args,
            new_jump_rate_args,
            new_transition_args,
            new_jump_time_transition_args,
            new_jump_set_args,
            new_jump_set_transition_args,
            new_jump_rate_bound,
        ) = solve_hsvd(
            flow_function,
            initial_value,
            (t0, t),
            jump_times=jump_times,
            jump_time_transition_function=jump_time_transition_function,
            jump_set_function=jump_set_function,
            jump_set_transition_function=jump_set_transition_function,
            t_eval=t_eval,
            flow_args=flow_args,
            jump_time_transition_args=jump_time_transition_args,
            jump_set_args=jump_set_args,
            jump_set_transition_args=jump_set_transition_args,
            jump_rate_args=jump_rate_args,
            transition_args=transition_args,
            jump_rate_bound=jump_rate_bound,
            return_args=True,
            solve_ivp_method=solve_ivp_method,
            **solve_ivp_options,
        )

        # Check if jump_rate_bound was violated, if necessary
        if new_jump_rate_bound > jump_rate_bound:
            jump_rate_bound = new_jump_rate_bound
            t = min(t0 - np.log(r) / jump_rate_bound, end_time)
            continue

        if check_jump_rate:
            max_jump_rate = max(
                [
                    jump_rate_function(time, value)
                    for time, value in zip(
                        func_partial.times, func_partial.values
                    )
                ]
            )
            if max_jump_rate > jump_rate_bound:
                jump_rate_bound = max_jump_rate
                t = min(t0 - np.log(r) / jump_rate_bound, end_time)
                continue

        flow_args = new_flow_args
        jump_rate_args = new_jump_rate_args
        transition_args = new_transition_args
        jump_time_transition_args = new_jump_time_transition_args
        jump_set_args = new_jump_set_args
        jump_set_transition_args = new_jump_set_transition_args

        if not store_t0:
            func_partial.delete_values(times=[t0])

        if t == end_time:
            break

        last_time = func_partial.times[-1]
        last_value = func_partial.values[-1]

        t0 = t

        # Check with acceptance-rejection method if possible jump is a jump of
        # the nonhomogeneous Poisson process
        s = random.random()
        if s <= jump_rate_function(last_time, last_value) / jump_rate_bound:
            values_after_jump = transition_function(last_time, last_value)
            (
                initial_value,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                jump_rate_bound,
            ) = _assign_args(
                values_after_jump,
                flow_args,
                jump_rate_args,
                transition_args,
                jump_time_transition_args,
                jump_set_args,
                jump_set_transition_args,
                new_jump_rate_bound,
                transition_function.returns_flow_args,
                transition_function.returns_jump_rate_args,
                transition_function.returns_transition_args,
                transition_function.returns_jump_time_transition_args,
                transition_function.returns_jump_set_args,
                transition_function.returns_jump_set_transition_args,
                transition_function.returns_jump_rate_bound,
            )
            store_t0 = True
        else:
            initial_value = last_value
            if t_eval is not None:
                if last_time not in t_eval:
                    func_partial.delete_values()
                    store_t0 = False

        # Concatenate existing solution with new one
        func = func.concatenate(func_partial)

        # Compute next jump of the homogeneous Poisson process
        r = random.random()
        t = min(t0 - np.log(r) / jump_rate_bound, end_time)

    func = func.concatenate(func_partial)

    return func


def solve_shsvd(
    flow_function,
    jump_rate_function,
    transition_function,
    initial_value,
    t_span,
    jump_rate_bound=None,
    jump_times=None,
    jump_time_transition_function=None,
    jump_set_function=None,
    jump_set_transition_function=None,
    custom_evaluation_functions=None,
    stepsize=1,
    t_eval=None,
    norm_order=2,
    collect_dimension=True,
    collect_jump_times=False,
    collect_dimension_jump_times=False,
    flow_args=None,
    jump_rate_args=None,
    transition_args=None,
    jump_time_transition_args=None,
    jump_set_args=None,
    jump_set_transition_args=None,
    cef_args=None,
    check_jump_rate=False,
    iterations=500,
    tol=1e-5,
    min_iterations=10,
    max_iterations=1000,
    convergence_estimate=None,
    convergence_estimate_type="max",
    estimation_mode="all",
    estimation_weights=None,
    solve_ivp_method="RK45",
    verbose=False,
    verbose_step=1,
    **solve_ivp_options,
):
    """
    Solves a stochastic hybrid system with multiple iterations.

    This function uses multiple runs to compute the behavior of the system
    defined by the flow_function, accounting for jumps and transitions,
    while checking for convergence based on specified criteria.

    Parameters
    ----------
        flow_function : callable
            The function that defines the flow of the system.
        jump_rate_function : callable
            The function that defines the jump rate of the system.
        transition_function : callable
            The function that defines the transition behavior.
        initial_value : array-like
            The initial state of the system.
        t_span : tuple
            A tuple specifying the start and end times for integration.
        jump_rate_bound : float, optional
            The upper bound of the jump rate. If none is provided, it will be
            estimated during simulation.
        jump_times : list, optional
            A list of times at which jumps occur.
        jump_time_transition_function : callable, optional
            A function that defines the transition at jump occurrences.
        jump_set_function : callable, optional
            A function that defines the conditions for a jump to occur.
        jump_set_transition_function : callable, optional
            A function that defines the transition behavior for jump sets.
        custom_evaluation_functions : list, optional
            Custom functions for additional evaluations during the run.
        stepsize : float, optional
            The step size for evaluations (default is 1).
        t_eval : array-like, optional
            Specific times at which to store the computed solution.
        norm_order : int, list or tuple, optional
            The order of the norm(s) to compute (default is 2).
        collect_dimension : bool, optional
            If True, compute mean and variance of system dimension.
        collect_jump_times : bool, optional
            If True, collect jump times during evaluation.
        collect_dimension_jump_times : bool, optional
            If True, collect dimension jump times during evaluation.
        flow_args : tuple, optional
            Additional arguments to pass to the flow_function.
        jump_rate_args : tuple, optional
            Additional arguments to pass to the jump_rate_function.
        transition_args : tuple, optional
            Additional arguments to pass to the transition_function.
        jump_time_transition_args : tuple, optional
            Additional arguments to pass to the jump_time_transition_function.
        jump_set_args : tuple, optional
            Additional arguments to pass to the jump_set_function.
        jump_set_transition_args : tuple, optional
            Additional arguments to pass to the jump_set_transition_function.
        cef_args : tuple, optional
            Additional arguments for custom evaluation functions.
        check_jump_rate : bool, optional
            If True, check if jump_rate_bound was violated in each step.
            Automatically set to True, if no jump_rate_bound is provided.
        iterations : int, optional
            The number of iterations to perform (default is 500).
        tol : float or list, optional
            The tolerance(s) for convergence (default is 1e-5). If multiple
            convergence_estimates are provided, tolerance can either be a
            number (one tolerance for all estimates) or a list of tolerances
            with the same length as convergence_estimates.
        min_iterations : int, optional
            Minimum number of iterations before convergence is checked
            (default is 10).
        max_iterations : int, optional
            Maximum number of iterations (default is 1000).
        convergence_estimate : list or tuple, optional
            Specifications for convergence estimates to check. Options are the
            dimension, any norm which is stored and any custom evaluation
            function that is provided. The elements of the list/tuple must be:
            'dimension', ('norm', norm_order) or ('cef', cef_number),
            where norm_order is the order of the norm to be evaluated and
            cef_number specifies the number of the custom evaluation function
            to be evaluated as the position in the custom evaluation function
            list.
        convergence_estimate_type : str or list, optional
            Type(s) of convergence estimates (default is 'max').
            Defines what has to be below the tolerance for convergence:
            'max': maximum
            'mean': mean
            'int': integral
            ('time', t): value at specific time t which must be in t_eval
        estimation_mode : str, optional
            Mode for convergence estimation. Options are:
            'all': All estimates must be below the tolerance(s)
            'any': Any estimate must be below the tolerance
            'sum': Sum of estimates must be below tolerance
            'weighted_sum': Weighted sum of estimates must be below
            tolerance. In this case, estimation_weights must be given.
        estimation_weights : array-like, optional
            Weights for estimating convergence (if estimation_mode is
            'weighted_sum'). Must be the same length as convergence_estimate.
            The first value specifies the factor of the component and the
            second the exponent.
        solve_ivp_method : str, optional
            The method to use for solving the initial value problem (default
            is 'RK45').
        verbose : bool, optional
            If True, print convergence information during execution.
        verbose_step : int, optional
            The step interval for verbose output (default is 1).
        solve_ivp_options : keyword arguments, optional
            Additional options for the solve_ivp function.

    Returns
    -------
        stats : StatsFunctionsOfVariableDimension
            An object containing statistics of the system's behavior after all
            iterations.
    """
    if verbose:
        print("Initialising")

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

                        def cef_with_args(values, fun=cef, args=cef_arg):
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
                elif ce[0].lower() == "cef" and ce[1] > len(stats.cef_means):
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


def _partial_t_eval(t_eval, t0, tf):
    """
    Prepares the evaluation times by filtering them within a specified range.

    This function ensures that the evaluation times fall between the provided
    start (t0) and end (tf) times, including both boundaries if not already
    present.

    Parameters:
        t_eval: array-like
            The specified evaluation times.
        t0: float
            The start of the time range.
        tf: float
            The end of the time range.

    Returns:
        list: The filtered evaluation times within the range [t0, tf].
    """
    if t_eval is None:
        return None

    t_eval = np.array(t_eval)
    t_eval_partial = t_eval[(t_eval >= t0) & (t_eval <= tf)].tolist()

    if t0 not in t_eval_partial:
        t_eval_partial.insert(0, t0)

    if tf not in t_eval_partial:
        t_eval_partial.append(tf)

    t_eval_partial = np.array(t_eval_partial)

    return t_eval_partial


def _assign_args(
    values,
    flow_args,
    jump_rate_args,
    transition_args,
    jump_time_transition_args,
    jump_set_args,
    jump_set_transition_args,
    jump_rate_bound,
    returns_flow_args,
    returns_jump_rate_args,
    returns_transition_args,
    returns_jump_time_transition_args,
    returns_jump_set_args,
    returns_jump_set_transition_args,
    returns_jump_rate_bound,
):
    """
    Assigns values to corresponding arguments based on the returns specified.

    This function processes the provided values and assigns them to the
    relevant arguments, returning the updated values. It checks which
    arguments should be returned based on the flags, allowing for flexible
    handling of the function's state.

    Parameters
    ----------
        values : list or array-like
            A list of values to be assigned to the corresponding arguments.
        flow_args : tuple
            Current values of additional arguments for the flow_function.
        jump_rate_args : tuple
            Current values of additional arguments for the jump_rate_function.
        transition_args : tuple
            Current values of additional arguments for the transition_function.
        jump_time_transition_args : tuple
            Current values of additional arguments for the
            jump_time_transition_function.
        jump_set_args : tuple
            Current values of additional arguments for the jump_set_function.
        jump_set_transition_args : tuple
            Current values of additional arguments for the
            jump_set_transition_function.
        jump_rate_bound : float
            Current value of the upper bound for the jump rate.
        returns_flow_args : bool
            Flag indicating whether flow_args have been returned.
        returns_jump_rate_args : bool
            Flag indicating whether jump_rate_args have been returned.
        returns_transition_args : bool
            Flag indicating whether transition_args have been returned.
        returns_jump_time_transition_args : bool
            Flag indicating whether jump_time_transition_args have been
            returned.
        returns_jump_set_args : bool
            Flag indicating whether jump_set_args have been returned.
        returns_jump_set_transition_args : bool
            Flag indicating whether jump_set_transition_args have been
            returned.
        returns_jump_rate_bound : bool
            Flag indicating whether jump_rate_bound has been returned.

    Returns
    -------
        state : np.ndarray
            The updated state of the system.
        flow_args : tuple
            Updated values of additional arguments for the flow_function.
        jump_rate_args : tuple
            Updated values of additional arguments for the jump_rate_function.
        transition_args : tuple
            Updated values of additional arguments for the transition_function.
        jump_time_transition_args : tuple
            Updated values of additional arguments for the
            jump_time_transition_function.
        jump_set_args : tuple
            Updated values of additional arguments for the jump_set_function.
        jump_set_transition_args : tuple
            Updated values of additional arguments for the
            jump_set_transition_function.
        jump_rate_bound : float
            Updated value of the upper bound for the jump rate.
    """
    returns = [
        returns_flow_args,
        returns_jump_rate_args,
        returns_transition_args,
        returns_jump_time_transition_args,
        returns_jump_set_args,
        returns_jump_time_transition_args,
        returns_jump_rate_bound,
    ]

    if any(returns):
        state = values[0]
        counter = 1
        if returns_flow_args:
            flow_args = values[counter]
            counter += 1
        if returns_jump_rate_args:
            jump_rate_args = values[counter]
            counter += 1
        if returns_transition_args:
            transition_args = values[counter]
            counter += 1
        if returns_jump_time_transition_args:
            jump_time_transition_args = values[counter]
            counter += 1
        if returns_jump_set_args:
            jump_set_args = values[counter]
            counter += 1
        if returns_jump_set_transition_args:
            jump_set_transition_args = values[counter]
            counter += 1
        if returns_jump_rate_bound:
            jump_rate_bound = values[counter]
            counter += 1
    else:
        state = values

    return (
        state,
        flow_args,
        jump_rate_args,
        transition_args,
        jump_time_transition_args,
        jump_set_args,
        jump_set_transition_args,
        jump_rate_bound,
    )


def _stack_args(
    jump_type,
    flow_args,
    jump_rate_args,
    transition_args,
    jump_time_transition_args,
    jump_set_args,
    jump_set_transition_args,
    jump_rate_bound,
    returns_flow_args,
    returns_jump_rate_args,
    returns_transition_args,
    returns_jump_time_transition_args,
    returns_jump_set_args,
    returns_jump_set_transition_args,
    returns_jump_rate_bound,
):
    """
    Stacks arguments based on the jump type and specified return flags.

    This function constructs a tuple of arguments to be passed to the
    respective functions based on the jump type. It combines the necessary
    arguments, ensuring that only the relevant ones are included based on
    the provided flags.

    Parameters
    ----------
        jump_type : str
            The type of jump ('time', 'set', or 'stochastic') to determine
            which arguments to stack and in which order.
        flow_args : tuple
            Additional arguments for the flow_function.
        jump_rate_args : tuple
            Additional arguments for the jump_rate_function.
        transition_args : tuple
            Additional arguments for the transition_function.
        jump_time_transition_args : tuple
            Additional arguments for the jump_time_transition_function.
        jump_set_args : tuple
            Additional arguments for the jump_set_function.
        jump_set_transition_args : tuple
            Additional arguments for the jump_set_transition_function.
        jump_rate_bound : float
            The upper bound for the jump rate.
        returns_flow_args : bool
            Flag indicating whether flow_args will be returned.
        returns_jump_rate_args : bool
            Flag indicating whether jump_rate_args will be returned.
        returns_jump_time_transition_args : bool
            Flag indicating whether jump_time_transition_args will be returned.
        returns_jump_set_args : bool
            Flag indicating whether jump_set_args will be returned.
        returns_jump_set_transition_args : bool
            Flag indicating whether jump_set_transition_args will be returned.
        returns_jump_rate_bound : bool
            Flag indicating whether jump_rate_bound will be returned.

    Returns
    -------
        args : tuple
            A tuple of stacked arguments based on the specified jump type
            and return flags. It combines the arguments for the transition type
            and the arguments which will be returned in one tuple.
    """
    # Assume that flow_args etc. are not None, if returns_flow_args is True
    args = ()
    if jump_type == "time" and jump_time_transition_args is not None:
        args += jump_time_transition_args
    if jump_type == "set" and jump_set_transition_args is not None:
        args += jump_set_transition_args
    if jump_type == "stochastic" and transition_args is not None:
        args += transition_args
    if returns_flow_args:
        args += flow_args
    if returns_jump_rate_args:
        args += jump_rate_args
    if returns_jump_time_transition_args and jump_type != "time":
        args += jump_time_transition_args
    if returns_jump_set_args:
        args += jump_set_args
    if returns_jump_set_transition_args and jump_type != "set":
        args += jump_set_transition_args
    if returns_jump_rate_bound:
        args += (jump_rate_bound,)

    return args


def _add_attributes(func):
    """
    Adds default attributes to a given function for tracking return flags.

    This function ensures that the specified function has certain attributes
    that indicate whether it returns additional arguments. If the attributes
    do not exist, they are initialized to False.

    Parameters
    ----------
        func : callable
            The function to which attributes will be added.

    Returns
    -------
        None
    """
    if not hasattr(func, "returns_flow_args"):
        func.returns_flow_args = False
    if not hasattr(func, "returns_jump_rate_args"):
        func.returns_jump_rate_args = False
    if not hasattr(func, "returns_transition_args"):
        func.returns_transition_args = False
    if not hasattr(func, "returns_jump_time_transition_args"):
        func.returns_jump_time_transition_args = False
    if not hasattr(func, "returns_jump_set_args"):
        func.returns_jump_set_args = False
    if not hasattr(func, "returns_jump_set_transition_args"):
        func.returns_jump_set_transition_args = False
    if not hasattr(func, "returns_jump_rate_bound"):
        func.returns_jump_rate_bound = False


if __name__ == "__main__":

    def flow_function(t, y):
        """
        Defines the flow of a simple linear system.

        This function returns the derivative of the system state based on the
        current state. In this case, it represents a linear decay.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.

        Returns:
            array_like: The derivative of the state, which is negative of y.
        """
        return -y

    def harmonic_oscillator(t, y, a, b):
        """
        Defines the dynamics of a harmonic oscillator.

        This function computes the derivatives of the state variables of a
        harmonic oscillator based on the equations of motion. The 1D-equation
        corresponds to just holding the actual position of the system.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system, where y[0] is the position and
                y[1] is the velocity.
            a: float
                The coefficient for the damping term.
            b: float
                The coefficient for the spring force.

        Returns:
            dy: list
                The derivatives of the state variables.
        """
        if len(y) == 1:
            dy = [0]
        elif len(y) == 2:
            dy = [y[1], -b * y[1] - a * y[0]]
        else:
            raise ValueError("Not a possible dimension")

        return dy

    def jump_rate_function(t, y):
        """
        Defines the jump rate for a stochastic process.

        This function returns the rate at which jumps occur based on the
        current state of the system.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.

        Returns:
            prob: float
                The jump rate, which is inversely proportional to the norm
                of y.
        """
        norm = np.linalg.norm(np.array(y))
        prob = min(1, 1 / norm)

        return prob

    def ho_jump_rate_function(t, y):
        """
        Defines the jump rate for a harmonic oscillator.

        This function returns the jump rate, given as the absolute value of the
        current position of the harmonic oscillator.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current position of the oscillator.

        Returns:
            float: The absolute value of the position, representing the jump
                rate.
        """
        return np.abs(y[0])  # np.linalg.norm(y[0])

    def transition_function(t, y):
        """
        Defines the transition behavior of the system.

        This function determines the next state of the system based on the
        current state. If the current state is 2D, then the next state is 1D
        with both values added. If it is 1D, then the next state is 2D with
        values 1/3*y and 2/3*y.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.

        Returns:
            y_new: list
                The new state after the transition.
        """
        if len(y) == 2:
            y_new = [y[0] + y[1]]
        elif len(y) == 1:
            y_new = [1 / 3 * y[0], 2 / 3 * y[0]]
        else:
            raise ValueError("Not the right dimension")

        return y_new

    def ho_transition_function(t, y, a):
        """
        Defines the transition behavior for the harmonic oscillator.

        This function computes the new state of the harmonic oscillator after
        a jump. It corresponds to holding the oscillator if it oscillates and
        releasing it with an initial velocity of a if it is held.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.
            a: float
                The initial velocity of the oscillator after releasing it.

        Returns:
            y_new: list
                The new state after the transition.
        """
        if len(y) == 1:
            y_new = [y[0], a]
        elif len(y) == 2:
            y_new = [y[0]]
        else:
            raise ValueError("Not a possible dimension")

        return y_new

    def ho_transition_function_2(t, y, c):
        """
        Defines the transition behavior for the harmonic oscillator.

        This function computes the new state of the harmonic oscillator after a
        jump. It corresponds to holding the oscillator if it oscillates and
        releasing it with an initial velocity proportional to the position with
        proportionality constant c if it is held.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.
            c: float
                A parameter that influences the transition behavior.

        Returns:
            list: The new state after the transition.
        """
        if len(y) == 1:
            y_new = [y[0], c * y[0]]
        elif len(y) == 2:
            y_new = [y[0]]
        else:
            raise ValueError("Not a possible dimension")

        return y_new

    def ho_energy(values, mass):
        """
        Computes the energy of a harmonic oscillator given its state.

        This function computes the energy based on the positions and velocities
        of the oscillator and the specified mass.

        Parameters:
            values: list or array-like
                A list of positions and (possibly) velocities of the
                oscillator.
            mass: float
                The mass of the oscillator.

        Returns:
            energy: list
                A list of energy values corresponding to each state.
        """
        squared_values = [np.array(value) ** 2 for value in values]
        energy = [1 / 2 * mass * np.sum(value) for value in squared_values]

        return energy

    def ho_first_coordinate(values):
        """
        Extracts the first coordinate from a list of coordinates.

        This function takes a list of coordinates and returns a list containing
        only the first coordinate from each entry.

        Parameters:
            values: list of lists or array-like
                A list of lists, where each inner list represents a state with
                multiple coordinates.

        Returns:
            first_coordinate: list
                A list of the first coordinates from the input values.
        """
        first_coordinate = [value[0] for value in values]

        return first_coordinate

    def jump_set_function(t, y, b):
        """
        Defines the jump set condition for the system.

        This function specifies the condition under which a jump occurs in the
        system based on the current state and a parameter. A jump occurs, when
        the first coordinate equals the parameter b.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.
            b: float
                The threshold parameter for the first coordinate of a jump
                occuring.

        Returns:
            float: The difference between the first coordinate and the jump
                set.
        """
        return y[0] + b

    def jump_set_transition_function(t, y, c):
        """
        Defines the transition behavior at jump occurrences.

        This function specifies the new state of the system when a jump occurs.
        After the jump, it is in state c.

        Parameters:
            t: float
                The current time (not used in this implementation).
            y: array-like
                The current state of the system.
            c: float
                The state after the jump.

        Returns:
            list: The new state after the jump transition.
        """
        return [c]

    y0 = np.array([1, 2])
    t_span = (0, 10)

    func_det = solve_hsvd(
        flow_function, y0, t_span, [-1, 1, 5, 7], transition_function
    )
    func_det.plot_coordinates(
        "Coordinate plot",
        True,
        ["first", "second"],
        "Time",
        "Value",
        True,
        grid=True,
        show_jump_times=True,
        show_dimension_jump_times=True,
    )
    func_det.plot_norm(
        order=[1, 2],
        title="Norm plot",
        legend=True,
        grid=True,
        show_dimension_jump_times=True,
    )

    func_stoch = solve_shsvd_single_run(
        flow_function, jump_rate_function, transition_function, y0, t_span, 1
    )  # , jump_times=[-1, 1, 5, 7])
    func_stoch.plot_coordinates(
        "Coordinate plot",
        True,
        ["first", "second"],
        "Time",
        "Value",
        True,
        grid=True,
        show_jump_times=True,
        show_dimension_jump_times=True,
    )
    func_stoch.plot_norm(
        order=[1, 2],
        title="Norm plot",
        legend=True,
        grid=True,
        show_dimension_jump_times=True,
    )

    stats = solve_shsvd(
        flow_function,
        jump_rate_function,
        transition_function,
        y0,
        t_span,
        1,
        norm_order=[1, 2],
        collect_jump_times=True,
    )

    t_end = 100
    y0 = [1, 0]
    t_span = (0, t_end)
    jump_rate_bound = np.linalg.norm(y0, ord=2)
    a = 1
    b = 0
    transition_args = (-0.3,)
    jump_times = np.arange(8.5, 100, 8.5)
    t_eval = np.linspace(0, t_end, 2 * t_end + 1)
    jump_set_args = (0.1,)
    jump_set_transition_args = (0.5,)

    jump_times = [20, 60]

    det_oscillator = solve_hsvd(
        harmonic_oscillator,
        y0,
        t_span,
        jump_time_transition_function=ho_transition_function_2,
        jump_set_function=jump_set_function,
        jump_set_transition_function=jump_set_transition_function,
        flow_args=(a, b),
        jump_time_transition_args=transition_args,
        jump_times=jump_times,
        t_eval=t_eval,
        jump_set_args=jump_set_args,
        jump_set_transition_args=jump_set_transition_args,
    )
    fig = det_oscillator.plot_coordinates()
    plt.figure(fig)
    plt.grid()
    plt.tight_layout()
    plt.savefig("Harmonic_oscillator_hold_and_push.png")

    random.seed(42)
    oscillator = solve_shsvd_single_run(
        harmonic_oscillator,
        ho_jump_rate_function,
        ho_transition_function_2,
        y0,
        t_span,
        1,
        jump_times=jump_times,
        flow_args=(a, b),
        transition_args=transition_args,
        t_eval=t_eval,
    )
    oscillator.plot_coordinates()
    print(oscillator.norm([1, 10, 50, 100]))
    # random.seed(42)
    oscillator_stats = solve_shsvd(
        harmonic_oscillator,
        ho_jump_rate_function,
        ho_transition_function_2,
        y0,
        t_span,
        jump_rate_bound,
        t_eval=t_eval,
        jump_times=jump_times,
        norm_order=[1, 2],
        collect_jump_times=True,
        flow_args=(a, b),
        transition_args=transition_args,
        iterations=50,
        custom_evaluation_functions=[ho_energy, ho_first_coordinate],
        cef_args=[(2,), None],
        convergence_estimate=[("norm", 2), ("norm", 2)],
        convergence_estimate_type=["max", "mean"],
        estimation_mode="sum",
        max_iterations=500,
        tol=1e-3,
        verbose=True,
        verbose_step=10,
    )
    oscillator_stats.plot_norm()
    oscillator_stats.plot_jump_times()
    oscillator_stats.plot_inter_jump_times()

    # random.seed(42)
    oscillator_stats_2 = solve_shsvd(
        harmonic_oscillator,
        ho_jump_rate_function,
        ho_transition_function_2,
        y0,
        t_span,
        jump_rate_bound,
        t_eval=t_eval,
        jump_times=jump_times,
        norm_order=[1, 2],
        collect_jump_times=True,
        flow_args=(a, b),
        transition_args=transition_args,
        iterations=50,
        custom_evaluation_functions=[ho_energy, ho_first_coordinate],
        cef_args=[(2,), None],
        convergence_estimate=[("norm", 2), ("norm", 2)],
        convergence_estimate_type=["max", "mean"],
        estimation_mode="sum",
        max_iterations=500,
        tol=1e-3,
        verbose=True,
        verbose_step=10,
    )
    oscillator_stats_2.plot_norm()
    oscillator_stats_2.plot_jump_times()
    oscillator_stats_2.plot_inter_jump_times()

    combined_oscillator_stats = oscillator_stats + oscillator_stats_2
    combined_oscillator_stats.plot_norm()
    combined_oscillator_stats.plot_jump_times()
    combined_oscillator_stats.plot_inter_jump_times()

    # print(oscillator_stats == oscillator_stats_2)
