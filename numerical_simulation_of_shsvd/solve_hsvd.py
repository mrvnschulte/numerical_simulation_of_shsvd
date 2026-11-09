"""
Provides a solver for hybrid systems of variable dimension.

The solver integrates continuous dynamics between prescribed jump times
and jump-set events and applies the corresponding transition functions.
"""

from __future__ import annotations
from functools import wraps
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from .function_of_variable_dimension import FunctionOfVariableDimension
from ._utils import _add_attributes, _assign_args, _partial_t_eval, _stack_args


Args = tuple[Any, ...] | None
HsvdResult = (
    FunctionOfVariableDimension
    | tuple[
        FunctionOfVariableDimension,
        Args,
        Args,
        Args,
        Args,
        Args,
        Args,
        float | None,
    ]
)


def solve_hsvd(
    flow_function: Callable,
    initial_value: ArrayLike,
    t_span: tuple[float, float],
    jump_times: ArrayLike | None = None,
    jump_time_transition_function: Callable | None = None,
    jump_set_function: Callable | None = None,
    jump_set_transition_function: Callable | None = None,
    t_eval: ArrayLike | None = None,
    flow_args: Args = None,
    jump_time_transition_args: Args = None,
    jump_set_args: Args = None,
    jump_set_transition_args: Args = None,
    jump_rate_args: Args = None,
    transition_args: Args = None,
    jump_rate_bound: float | None = None,
    return_args: bool = False,
    solve_ivp_method: str = "RK45",
    **solve_ivp_options: Any,
) -> HsvdResult:
    """
    Solves a hybrid system of variable dimension.

    The continuous dynamics are integrated with scipy.integrate.solve_ivp.
    Transitions may occur at prescribed jump times or when a jump set event
    is detected. A transition may change the state dimension and optionally
    update additional arguments used by the solver.

    Parameters
    ----------
    flow_function : callable
        Function defining the continuous dynamics. It is called as
        flow_function(t, y, *flow_args) and returns the state derivatives.
    initial_value : array-like
        Initial state of the system.
    t_span : tuple of float
        Start and end times of the integration.
    jump_times : list or np.ndarray or None, optional
        Prescribed jump times. Times before the start of t_span are ignored.
        A nonempty list or array must be provided when using this option.
        The default is None.
    jump_time_transition_function : callable or None, optional
        Transition function applied at prescribed jump times. It must be
        provided if jump_times is not None. The default is None.
    jump_set_function : callable or None, optional
        Event function defining the jump set. It is called as
        jump_set_function(t, y, *jump_set_args). Its zeros are treated as
        terminal events. The default is None.
    jump_set_transition_function : callable or None, optional
        Transition function applied when the jump set is reached. If None,
        jump_time_transition_function is used when available. The default
        is None.
    t_eval : array-like or None, optional
        Times at which the solution is stored. Integration segment
        boundaries are included as needed. If None, solve_ivp chooses the
        output times. The default is None.
    flow_args : tuple or None, optional
        Additional arguments passed to flow_function. The default is None.
    jump_time_transition_args : tuple or None, optional
        Additional arguments for jump_time_transition_function. The default
        is None.
    jump_set_args : tuple or None, optional
        Additional arguments passed to jump_set_function. The default is
        None.
    jump_set_transition_args : tuple or None, optional
        Additional arguments for jump_set_transition_function. The default
        is None.
    jump_rate_args : tuple or None, optional
        Additional arguments for the jump rate function used by
        solve_shsvd_single_run. They can be updated by transition functions.
        The default is None.
    transition_args : tuple or None, optional
        Additional arguments for the stochastic transition function used by
        solve_shsvd_single_run. They can be updated by transition functions.
        The default is None.
    jump_rate_bound : float or None, optional
        Jump rate bound used by solve_shsvd_single_run. It can be updated by
        transition functions. The default is None.
    return_args : bool, optional
        If True, return the current additional arguments and jump rate bound
        together with the solution. The default is False.
    solve_ivp_method : str, optional
        Integration method passed to solve_ivp. The default is 'RK45'.
    **solve_ivp_options : dict
        Additional keyword arguments passed to solve_ivp.

    Returns
    -------
    func : FunctionOfVariableDimension
        Solution represented by time-value pairs.
    flow_args : tuple or None, optional
        Current flow arguments. Returned only if return_args is True.
    jump_rate_args : tuple or None, optional
        Current jump rate arguments. Returned only if return_args is True.
    transition_args : tuple or None, optional
        Current stochastic transition arguments. Returned only if
        return_args is True.
    jump_time_transition_args : tuple or None, optional
        Current time transition arguments. Returned only if return_args is
        True.
    jump_set_args : tuple or None, optional
        Current jump set arguments. Returned only if return_args is True.
    jump_set_transition_args : tuple or None, optional
        Current jump set transition arguments. Returned only if return_args
        is True.
    jump_rate_bound : float or None, optional
        Current jump rate bound. Returned only if return_args is True.

    Raises
    ------
    ValueError
        If jump_times is provided without a time transition function, or if
        a jump set function is provided without an available transition
        function.

    Notes
    -----
    Transition functions receive the current time and state, followed by
    their additional arguments. By default, they return the new state.
    Return-flag attributes can be set on a transition function to indicate
    that it also returns updated additional arguments.

    A prescribed jump at the initial time is applied before integration.
    Prescribed jumps at the final time are not applied.
    """
    # Set additional arguments in functions
    if flow_args is not None:

        @wraps(flow_function)
        def flow_function(
            t: float, y: np.ndarray, fun: Callable = flow_function
        ):
            return fun(t, y, *flow_args)

    if jump_time_transition_function is not None:
        _add_attributes(jump_time_transition_function)

        @wraps(jump_time_transition_function)
        def jump_time_transition_function(
            t: float,
            y: np.ndarray,
            fun: Callable = jump_time_transition_function,
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
            def jump_set_function(
                t: float, y: np.ndarray, fun: Callable = jump_set_function
            ):
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
                t: float,
                y: np.ndarray,
                fun: Callable = jump_set_transition_function,
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
