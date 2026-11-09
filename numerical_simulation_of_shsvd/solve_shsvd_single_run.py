"""
Provides a solver for a single trajectory of a stochastic hybrid system
of variable dimension.

The solver combines continuous dynamics, prescribed jumps, jump-set
events, and stochastic transitions.
"""

from __future__ import annotations
from functools import wraps
import random
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .function_of_variable_dimension import FunctionOfVariableDimension
from .solve_hsvd import solve_hsvd
from ._utils import _add_attributes, _assign_args, _stack_args


Args = tuple[Any, ...] | None


def solve_shsvd_single_run(
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
    t_eval: ArrayLike | None = None,
    flow_args: Args = None,
    jump_rate_args: Args = None,
    transition_args: Args = None,
    jump_time_transition_args: Args = None,
    jump_set_args: Args = None,
    jump_set_transition_args: Args = None,
    check_jump_rate: bool = False,
    rng: np.random.Generator | None = None,
    solve_ivp_method: str = "RK45",
    **solve_ivp_options: Any,
) -> FunctionOfVariableDimension:
    """
    Computes a single trajectory of a stochastic hybrid system of variable
    dimension.

    The continuous dynamics and forced transitions are solved with
    solve_hsvd. Stochastic jump candidates are generated using a homogeneous
    Poisson process and accepted according to the current jump rate.

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
        Upper bound for the jump rate used to generate jump candidates. If
        None or 0, it is initialized from the jump rate at the initial state
        and check_jump_rate is enabled. The default is None.
    jump_times : list or np.ndarray or None, optional
        Prescribed jump times. A nonempty list or array must be provided when
        using this option. The default is None.
    jump_time_transition_function : callable or None, optional
        Transition function applied at prescribed jump times. If None and
        jump_times is provided, transition_function is used. The default is
        None.
    jump_set_function : callable or None, optional
        Event function defining the jump set. It is called as
        jump_set_function(t, y, *jump_set_args). Its zeros are treated as
        terminal events. The default is None.
    jump_set_transition_function : callable or None, optional
        Transition function applied when the jump set is reached. If None and
        jump_set_function is provided, transition_function is used. The
        default is None.
    t_eval : array-like or None, optional
        Times at which the solution is stored. Segment boundaries and jump
        times may also be included. If None, solve_ivp chooses the output
        times. The default is None.
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
        Additional arguments passed to jump_set_function. The default is
        None.
    jump_set_transition_args : tuple or None, optional
        Additional arguments for jump_set_transition_function. The default is
        None.
    check_jump_rate : bool, optional
        If True, check the jump rate at the stored values of each computed
        solution segment. If a larger rate is found, update the bound and
        recompute the segment. The default is False.
    rng : np.random.Generator or None, optional
        Random number generator used for jump candidates and acceptance
        decisions. If None, a new generator is created with
        np.random.default_rng(). The default is None.
    solve_ivp_method : str, optional
        Integration method passed to solve_ivp. The default is 'RK45'.
    **solve_ivp_options : dict
        Additional keyword arguments passed to solve_ivp.

    Returns
    -------
    func : FunctionOfVariableDimension
        Computed trajectory represented by time-value pairs.

    Notes
    -----
    Transition functions receive the current time and state, followed by
    their additional arguments. They normally return the new state.
    Return-flag attributes can indicate that updated additional arguments or
    an updated jump rate bound are also returned.

    Random numbers are drawn from rng. The state of a provided generator
    advances during the simulation.
    """
    # Initialise random number generator if none is provided
    if rng is None:
        rng = np.random.default_rng()

    # Set additional arguments in functions
    if jump_rate_args is not None:

        @wraps(jump_rate_function)
        def jump_rate_function(
            t: float, y: np.ndarray, fun: Callable = jump_rate_function
        ):
            return fun(t, y, *jump_rate_args)

    _add_attributes(transition_function)

    if transition_args is not None:

        @wraps(transition_function)
        def transition_function(
            t: float, y: np.ndarray, fun: Callable = transition_function
        ):
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
    # Uses 1.0 - rng.random() because then r is from (0,1].
    if jump_rate_bound is None or jump_rate_bound == 0:
        jump_rate_bound = max(jump_rate_function(t0, initial_value), 1e-10)
        check_jump_rate = True
    r = 1.0 - rng.random()
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
        s = rng.random()
        if s < jump_rate_function(last_time, last_value) / jump_rate_bound:
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
        # Uses 1.0 - rng.random() because then r is from (0,1].
        r = 1.0 - rng.random()
        t = min(t0 - np.log(r) / jump_rate_bound, end_time)

    func = func.concatenate(func_partial)

    return func
