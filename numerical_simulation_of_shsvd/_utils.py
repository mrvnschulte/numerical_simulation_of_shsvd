"""
Provides internal utility functions for the hybrid system solvers.

The functions manage transition return flags and additional arguments,
prepare evaluation times, and compare or modify lists of jump times.
"""

from __future__ import annotations
from collections import Counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


Args = tuple[Any, ...] | None


def _add_attributes(func: Callable) -> None:
    """
    Adds missing return flags to a transition function.

    Existing attributes are left unchanged. Missing attributes are initialized
    to False. These flags indicate which additional arguments are returned by
    the transition function.

    Parameters
    ----------
    func : callable
        Transition function to which the attributes are added.
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


def _assign_args(
    values: Any,
    flow_args: Args,
    jump_rate_args: Args,
    transition_args: Args,
    jump_time_transition_args: Args,
    jump_set_args: Args,
    jump_set_transition_args: Args,
    jump_rate_bound: float | None,
    returns_flow_args: bool,
    returns_jump_rate_args: bool,
    returns_transition_args: bool,
    returns_jump_time_transition_args: bool,
    returns_jump_set_args: bool,
    returns_jump_set_transition_args: bool,
    returns_jump_rate_bound: bool,
) -> tuple[Any, Args, Args, Args, Args, Args, Args, float | None]:
    """
    Assigns the values returned by a transition function.

    If at least one return flag is True, the first entry of values is treated
    as the state. The following entries replace the arguments indicated by
    the flags, in the order listed in the return section. If all flags are
    False, values itself is treated as the state.

    Parameters
    ----------
    values : Any
        State or sequence containing the state followed by updated arguments.
    flow_args : tuple or None
        Current additional arguments for the flow function.
    jump_rate_args : tuple or None
        Current additional arguments for the jump rate function.
    transition_args : tuple or None
        Current additional arguments for the stochastic transition function.
    jump_time_transition_args : tuple or None
        Current additional arguments for the time transition function.
    jump_set_args : tuple or None
        Current additional arguments for the jump set function.
    jump_set_transition_args : tuple or None
        Current additional arguments for the jump set transition function.
    jump_rate_bound : float or None
        Current upper bound for the jump rate.
    returns_flow_args : bool
        Whether values contains updated flow arguments.
    returns_jump_rate_args : bool
        Whether values contains updated jump rate arguments.
    returns_transition_args : bool
        Whether values contains updated stochastic transition arguments.
    returns_jump_time_transition_args : bool
        Whether values contains updated time transition arguments.
    returns_jump_set_args : bool
        Whether values contains updated jump set arguments.
    returns_jump_set_transition_args : bool
        Whether values contains updated jump set transition arguments.
    returns_jump_rate_bound : bool
        Whether values contains an updated jump rate bound.

    Returns
    -------
    state : Any
        State returned by the transition function.
    flow_args : tuple or None
        Updated flow arguments.
    jump_rate_args : tuple or None
        Updated jump rate arguments.
    transition_args : tuple or None
        Updated stochastic transition arguments.
    jump_time_transition_args : tuple or None
        Updated time transition arguments.
    jump_set_args : tuple or None
        Updated jump set arguments.
    jump_set_transition_args : tuple or None
        Updated jump set transition arguments.
    jump_rate_bound : float or None
        Updated jump rate bound.
    """
    returns = [
        returns_flow_args,
        returns_jump_rate_args,
        returns_transition_args,
        returns_jump_time_transition_args,
        returns_jump_set_args,
        returns_jump_set_transition_args,
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


def _convert_to_list(value: float | ArrayLike) -> list:
    """
    Converts a scalar or NumPy array to a list.

    Lists are returned unchanged. A scalar becomes a one-element list, and
    a NumPy array is converted with list().

    Parameters
    ----------
    value : scalar, np.ndarray, or list
        Value to convert.

    Returns
    -------
    list
        Converted value.

    Raises
    ------
    TypeError
        If value is not a scalar, NumPy array, or list.
    """
    if np.isscalar(value):
        value = [value]
    elif isinstance(value, np.ndarray):
        value = list(value)
    elif not isinstance(value, list):
        raise TypeError(
            f"{value} is expected to be of type scalar, array or list"
        )

    return value


def _is_sublist(list1: list[list], list2: list[list]) -> bool:
    """
    Checks whether list1 is contained in list2, accounting for multiplicity.

    The order of the inner lists is ignored. Each inner list is converted to
    a tuple for comparison.

    Parameters
    ----------
    list1 : list of lists
        Inner lists to check for.
    list2 : list of lists
        Inner lists in which to search.

    Returns
    -------
    bool
        True if every inner list in list1 occurs at least as often in list2.
    """
    counter1 = Counter(map(tuple, list1))
    counter2 = Counter(map(tuple, list2))

    return all(counter1[item] <= counter2[item] for item in counter1)


def _list_difference(list1: list[list], list2: list[list]) -> list:
    """
    Removes the inner lists in list1 from list2, accounting for multiplicity.

    Parameters
    ----------
    list1 : list of lists
        Inner lists to remove. Should be a sublist of list2.
    list2 : list of lists
        Inner lists from which to remove the entries.

    Returns
    -------
    result : list of lists
        Remaining inner lists. Equal entries are grouped together.

    Raises
    ------
    ValueError
        If list1 is not contained in list2, accounting for multiplicity.
    """
    if not _is_sublist(list1, list2):
        raise ValueError(f"{list1} is not a sublist of {list2}")

    counter1 = Counter(map(tuple, list1))
    counter2 = Counter(map(tuple, list2))

    result = []
    for item in counter2:
        count_difference = counter2[item] - counter1[item]
        result.extend([list(item)] * count_difference)

    return result


def _partial_t_eval(
    t_eval: ArrayLike | None, t0: float, tf: float
) -> np.ndarray | None:
    """
    Restricts evaluation times to an interval and adds its endpoints.

    The order of retained evaluation times is preserved. If t0 or tf is
    missing, it is inserted at the beginning or appended at the end,
    respectively.

    Parameters
    ----------
    t_eval : array-like or None
        Evaluation times. If None, no evaluation grid is constructed.
    t0 : float
        Start of the interval.
    tf : float
        End of the interval.

    Returns
    -------
    t_eval_partial : np.ndarray or None
        Evaluation times in [t0, tf], including both endpoints, or None if
        t_eval is None.
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


def _stack_args(
    jump_type: str,
    flow_args: Args,
    jump_rate_args: Args,
    transition_args: Args,
    jump_time_transition_args: Args,
    jump_set_args: Args,
    jump_set_transition_args: Args,
    jump_rate_bound: float | None,
    returns_flow_args: bool,
    returns_jump_rate_args: bool,
    returns_transition_args: bool,
    returns_jump_time_transition_args: bool,
    returns_jump_set_args: bool,
    returns_jump_set_transition_args: bool,
    returns_jump_rate_bound: bool,
) -> tuple[Any, ...]:
    """
    Constructs the additional arguments for a transition function.

    First adds the arguments associated with jump_type. It then appends
    selected argument tuples according to the return flags. The jump rate
    bound is appended as a single argument when requested.

    Parameters
    ----------
    jump_type : str
        Transition type: 'time', 'set', or 'stochastic'.
    flow_args : tuple or None
        Additional arguments for the flow function.
    jump_rate_args : tuple or None
        Additional arguments for the jump rate function.
    transition_args : tuple or None
        Additional arguments for the stochastic transition function.
    jump_time_transition_args : tuple or None
        Additional arguments for the time transition function.
    jump_set_args : tuple or None
        Additional arguments for the jump set function.
    jump_set_transition_args : tuple or None
        Additional arguments for the jump set transition function.
    jump_rate_bound : float or None
        Current upper bound for the jump rate.
    returns_flow_args : bool
        Whether to append flow_args.
    returns_jump_rate_args : bool
        Whether to append jump_rate_args.
    returns_transition_args : bool
        Whether to append transition_args for other jump types.
    returns_jump_time_transition_args : bool
        Whether to append jump_time_transition_args for other jump types.
    returns_jump_set_args : bool
        Whether to append jump_set_args.
    returns_jump_set_transition_args : bool
        Whether to append jump_set_transition_args for other jump types.
    returns_jump_rate_bound : bool
        Whether to append jump_rate_bound.

    Returns
    -------
    args : tuple
        Additional arguments to pass to the transition function.

    Notes
    -----
    Argument tuples selected by a return flag must not be None.
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
    if returns_transition_args and jump_type != "stochastic":
        args += transition_args
    if returns_jump_time_transition_args and jump_type != "time":
        args += jump_time_transition_args
    if returns_jump_set_args:
        args += jump_set_args
    if returns_jump_set_transition_args and jump_type != "set":
        args += jump_set_transition_args
    if returns_jump_rate_bound:
        args += (jump_rate_bound,)

    return args
