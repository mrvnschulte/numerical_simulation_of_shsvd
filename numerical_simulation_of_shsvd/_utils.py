from collections import Counter

import numpy as np
from numpy.typing import ArrayLike


def _partial_t_eval(t_eval: ArrayLike, t0: float, tf: float) -> list:
    """
    Prepares the evaluation times by filtering them within a specified range.

    This function ensures that the evaluation times fall between the provided
    start (t0) and end (tf) times, including both boundaries if not already
    present.

    Parameters
    ----------
    t_eval : array-like
        The specified evaluation times.
    t0 : float
        The start of the time range.
    tf : float
        The end of the time range.

    Returns
    -------
    list
        The filtered evaluation times within the range [t0, tf].
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


def _is_sublist(list1: list[list], list2: list[list]) -> bool:
    """
    Checks if list1 is a sublist of list2.

    Parameters
    ----------
    list1 : list of lists
        List that should be checked for being a sublist of list2.
    list2 : list of lists
        List that should be checked for list1 being a sublist of it.

    Returns
    -------
    bool
        True, if list1 is a sublist of list2.
    """
    counter1 = Counter(map(tuple, list1))
    counter2 = Counter(map(tuple, list2))

    return all(counter1[item] <= counter2[item] for item in counter1)


@staticmethod
def _list_difference(list1: list[list], list2: list[list]) -> list:
    """
    Remove list1 from list2 if it is a sublist.

    Parameters
    ----------
    list1 : list of lists
        List that contains the elements to be removed.
    list2 : list of lists
        List where the elements from list1 should be removed from.

    Returns
    -------
    result : list
        List that contains all elements in list2, but not in list1,
        accounting for multiple entrances.

    Raises
    ------
    ValueError
        If list1 is not a sublist of list2.
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
