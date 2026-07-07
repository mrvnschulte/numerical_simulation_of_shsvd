from __future__ import annotations
import bisect

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as la


class FunctionOfVariableDimension:
    """
    A function whose output dimension may change over time.

    Represents a right-continuous function defined by a list of time-value
    pairs. The output dimension may jump at specified times, which are
    represented by two consecutive entries with the same time stamp (the
    left-limit value followed by the right-limit value).

    Attributes
    ----------
    times : list of float
        Time points at which the function is defined, in ascending order.
        A jump at time ``t`` is represented by two consecutive entries
        with the same value of ``t``.
    values : list of array-like
        Function values at the corresponding time points in ``self.times``.
    """

    def __init__(
        self,
        times: float | ArrayLike | None = None,
        values: float | ArrayLike | None = None,
        checks: bool = True,
    ) -> None:
        """
        Initializes the function with given times and values.

        Parameters
        ----------
        times : float or array-like, optional
            A sequence of time points. The default is None.
        values : float or array-like, optional
            A sequence of values corresponding to the times. The default is
            None.
        checks : bool, optional
            If True, sanity checks of the input are performed. This increases
            computation time significantly, hence if sanity of input can be
            assured, it should be set to False. The default is True.

        Raises
        ------
        ValueError
            If times and values are not the same length.
        TypeError
            If times and values are not numbers or not array-like.
        """
        if times is None and values is None:
            self.times = []
            self.values = []

        else:
            if checks or np.isscalar(times) or np.isscalar(values):
                times = self._convert_to_list(times)
                values = self._convert_to_list(values)

            # Check if initialisation is meaningful
            if not (
                isinstance(times, (list, np.ndarray))
                and isinstance(values, (list, np.ndarray))
            ):
                raise TypeError("Times and/or values are not array-like")
            if not all(
                isinstance(t, (int, float, np.floating, np.integer))
                for t in times
            ):
                raise TypeError(
                    f"All entries in times must be numeric, got: "
                    f"{[t for t in times if not isinstance(t, (int, float, np.floating, np.integer))]}"
                )
            if not (len(times) == len(values)):
                raise ValueError("Times and values are not of same length")
            for index, value in enumerate(values):
                if checks or np.isscalar(value):
                    values[index] = self._convert_to_list(value)
                if isinstance(values[index], (list, np.ndarray)):
                    if not all(
                        isinstance(v, (int, float, np.floating, np.integer))
                        for v in values[index]
                    ):
                        raise TypeError(
                            f"All entries in values must be numeric, got: {values[index]}"
                        )
                    continue
                else:
                    raise TypeError(
                        f"{value} in values is not array-like nor can it be converted"
                    )

            if checks:
                times, values = self._delete_double_values(
                    times, values, initialisation=True
                )

            self.times = times
            self.values = values

            # Sanity checks
            if checks:
                self._sort()
                self._check_multiple_jumps()
                self._check_dimension()

    def __call__(self, t: float | ArrayLike) -> np.ndarray | list[np.ndarray]:
        """
        Returns the interpolated value(s) of the function at the given time(s).

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the function.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            The interpolated value(s) at the given time(s).

        Raises
        ------
        ValueError
            If the time is outside the range of defined times.
        """
        if isinstance(t, (list, np.ndarray)):
            value = [self._interpolate_value(time) for time in t]
        else:
            value = self._interpolate_value(t)

        return value

    @staticmethod
    def _convert_to_list(value: float | ArrayLike) -> list:
        """
        Converts a scalar or array to a list; lists are returned unchanged.

        Parameters
        ----------
        value : float, or array-like
            The input to be converted.

        Returns
        -------
        value : list
            Converted list.
        """
        if np.isscalar(value):
            value = [value]
        elif isinstance(value, np.ndarray):
            value = list(value)

        return value

    def add_values(
        self,
        times: float | ArrayLike,
        values: float | ArrayLike,
        checks: bool = True,
    ) -> None:
        """
        Adds new time-value pairs to the function.

        Parameters
        ----------
        times : float or array-like
            The new time(s) to add. If a single float is provided, a single
            time-value pair is added. If array-like, multiple pairs are added.
        values : float or array-like
            The value(s) corresponding to the given time(s). If ``times`` is a
            single float, ``values`` must be a single array-like. If ``times``
            is array-like, ``values`` must be a list of array-likes of the same
            length.
        checks : bool, optional
            If ``True``, duplicate time-value pairs are removed and the
            function is sorted, checked for multiple jumps and consistent
            dimensions after insertion. Set to ``False`` if the input is
            guaranteed to be valid, as checks increase computation time
            significantly. The default is ``True``.

        Raises
        ------
        ValueError
            If ``times`` and ``values`` are not of the same length, if a
            dimension change occurs without a specified jump, or if more than
            one jump is defined at the same time.

        Notes
        -----
        When ``checks=False``, only an exact duplicate of the last stored
        time-value pair is silently skipped. No sorting or further validation
        is performed.

        Examples
        --------
        Add a single time-value pair:

        >>> func = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        >>> func.add_values(2.0, [5, 6])

        Add multiple time-value pairs at once:

        >>> func.add_values([3.0, 4.0], [[7, 8], [9, 10]])
        """
        # Convert everything to lists
        is_scalar = np.isscalar(times)
        times = self._convert_to_list(times)
        values = self._convert_to_list(values)

        if is_scalar:
            values = [values]
        else:
            values = [self._convert_to_list(v) for v in values]

        if checks:
            times, values = self._delete_double_values(times, values)
        elif (
            len(self.times) > 0
            and self.times[-1] == times[0]
            and len(self.values[-1]) == len(values[0])
            and np.allclose(self.values[-1], values[0])
        ):
            times = times[1:]
            values = values[1:]

        self.times.extend(times)
        self.values.extend(values)

        if checks:
            self._sort()
            self._check_multiple_jumps()
            self._check_dimension()

    def delete_values(
        self,
        times: float | list[float] | None = None,
        values: ArrayLike | list[ArrayLike] | None = None,
        indices: int | list[int] | None = None,
    ) -> None:
        """
        Deletes time-value pairs from the function.

        If nothing is provided to the function, it deletes the last time-value
        pair. Otherwise, the time and/or value and/or index of the time-value
        pair to be deleted, can be provided.

        Parameters
        ----------
        times : float or list of float, optional
            The time(s) to delete.
        values : array-like or list of array-like, optional
            The value(s) to delete.
        indices : int or list of int, optional
            The indices of the pairs to delete.

        Raises
        ------
        ValueError
            If a time to be deleted is a dimension jump time or a given
            time-value(-index) pair is not part of the function.
        TypeError
            If input types do not match, i.e. one time, but multiple values
            and/or indices are provided.
        """
        # If nothing is provided, delete the last time-value pair
        if times is None and values is None and indices is None:
            del self.times[-1]
            del self.values[-1]

        elif times in self.times:
            # Dimension jump times cannot be deleted
            if times in self.dimension_jump_times():
                raise ValueError("Cannot delete dimension jump time")

            # If only a time is provided, delete this (twice if jump time)
            if values is None and indices is None:
                index = self.times.index(times)
                if times in self.jump_times():
                    del self.times[index + 1]
                    del self.values[index + 1]
                del self.times[index]
                del self.values[index]

            # If time and value is provided, delete this if existing
            elif indices is None and isinstance(values, (list, np.ndarray)):
                index = self.times.index(times)
                if self.values[index] != values:
                    if (
                        times not in self.jump_times()
                        or self.values[index + 1] != values
                    ):
                        raise ValueError(
                            f"Time {times} not found with value {values}"
                        )
                    del self.times[index + 1]
                    del self.values[index + 1]
                else:
                    del self.times[index]
                    del self.values[index]

            # If time and index is provided, delete this if time is at index
            elif values is None and isinstance(
                indices, (int, np.int16, np.int32, np.int64)
            ):
                if self.times[indices] == times:
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(
                        f"Time at index {indices} does not have value {times}"
                    )

            # If time, value and index are provided, delete this if they match
            elif isinstance(values, (list, np.ndarray)) and isinstance(
                indices, (int, np.int16, np.int32, np.int64)
            ):
                if (
                    self.times[indices] == times
                    and self.values[indices] == values
                ):
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(
                        f"Time or value at index {indices} does not have value {times} or {values}, respectively"
                    )

            else:
                raise TypeError("Input types do not match")

        # If multiple times are provided, delete all of them together with
        # the corresponding values and/or indices
        elif isinstance(times, (list, np.ndarray)):
            if values is None and indices is None:
                for time in times:
                    self.delete_values(time)
            elif indices is None:
                for time, value in zip(times, values):
                    self.delete_values(time, value)
            elif values is None:
                indices = list(indices)
                for i, (time, index) in enumerate(zip(times, indices)):
                    self.delete_values(time, indices=index)
                    for j in range(i + 1, len(indices)):
                        if indices[j] > index:
                            indices[j] -= 1

            else:
                indices = list(indices)
                for i, (time, value, index) in enumerate(
                    zip(times, values, indices)
                ):
                    self.delete_values(time, value, index)
                    for j in range(i + 1, len(indices)):
                        if indices[j] > index:
                            indices[j] -= 1

        # If time is not stored, raise an error
        elif isinstance(times, (int, float, np.integer, np.floating)):
            raise ValueError(f"{times} is not in the times of the function")

        elif values in self.values:
            # If value and index are provided, delete this if they match
            if isinstance(
                indices,
                (
                    int,
                    np.int16,
                    np.int32,
                    np.int64,
                ),
            ):
                if self.values[indices] == values:
                    del self.times[indices]
                    del self.values[indices]
            # If only value is provided, search for this and delete it
            elif indices is None:
                index = self.values.index(values)
                del self.times[index]
                del self.values[index]

        # If multiple values are provided, delete each together with the
        # corresponding index if provided
        elif isinstance(values, (list, np.ndarray)) and isinstance(
            values[0], (list, np.ndarray)
        ):
            if indices is None:
                for value in values:
                    self.delete_values(values=value)
            else:
                indices = list(indices)
                for i, (value, index) in enumerate(zip(values, indices)):
                    self.delete_values(values=value, indices=index)
                    for j in range(i + 1, len(indices)):
                        if indices[j] > index:
                            indices[j] -= 1

        # If value is provided, that is not stored, raise an error
        elif values is not None:
            raise ValueError(
                f"Value {values} is not stored as a value of the function"
            )

        # If an index is provided, delete the corresponding time-value pair
        elif isinstance(indices, (int, np.int16, np.int32, np.int64)):
            if indices >= len(self.times):
                raise ValueError(
                    f"Index {indices} is greater than number of stored time-value pairs ({len(self.times)})"
                )
            del self.times[indices]
            del self.values[indices]

        # If multiple indices are provided, delete the corresponding time-value
        # pairs
        elif isinstance(indices, (list, np.ndarray)):
            indices = list(indices)
            for i, index in enumerate(indices):
                self.delete_values(indices=index)
                for j in range(i + 1, len(indices)):
                    if indices[j] > index:
                        indices[j] -= 1

        else:
            raise TypeError("Input format not correct")

    def maximal_time(self) -> float:
        """
        Returns the maximum time in the function.

        Returns
        -------
        float
            The maximum time, or -inf if no times are defined.
        """
        if len(self.times) > 0:
            return self.times[-1]
        else:
            return -np.inf

    def minimal_time(self) -> float:
        """
        Returns the minimum time in the function.

        Returns
        -------
        float
            The minimum time, or +inf if no times are defined.
        """
        if len(self.times) > 0:
            return self.times[0]
        else:
            return np.inf

    def dimension(self, t: float | ArrayLike | None = None) -> int | list[int]:
        """
        Returns the dimension of the function at the given time(s).

        Parameters
        ----------
        t : float, array-like, or None, optional
            The time point(s) at which to evaluate the dimension. If
            ``None``, the dimension at every entry of ``self.times`` is
            returned. The default is ``None``.

        Returns
        -------
        int or list of int
            A single integer if ``t`` is a scalar, or a list of integers
            if ``t`` is array-like or ``None``.

        Raises
        ------
        TypeError
            If ``t`` is neither a scalar nor array-like.
        """
        # If t is not provided, return dimension for all times
        if t is None:
            dimension_values = [len(value) for value in self.values]
            return dimension_values

        # If t is a number, return the dimension at that time
        elif isinstance(t, (int, float, np.integer, np.floating)):
            dimension_values = self.dimension()
            if t == self.times[-1]:
                return dimension_values[-1]
            index_t = bisect.bisect_right(self.times, t)
            dimension_t = dimension_values[index_t]
            return dimension_t

        # If t is array-like, return the dimensions at these times
        elif isinstance(t, (list, np.ndarray, tuple)):
            dimensions_t = [self.dimension(time) for time in t]
            return dimensions_t

        # If t is neither, raise an error
        else:
            raise TypeError(f"{t} is not array-like nor a number")

    def jump_times(
        self, return_index: bool = False
    ) -> list[float] | tuple[list[float], list[int]]:
        """
        Returns the times at which jumps occur in the function.

        Parameters
        ----------
        return_index : bool, optional
            If ``True``, also return the indices of the jump times in
            ``self.times``. The default is ``False``.

        Returns
        -------
        jump_times : list of float
            The times at which jumps occur.
        indices : list of int
            The indices of the jump times in ``self.times``. Only returned
            if ``return_index=True``.
        """
        if return_index:
            unique_times, indices, counts = np.unique(
                self.times, return_counts=True, return_index=return_index
            )
            jump_indices = list(indices[counts == 2])
        else:
            unique_times, counts = np.unique(
                self.times, return_counts=True, return_index=return_index
            )
        jump_times = list(unique_times[counts == 2])

        if return_index:
            return jump_times, jump_indices
        else:
            return jump_times

    def dimension_jump_times(
        self, return_index: bool = False
    ) -> list[float] | tuple[list[float], list[int]]:
        """
        Returns the times at which dimension jumps occur.

        Parameters
        ----------
        return_index : bool, optional
            If ``True``, also return the indices of the dimension jump times
            in ``self.times``. The default is ``False``.

        Returns
        -------
        dimension_jump_times : list of float
            The times at which dimension jumps occur.
        indices : list of int
            The indices of the dimension jump times in ``self.times``. Only
            returned if ``return_index=True``.
        """
        jump_times, indices = self.jump_times(return_index=True)
        dimension_jump_indices = [
            index
            for index in indices
            if (len(self.values[index]) != len(self.values[index + 1]))
        ]
        times = np.array(self.times)
        dimension_jump_times = list(times[dimension_jump_indices])

        if return_index:
            return dimension_jump_times, dimension_jump_indices
        else:
            return dimension_jump_times

    def non_dimension_jump_times(
        self, return_index: bool = False
    ) -> list[float] | tuple[list[float], list[int]]:
        """
        Returns the jump times at which the dimension does not change.

        Parameters
        ----------
        return_index : bool, optional
            If ``True``, also return the indices of the non-dimension jump
            times in ``self.times``. The default is ``False``.

        Returns
        -------
        non_dimension_jump_times : list of float
            The jump times at which there are no dimension jumps.
        non_dimension_jump_indices : list of int
            The indices of the non-dimension jump times in ``self.times``. Only
            returned if ``return_index=True``.
        """
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_jump_times, dimension_jump_indices = (
            self.dimension_jump_times(return_index=True)
        )
        non_dimension_jump_times = list(
            np.array(jump_times)[~np.isin(jump_times, dimension_jump_times)]
        )

        if return_index:
            non_dimension_jump_indices = list(
                np.array(jump_indices)[
                    ~np.isin(jump_indices, dimension_jump_indices)
                ]
            )
            return non_dimension_jump_times, non_dimension_jump_indices
        else:
            return non_dimension_jump_times

    def norm(
        self,
        t: float | ArrayLike | None = None,
        order: int | float | None = None,
        return_times: bool = False,
    ) -> list[float] | tuple[list[float], list[float]]:
        """
        Computes the norm of the function values at the given time(s).

        Parameters
        ----------
        t : float, array-like, or None, optional
            The time point(s) at which to evaluate the norm. If a scalar is
            provided, a single-element list is returned. If ``None``, the
            norm is evaluated at all entries of ``self.times``. The default
            is ``None``.
        order : int, float, or None, optional
            The order of the norm as accepted by ``scipy.linalg.norm``
            (e.g. ``1``, ``2``, ``np.inf``). If ``None``, the 2-norm is
            used. The default is ``None``.
        return_times : bool, optional
            If ``True``, ``self.times`` is returned alongside the norms.
            The default is ``False``.

        Returns
        -------
        value_norms : list of float
            The computed norms.
        times : list of float
            ``self.times``. Only returned if ``return_times=True``.
        """
        if t is None:
            values = self.values
        elif np.isscalar(t):
            values = [self(t)]
        else:
            values = self(t)

        value_norms = [float(la.norm(value, ord=order)) for value in values]

        if return_times:
            return value_norms, self.times

        return value_norms

    def left_limit(
        self, t: float | ArrayLike
    ) -> np.ndarray | list[np.ndarray]:
        """
        Returns the left-limit value(s) of the function at the given time(s).

        For times that are not jump times, the left limit equals the function
        value. For jump times, the left limit is the value immediately before
        the jump.

        Parameters
        ----------
        t : float or array-like
            The time(s) at which to evaluate the left limit(s).

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            The left-limit value(s) at the given time(s).
        """
        if isinstance(t, (list, np.ndarray)):
            value = [self.left_limit(time) for time in t]
        elif t in self.jump_times():
            value = self.values[self.times.index(t)]
        else:
            value = self(t)

        return value

    def concatenate(
        self, other: FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Concatenates this function with another function of variable dimension.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to concatenate with.

        Returns
        -------
        func : FunctionOfVariableDimension
            The concatenated function.

        Raises
        ------
        TypeError
            If the other object is not a function of variable dimension.
        ValueError
            If the dimensions of the functions do not match at overlap.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            raise TypeError(f"{other} is not a function of variable dimension")

        # Handle the special case of an empty function
        if len(self.times) == 0:
            func = other
        elif len(other.times) == 0:
            func = self

        # If the last time of this function equals the first time of the
        # other function, append the times and values
        elif np.isclose(self.times[-1], other.times[0]):
            if len(self.values[-1]) == len(other.values[0]):
                if np.allclose(self.values[-1], other.values[0]):
                    func = FunctionOfVariableDimension(
                        self.times + other.times[1:],
                        self.values + other.values[1:],
                        checks=False,
                    )
                else:
                    func = FunctionOfVariableDimension(
                        self.times + other.times,
                        self.values + other.values,
                        checks=False,
                    )
            else:
                func = FunctionOfVariableDimension(
                    self.times + other.times,
                    self.values + other.values,
                    checks=False,
                )

        # If the last time of this function is smaller than the first time of
        # the other function, append the times and values if the last and first
        # dimension match. Otherwise, raise an error.
        elif self.times[-1] < other.times[0]:
            if (
                self.dimension(self.times[-1])
                != other.dimension(other.times[0])
            ) and not other.times[0] in other.jump_times():
                raise ValueError(
                    "Dimensions of function do not coincide (only possible when last time for first function and first time for second function are the same)"
                )
            if (
                (
                    self.dimension(self.times[-1])
                    != other.dimension(other.times[0])
                )
                and other.times[0] in other.jump_times()
                and self.dimension(self.times[-1])
                != len(other.left_limit(other.times[0]))
            ):
                raise ValueError(
                    "Dimensions of function do not coincide (only possible when last time for first function and first time for second function are the same)"
                )

            func = FunctionOfVariableDimension(
                self.times + other.times, self.values + other.values
            )

        # If both functions overlap, check if they coincide at overlapping
        # times. If not, raise an error, otherwise merge times and values.
        else:
            jump_position = 0
            for index_self, t in enumerate(self.times):
                if t >= other.times[0]:
                    if t in other.times:
                        index_other = other.times.index(t) + jump_position
                        if t in other.jump_times():
                            jump_position = 1 - jump_position
                        if (
                            self.values[index_self]
                            != other.values[index_other]
                        ):
                            raise ValueError(
                                "Functions do not coincide at overlapping times"
                            )
                    else:
                        if self.dimension(t) != other.dimension(t):
                            raise ValueError(
                                "Functions do not coincide at overlapping times"
                            )

            func = FunctionOfVariableDimension(
                self.times + other.times, self.values + other.values
            )

        return func

    def plot_coordinates(
        self,
        title: str = "Plot of the coordinates",
        legend: bool = True,
        labels: list[str] | None = None,
        xlabel: str = "Time",
        ylabel: str = "Value",
        separate_plots: bool = False,
        sharex: bool = True,
        sharey: bool = True,
        grid: bool = False,
        show_jump_times: bool = False,
        show_dimension_jump_times: bool = False,
    ) -> plt.Figure | list[plt.Figure]:
        """
        Plots the coordinates of the function over time.

        Parameters
        ----------
        title : str, optional
            The title of the plot. The default is
            ``'Plot of the coordinates'``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``True``.
        labels : list of str or None, optional
            Custom legend labels for each coordinate. If ``None``, labels
            are generated automatically (e.g. ``'Coordinate 1'``). The
            default is ``None``.
        xlabel : str, optional
            Label for the x-axis. The default is ``'Time'``.
        ylabel : str, optional
            Label for the y-axis. The default is ``'Value'``.
        separate_plots : bool, optional
            If ``True``, each coordinate is plotted in a separate figure.
            The default is ``False``.
        sharex : bool, optional
            If ``True`` and ``separate_plots=True``, all coordinate figures
            share the same x-axis limits. The default is ``True``.
        sharey : bool, optional
            If ``True`` and ``separate_plots=True``, all coordinate figures
            share the same y-axis limits. The default is ``True``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.
        show_jump_times : bool, optional
            If ``True``, jump times are indicated as red dashed vertical
            lines. The default is ``False``.
        show_dimension_jump_times : bool, optional
            If ``True``, dimension jump times are indicated as green dashed
            vertical lines. The default is ``False``.

        Returns
        -------
        fig : plt.Figure or list of plt.Figure
            The created figure object.
        """
        # Create figures
        fig = plt.figure()
        dimension_jump_times, dimension_jump_indices = (
            self.dimension_jump_times(return_index=True)
        )
        dimensions = self.dimension()
        max_dimension = max(dimensions)
        prev_index = 0

        if not labels:
            labels = [f"Coordinate {i+1}" for i in range(max_dimension)]

        if separate_plots:
            figures = []
            for i in range(max_dimension):
                figures.append(plt.figure())

        # Plot the function iteratively between all dimension jump times
        for index in dimension_jump_indices:
            plt.figure(fig)
            plt.gca().set_prop_cycle(None)
            current_dimension = dimensions[prev_index]
            if current_dimension == max_dimension:
                lines = plt.plot(
                    self.times[prev_index : index + 1],
                    self.values[prev_index : index + 1],
                )
            else:
                plt.plot(
                    self.times[prev_index : index + 1],
                    self.values[prev_index : index + 1],
                )

            if separate_plots:
                for i in range(current_dimension):
                    plt.figure(figures[i])
                    plt.gca().set_prop_cycle(None)
                    values = [
                        value[i]
                        for value in self.values[prev_index : index + 1]
                    ]
                    plt.plot(self.times[prev_index : index + 1], values)

            prev_index = index + 1

        plt.figure(fig)
        plt.gca().set_prop_cycle(None)
        current_dimension = dimensions[prev_index]
        if current_dimension == max_dimension:
            lines = plt.plot(self.times[prev_index:], self.values[prev_index:])
        else:
            plt.plot(self.times[prev_index:], self.values[prev_index:])

        # Add specified design objects
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()

        # Plot vertical lines for (dimension) jump times, if desired
        if show_jump_times:
            jump_times = self.jump_times()
            ylim = plt.ylim()
            vlines = plt.vlines(jump_times, ylim[0], ylim[1], "red", "--")
            plt.ylim(ylim)
            if legend:
                plt.legend(lines + [vlines], labels + ["Jump times"])

        if show_dimension_jump_times:
            jump_times = self.dimension_jump_times()
            ylim = plt.ylim()
            dvlines = plt.vlines(jump_times, ylim[0], ylim[1], "green", "--")
            plt.ylim(ylim)
            if legend:
                if show_jump_times:
                    plt.legend(
                        lines + [vlines, dvlines],
                        labels
                        + ["Non-dimension jump times", "Dimension jump times"],
                    )
                else:
                    plt.legend(
                        lines + [dvlines], labels + ["Dimension jump times"]
                    )

        # Add design objects to separate coordinate plots, if desired
        if separate_plots and (
            sharex or sharey or title or grid or show_jump_times
        ):
            xlim = plt.xlim()
            ylim = plt.ylim()
            for i in range(max_dimension):
                plt.figure(figures[i])
                if title:
                    plt.title(f"{labels[i]} plot")
                if grid:
                    plt.grid()
                if sharex:
                    plt.xlim(xlim)
                if sharey:
                    plt.ylim(ylim)
                if show_jump_times:
                    jump_times = self.jump_times()
                    ylim = plt.ylim()
                    vlines = plt.vlines(
                        jump_times, ylim[0], ylim[1], "red", "--"
                    )
                    plt.ylim(ylim)
                    if legend:
                        plt.legend(lines + [vlines], labels + ["Jump times"])
                if show_dimension_jump_times:
                    jump_times = self.dimension_jump_times()
                    ylim = plt.ylim()
                    dvlines = plt.vlines(
                        jump_times, ylim[0], ylim[1], "green", "--"
                    )
                    plt.ylim(ylim)
                    if legend:
                        if show_jump_times:
                            plt.legend(
                                lines + [vlines, dvlines],
                                labels
                                + [
                                    "Non-dimension jump times",
                                    "Dimension jump times",
                                ],
                            )
                        else:
                            plt.legend(
                                lines + [dvlines],
                                labels + ["Dimension jump times"],
                            )

        # Plot dimension coordinates after last jump in separate plot
        if separate_plots:
            for i in range(current_dimension):
                plt.figure(figures[i])
                plt.gca().set_prop_cycle(None)
                values = [value[i] for value in self.values[prev_index:]]
                plt.plot(self.times[prev_index:], values)

        plt.show()

        if separate_plots:
            return figures
        return fig

    def plot_norm(
        self,
        order: int | float | list[int | float] = 2,
        title: str = "Plot of the norm",
        legend: bool = True,
        labels: list[str] | None = None,
        xlabel: str = "Time",
        ylabel: str = "Value",
        grid: bool = False,
        show_jump_times: bool = False,
        show_dimension_jump_times: bool = False,
    ) -> plt.Figure:
        """
        Plots the norm of the function over time.

        Parameters
        ----------
        order : int, float, or list of int or float, optional
            The norm order(s) to plot as accepted by ``scipy.linalg.norm``
            (e.g. ``1``, ``2``, ``np.inf``). A list may be provided to
            overlay multiple norms in the same figure. The default is ``2``
            (Euclidean norm).
        title : str, optional
            The title of the plot. The default is ``'Plot of the norm'``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``True``.
        labels : list of str or None, optional
            Custom legend labels for each norm. If ``None``, labels are
            generated automatically (e.g. ``'2 norm'``). The default is
            ``None``.
        xlabel : str, optional
            Label for the x-axis. The default is ``'Time'``.
        ylabel : str, optional
            Label for the y-axis. The default is ``'Value'``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.
        show_jump_times : bool, optional
            If ``True``, jump times are indicated as red dashed vertical
            lines. The default is ``False``.
        show_dimension_jump_times : bool, optional
            If ``True``, dimension jump times are indicated as green dashed
            vertical lines. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.
        """
        # Create figure
        fig = plt.figure()

        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order
        if not labels:
            labels = [f"{order} norm" for order in order_list]

        # Compute and plot norms
        lines = []
        for order in order_list:
            value_norms = self.norm(order=order)
            (line,) = plt.plot(self.times, value_norms)
            lines.append(line)

        # Add specified design objects
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()

        # Add vertical lines for (dimension) jump times, if desired
        if show_jump_times:
            jump_times = self.jump_times()
            ylim = plt.ylim()
            vlines = plt.vlines(jump_times, ylim[0], ylim[1], "red", "--")
            plt.ylim(ylim)
            if legend:
                plt.legend(lines + [vlines], labels + ["Jump times"])
        if show_dimension_jump_times:
            jump_times = self.dimension_jump_times()
            ylim = plt.ylim()
            dvlines = plt.vlines(jump_times, ylim[0], ylim[1], "green", "--")
            plt.ylim(ylim)
            if legend:
                if show_jump_times:
                    plt.legend(
                        lines + [vlines, dvlines],
                        labels
                        + ["Non-dimension jump times", "Dimension jump times"],
                    )
                else:
                    plt.legend(
                        lines + [dvlines], labels + ["Dimension jump times"]
                    )

        return fig

    def __neg__(self) -> FunctionOfVariableDimension:
        """
        Negates the values of the function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with negated values.
        """
        negative_values = []
        for value in self.values:
            negative_values.append(list(-np.array(value)))
        func = FunctionOfVariableDimension(self.times, negative_values)
        return func

    def __add__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Adds the values of this function to another value or function.

        Parameters
        ----------
        other : float, array-like, or FunctionOfVariableDimension
            The value or function to add to the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the added values.

        Raises
        ------
        ValueError
            If dimensions do not match when adding another function or an
            array.
        """
        # If number is added, add it to every value
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) + other)
            return func
        # If array is added, add every component to the respective coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Can only add list/array of numbers")
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) + other[:dimension])
            return func
        # If another function is added, add function values at every time, if
        # possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only add functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only add functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) + np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    + np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __radd__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Supports addition when this function is on the right-hand side.

        Delegates to :meth:`__add__`. Called when ``other + self`` is
        evaluated and ``other`` does not support the operation.

        Parameters
        ----------
        other : float or array-like
            The left-hand operand.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance with the added values.
        """
        return self + other

    def __sub__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Subtracts a value or function from the values of this function.

        Parameters
        ----------
        other : float, array-like, or FunctionOfVariableDimension
            The value or function to subtract from the values of this
            function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the subtracted values.

        Raises
        ------
        ValueError
            If dimensions do not match when subtracting another function
            or an array.
            If the functions do not share the same time range.
        """
        # If number is subtracted, subtract it from every value
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) - other)
            return func
        # If array is subtracted, subtract every component from the respective
        # coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Can only subtract list/array of numbers")
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) - other[:dimension])
            return func
        # If another function is subtracted, subtract function values at every
        # time, if possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only subtract functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only subtract functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) - np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    - np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __rsub__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Subtracts the values of this function from another value.

        Parameters
        ----------
        other : float, or array-like
            The value from which to subtract the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the subtracted values.
        """
        return -self + other

    def __mul__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Multiplies the values of this function by another value or function.

        Parameters
        ----------
        other : float, array-like, or FunctionOfVariableDimension
            The value or function to multiply with the values of this
            function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the multiplied values.

        Raises
        ------
        ValueError
            If dimensions do not match when multiplying by another function
            or an array.
        """
        # If number is multiplied, multiply it with every value
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) * other)
            return func
        # If array is multiplied, multiply every component with the respective
        # coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Can only multiply list/array of numbers")
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) * other[:dimension])
            return func
        # If another function is multiplied, multiply function values at every
        # time, if possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only multiply functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only multiply functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) * np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    * np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __rmul__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Multiplies another value with the values of this function.

        Parameters
        ----------
        other : float or array-like
            The value to multiply with the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the multiplied values.
        """
        return self * other

    def __truediv__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Divides the values of this function by another value or function.

        Parameters
        ----------
        other : float, array-like, or FunctionOfVariableDimension
            The value or function to divide the values of this
            function by.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the divided values.

        Raises
        ------
        ValueError
            If dimensions do not match when dividing by another function or
            an array.
        """
        # If division by a number, divide every value by it
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) / other)
            return func
        # If division by an array, divide every coordinate by the respective
        # component
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Can only divide by list/array of numbers")
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) / other[:dimension])
            return func
        # If function is divided by another, divide function values at every
        # time, if possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only divide functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only divide functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) / np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    / np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __rtruediv__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Divides another value by the values of this function.

        Parameters
        ----------
        other : float or array-like
            The value to divide by the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the divided values.
        """
        # If number is divided, divide it by every value
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(other / np.array(value))
            return func
        # If array is divided, divide every component by the respective
        # coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Can only divide by list/array of numbers")
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(other[:dimension] / np.array(value))
            return func
        else:
            return NotImplemented

    def __mod__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Computes the element-wise modulus of this function's values.

        Parameters
        ----------
        other: float, array-like, or FunctionOfVariableDimension
            The value or function to compute the modulus with.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the modulus values.

        Raises
        ------
        ValueError
            If dimensions do not match when taking modulus with another
            function or an array.
        """
        # If modulus to number is computed, compute modulus of every value to
        # it
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) % other)
            return func
        # If modulus to array is computed, compute modulus of every coordinate
        # with respect to the respective component
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError(
                        "Can only take modulus to list/array of numbers"
                    )
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) % other[:dimension])
            return func
        # If modulus to another function is computed, compute modulus of
        # function values at every time, if possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only take modulus of functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only take modulus of functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) % np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    % np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __rmod__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Computes the modulus of another value with the function values.

        Parameters
        ----------
        other : float or array-like
            The value to compute modulus with the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the modulus values.
        """
        # If modulus to number is computed, compute modulus of every value to
        # it
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(other % np.array(value))
            return func
        # If modulus to array is computed, compute modulus of every coordinate
        # with respect to the respective component
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError(
                        "Can only take modulus to list/array of numbers"
                    )
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(other[:dimension] % np.array(value))
            return func
        else:
            return NotImplemented

    def __pow__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Raises the function values to the given power element-wise.

        Parameters
        ----------
        other : float, array-like, or FunctionOfVariableDimension
            The value or function to raise the values of this function to.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the exponentiated values.

        Raises
        ------
        ValueError
            If dimensions do not match when exponentiating by another
            function or an array.
        """
        # If power to number is computed, compute power of every value to the
        # number
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) ** other)
            return func
        # If function to the power of array is computed, compute power of every
        # coordinate to the respective component
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError(
                        "Can only take power of list/array of numbers"
                    )
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(np.array(value) ** other[:dimension])
            return func
        # If power to another function is computed, compute powers of function
        # values at every time, if possible
        elif isinstance(other, FunctionOfVariableDimension):
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Can only exponentiate functions with same minimal "
                    "and maximal time"
                )
            if not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only exponentiate functions with the same "
                    "dimension at each time"
                )
            times = list(np.unique(self.times + other.times))
            jump_times = list(
                np.unique(self.jump_times() + other.jump_times())
            )
            values = [
                list(np.array(self(time)) ** np.array(other(time)))
                for time in times
            ]
            left_limits = [
                list(
                    np.array(self.left_limit(jump_time))
                    ** np.array(other.left_limit(jump_time))
                )
                for jump_time in jump_times
            ]
            func = FunctionOfVariableDimension(
                jump_times + times, left_limits + values
            )
            return func
        else:
            return NotImplemented

    def __rpow__(
        self, other: float | ArrayLike
    ) -> FunctionOfVariableDimension:
        """
        Raises another value to the power of the values of this function.

        Parameters
        ----------
        other : float or array-like
            The value to raise to the powers of the values of this
            function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the exponentiated values.
        """
        # If power to number is computed, compute power of number to every
        # value
        if np.isscalar(other) and not isinstance(other, (bool, str)):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(other ** np.array(value))
            return func
        # If array to the power of function is computed, compute power of every
        # component to the respective coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError(
                        "Can only take power of list/array of numbers"
                    )
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                dimension = len(value)
                if array_length < dimension:
                    raise ValueError(
                        "Array must be at least of length of "
                        f"maximal dimension. Dimension of {value}"
                        f"at time {func.times[i]} is {dimension}."
                    )
                func.values[i] = list(other[:dimension] ** np.array(value))
            return func
        else:
            return NotImplemented

    def __eq__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for equality.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the functions are equal, False otherwise.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            return False
        else:
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                return False
            elif not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                return False
            else:
                bool1 = all(
                    [
                        np.all(np.array(value1) == np.array(value2))
                        for value1, value2 in zip(
                            self(self.times), other(self.times)
                        )
                    ]
                )
                bool2 = all(
                    [
                        np.all(np.array(value1) == np.array(value2))
                        for value1, value2 in zip(
                            self(other.times), other(other.times)
                        )
                    ]
                )
                bool3 = all(
                    [
                        np.all(np.array(value1) == np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(self.jump_times()),
                            other.left_limit(self.jump_times()),
                        )
                    ]
                )
                bool4 = all(
                    [
                        np.all(np.array(value1) == np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(other.jump_times()),
                            other.left_limit(other.jump_times()),
                        )
                    ]
                )
                return bool1 and bool2 and bool3 and bool4

    def __ne__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for inequality.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the functions are not equal, False otherwise.
        """
        return not self == other

    def __gt__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for being greater.

        Compares this function with another one for being greater at all times
        and in every coordinate.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the coordinate values of this function are greater than
            the other ones, False otherwise.

        Raises
        ------
        ValueError
            If the functions have different time ranges or dimensions.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            return NotImplemented
        else:
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Cannot compare functions with minimal times "
                    f"{self.minimal_time()} and "
                    f"{other.minimal_time()} and maximal times"
                    f"{self.maximal_time()} and "
                    f"{other.maximal_time()}"
                )
            elif not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only compare functions of the same "
                    "dimension at each time"
                )
            else:
                bool1 = all(
                    [
                        np.all(np.array(value1) > np.array(value2))
                        for value1, value2 in zip(
                            self(self.times), other(self.times)
                        )
                    ]
                )
                bool2 = all(
                    [
                        np.all(np.array(value1) > np.array(value2))
                        for value1, value2 in zip(
                            self(other.times), other(other.times)
                        )
                    ]
                )
                bool3 = all(
                    [
                        np.all(np.array(value1) > np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(self.jump_times()),
                            other.left_limit(self.jump_times()),
                        )
                    ]
                )
                bool4 = all(
                    [
                        np.all(np.array(value1) > np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(other.jump_times()),
                            other.left_limit(other.jump_times()),
                        )
                    ]
                )
                return bool1 and bool2 and bool3 and bool4

    def __lt__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for being less.

        Compares this function with another one for being less at all times and
        in every coordinate.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the coordinate values of this function are less than
            the other ones, False otherwise.

        Raises
        ------
        ValueError
            If the functions have different time ranges or dimensions.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            return NotImplemented
        else:
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Cannot compare functions with minimal times "
                    f"{self.minimal_time()} and "
                    f"{other.minimal_time()} and maximal times"
                    f"{self.maximal_time()} and "
                    f"{other.maximal_time()}"
                )
            elif not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only compare functions of the same "
                    "dimension at each time"
                )
            else:
                bool1 = all(
                    [
                        np.all(np.array(value1) < np.array(value2))
                        for value1, value2 in zip(
                            self(self.times), other(self.times)
                        )
                    ]
                )
                bool2 = all(
                    [
                        np.all(np.array(value1) < np.array(value2))
                        for value1, value2 in zip(
                            self(other.times), other(other.times)
                        )
                    ]
                )
                bool3 = all(
                    [
                        np.all(np.array(value1) < np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(self.jump_times()),
                            other.left_limit(self.jump_times()),
                        )
                    ]
                )
                bool4 = all(
                    [
                        np.all(np.array(value1) < np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(other.jump_times()),
                            other.left_limit(other.jump_times()),
                        )
                    ]
                )
                return bool1 and bool2 and bool3 and bool4

    def __ge__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for being greater or equal.

        Compares this function with another one for being greater or equal at
        all times and in every coordinate.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the coordinate values of this function are greater than
            or equal to the other ones, False otherwise.

        Raises
        ------
        ValueError
            If the functions have different time ranges or dimensions.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            return NotImplemented
        else:
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Cannot compare functions with minimal times "
                    f"{self.minimal_time()} and "
                    f"{other.minimal_time()} and maximal times"
                    f"{self.maximal_time()} and "
                    f"{other.maximal_time()}"
                )
            elif not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only compare functions of the same "
                    "dimension at each time"
                )
            else:
                bool1 = all(
                    [
                        np.all(np.array(value1) >= np.array(value2))
                        for value1, value2 in zip(
                            self(self.times), other(self.times)
                        )
                    ]
                )
                bool2 = all(
                    [
                        np.all(np.array(value1) >= np.array(value2))
                        for value1, value2 in zip(
                            self(other.times), other(other.times)
                        )
                    ]
                )
                bool3 = all(
                    [
                        np.all(np.array(value1) >= np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(self.jump_times()),
                            other.left_limit(self.jump_times()),
                        )
                    ]
                )
                bool4 = all(
                    [
                        np.all(np.array(value1) >= np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(other.jump_times()),
                            other.left_limit(other.jump_times()),
                        )
                    ]
                )
                return bool1 and bool2 and bool3 and bool4

    def __le__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Compares this function with another one for being less or equal.

        Compares this function with another one for being less or equal at all
        times and in every coordinate.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            The function to compare with.

        Returns
        -------
        bool
            True if the coordinate values of this function are less than or
            equal to the other ones, False otherwise.

        Raises
        ------
        ValueError
            If the functions have different time ranges or dimensions.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            return NotImplemented
        else:
            if not (
                self.minimal_time() == other.minimal_time()
                and self.maximal_time() == other.maximal_time()
            ):
                raise ValueError(
                    "Cannot compare functions with minimal times "
                    f"{self.minimal_time()} and "
                    f"{other.minimal_time()} and maximal times"
                    f"{self.maximal_time()} and "
                    f"{other.maximal_time()}"
                )
            elif not (
                self.dimension(self.times) == other.dimension(self.times)
                and self.dimension(other.times) == other.dimension(other.times)
            ):
                raise ValueError(
                    "Can only compare functions of the same "
                    "dimension at each time"
                )
            else:
                bool1 = all(
                    [
                        np.all(np.array(value1) <= np.array(value2))
                        for value1, value2 in zip(
                            self(self.times), other(self.times)
                        )
                    ]
                )
                bool2 = all(
                    [
                        np.all(np.array(value1) <= np.array(value2))
                        for value1, value2 in zip(
                            self(other.times), other(other.times)
                        )
                    ]
                )
                bool3 = all(
                    [
                        np.all(np.array(value1) <= np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(self.jump_times()),
                            other.left_limit(self.jump_times()),
                        )
                    ]
                )
                bool4 = all(
                    [
                        np.all(np.array(value1) <= np.array(value2))
                        for value1, value2 in zip(
                            self.left_limit(other.jump_times()),
                            other.left_limit(other.jump_times()),
                        )
                    ]
                )
                return bool1 and bool2 and bool3 and bool4

    def _sort(self) -> None:
        """
        Sorts times and values in ascending order.

        Duplicate time points preserve their original relative order,
        which is required to correctly represent jump discontinuities.

        Notes
        -----
        Modifies ``self.times`` and ``self.values`` in-place.
        """
        sorted_pairs = sorted(
            zip(self.times, self.values, range(len(self.times))),
            key=lambda x: (x[0], x[2]),
        )
        times, values = map(list, zip(*[(x[0], x[1]) for x in sorted_pairs]))

        self.times = times
        self.values = values

    def _delete_double_values(
        self,
        times: list[float],
        values: list[ArrayLike],
        initialisation: bool = False,
    ) -> tuple[list[float], list[ArrayLike]]:
        """
        Removes duplicate time-value pairs.

        During initialisation, duplicates within ``times`` and ``values``
        themselves are removed. After initialisation, pairs that already
        exist in ``self.times`` and ``self.values`` are removed.

        Parameters
        ----------
        times : list of float
            Candidate time points.
        values : list of array-like
            Candidate values corresponding to ``times``.
        initialisation : bool, optional
            If ``True``, duplicates are searched within ``times``/``values``
            only (no comparison against the stored data). The default is
            ``False``.

        Returns
        -------
        times : list of float
            Deduplicated time points.
        values : list of array-like
            Deduplicated values.
        """
        times = list(times)
        values = list(values)

        ref_times = times if initialisation else self.times
        ref_values = values if initialisation else self.values

        indices_to_remove = set()

        for index_1, (time, value) in enumerate(zip(times, values)):
            indices = [
                idx
                for idx, ref_time in enumerate(ref_times)
                if ref_time == time
                and (idx > index_1 if initialisation else True)
            ]

            for idx in indices:
                ref_val = ref_values[idx]
                if len(ref_val) == len(value) and np.allclose(ref_val, value):
                    indices_to_remove.add(index_1)

        for idx in sorted(indices_to_remove, reverse=True):
            del times[idx]
            del values[idx]

        return times, values

    def _check_multiple_jumps(self) -> None:
        """
        Raises an error if multiple jumps occur at the same time.

        Raises
        ------
        ValueError
            If more than one jump occurs at the same time.
        """
        unique_times, indices, counts = np.unique(
            self.times, return_counts=True, return_index=True
        )
        jump_indices = indices[counts > 2]

        if len(jump_indices) > 0:
            raise ValueError(
                f"More than one jump occurs at time(s): {np.array(self.times)[jump_indices]}"
            )

    def _check_dimension(self) -> None:
        """
        Raises an error if the function dimension changes without a jump.

        A dimension change is only valid if it coincides with an explicitly
        defined jump time (i.e. two consecutive entries share the same time
        stamp). A dimension change at any other time raises an error.

        Raises
        ------
        ValueError
            If the dimension changes at a time that is not a jump time.
        """
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_values = self.dimension()

        if len(jump_indices) == 0:
            segments = [(0, len(dimension_values))]
        else:
            segments = (
                [(0, jump_indices[0] + 1)]
                + [
                    (jump_indices[i] + 1, jump_indices[i + 1] + 1)
                    for i in range(len(jump_indices) - 1)
                ]
                + [(jump_indices[-1] + 1, len(dimension_values))]
            )

        for start, end in segments:
            dimension = dimension_values[start]
            for j in range(start + 1, end):
                if dimension_values[j] != dimension:
                    raise ValueError(
                        f"Function changes dimension at {self.times[j]} without specified jump"
                    )

    def _interpolate_value(self, t: float) -> np.ndarray:
        """
        Linearly interpolates the value of the function at a given time.

        Parameters
        ----------
        t : float
            The time at which to interpolate the value.

        Returns
        -------
        value : np.ndarray
            The interpolated value.

        Raises
        ------
        ValueError
            If the time is outside the range of defined times.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(
                f"t is not between {self.times[0]} and {self.times[-1]} (first and last value of times)"
            )
        elif t == self.times[-1]:
            return np.array(self.values[-1])

        index_t = bisect.bisect_right(self.times, t)

        interp_times = np.array(self.times[index_t - 1 : index_t + 1])
        interp_values = np.array(self.values[index_t - 1 : index_t + 1])

        value = interp_values[0] + (
            ((t - interp_times[0]) / (interp_times[1] - interp_times[0]))
            * (interp_values[1] - interp_values[0])
        )

        return value
