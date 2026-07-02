import bisect
from collections import Counter
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.linalg as la

from .stats_function_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)


class FunctionOfVariableDimension:
    """
    A class for functions of variable dimension.

    A class to represent a function of variable dimension defined by
    time-value pairs.

    Attributes
    ----------
    times : list
        A list of time points of the function.
    values : list
        A list of corresponding values at the given time points.
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
            If times and values are not the same length or not array-like.
        """
        if times is None and values is None:
            self.times = []
            self.values = []

        else:
            if checks:
                times = self._convert_to_list(times)
                values = self._convert_to_list(values)

            # Check if initialisation is meaningful
            if not (
                isinstance(times, (list, np.ndarray))
                and isinstance(values, (list, np.ndarray))
            ):
                raise ValueError("Times and/or values are not array-like")
            if not (len(times) == len(values)):
                raise ValueError("Times and values are not of same length")
            for index, value in enumerate(values):
                if checks:
                    values[index] = self._convert_to_list(value)
                if isinstance(values[index], (list, np.ndarray)):
                    continue
                else:
                    raise ValueError(
                        f"{value} in values is not array-like \
                                     nor can it be converted"
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

    def __call__(self, t: float | ArrayLike) -> float | list:
        """
        Returns the interpolated value(s) of the function at the given time(s).

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the function.

        Returns
        -------
        value : float or list
            The interpolated value(s) at the given time(s).
        """
        if isinstance(t, (list, np.ndarray)):
            value = [self._interpolate_value(time) for time in t]
        else:
            value = self._interpolate_value(t)

        return value

    @staticmethod
    def _convert_to_list(number: float | np.ndarray) -> list:
        """
        Converts a number or np.ndarray to a list.

        Parameters
        ----------
        number : float, or np.ndarray
            The input to be converted.

        Returns
        -------
        number : list
            Converted list.
        """
        if np.isscalar(number):
            number = [number]
        elif isinstance(number, np.ndarray):
            number = list(number)

        return number

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
        self, times: list, values: list, initialisation: bool = False
    ) -> tuple[list, list]:
        """
        Removes duplicate time-value pairs from the lists.

        Removes duplicate time-value pairs from times, values during
        initialisation and duplicates between times, values and self.times and
        self.values after initialisation.

        Parameters
        ----------
        times : list
            The list of times.
        values : list
            The list of values.
        initialisation : bool, optional
            Indicates if this is during initialization (default is False).

        Returns
        -------
        times : list
            The list of times without duplicate time-value pairs.
        values : list
            The list of values without duplicate time-value pairs.
        """
        indices_to_remove = []

        for index_1, (time, value) in enumerate(zip(times, values)):
            # Find all duplicate times
            if initialisation:
                indices = [
                    index
                    for index, self_time in enumerate(times)
                    if self_time == time and index > index_1
                ]
            else:
                indices = [
                    index
                    for index, self_time in enumerate(self.times)
                    if self_time == time
                ]
            # Check whether found duplicate times are not jumps
            for index in indices:
                if initialisation:
                    if len(values[index]) == len(value):
                        if np.allclose(values[index], value):
                            indices_to_remove.insert(0, index_1)
                else:
                    if len(self.values[index]) == len(value):
                        if np.allclose(self.values[index], value):
                            indices_to_remove.insert(0, index_1)

        for index in indices_to_remove:
            del times[index]
            del values[index]

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
                f"More than one jump occurs at time(s): \
                             {np.array(self.times)[jump_indices]}"
            )

    def _check_dimension(self) -> None:
        """
        Checks if the function changes dimension unexpectedly.

        Checks if the function changes dimension at a time where no two values
        are given, i.e. the one before and the one after the change of
        dimension.

        Raises
        ------
        ValueError
            If the function changes dimension unexpectedly.
        """
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_values = self.dimension()

        for i in range(len(jump_indices) - 1):

            start = jump_indices[i] + 1
            end = jump_indices[i + 1] + 1

            dimension = dimension_values[start]

            for j in range(start + 1, end):
                if dimension_values[j] != dimension:
                    raise ValueError(
                        f"Function changes dimension at \
                                     {self.times[j]} without specified jump"
                    )

    def _interpolate_value(self, t: float) -> float:
        """
        Interpolates the value of the function at a given time.

        Parameters
        ----------
        t : float
            The time at which to interpolate the value.

        Returns
        -------
        value : float
            The interpolated value.

        Raises
        ------
        ValueError
            If the time is outside the range of defined times.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(
                f"t is not between {self.times[0]} and \
                             {self.times[-1]} (first and last value of times)"
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
            The new time(s) to add.
        values : float, list or array-like
            The new value(s) corresponding to the times.
        checks : bool, optional
            If True, sanity checks are performed.
        """
        # Convert everything to lists
        if np.isscalar(times):
            single_value = True
        else:
            single_value = False
        times = self._convert_to_list(times)
        values = self._convert_to_list(values)
        if not single_value:
            for index, value in enumerate(values):
                values[index] = self._convert_to_list(value)
        else:
            values = [values]

        if checks:
            times, values = self._delete_double_values(times, values)
        elif len(self.times) > 0:
            if self.times[-1] == times[0] and len(self.values[-1]) == len(
                values[0]
            ):
                if np.allclose(self.values[-1], values[0]):
                    times = times[1:]
                    values = values[1:]

        self.times = self.times + times
        self.values = self.values + values

        # Sanity checks
        if checks:
            self._sort()
            self._check_multiple_jumps()
            self._check_dimension()

    def delete_values(
        self,
        times: float | list | None = None,
        values: list | None = None,
        indices: int | list | None = None,
    ) -> None:
        """
        Deletes time-value pairs from the function.

        If nothing is provided to the function, it deletes the last time-value
        pair. Otherwise, the time and/or value and/or index of the time-value
        pair to be deleted, can be provided.

        Parameters
        ----------
        times : float or list, optional
            The time(s) to delete.
        values : list, optional
            The value(s) to delete.
        indices : int or list, optional
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
                            f"Time {times} not found with value \
                                         {values}"
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
                        f"Times at index {indices} does not \
                                     have value {times}"
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
                        f"Times or values at index {indices} \
                                     does not have value {times} or {values},\
                                         respectively"
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
                for time, index in zip(times, indices):
                    self.delete_values(time, indices=index)
            else:
                for time, value, index in zip(times, values, index):
                    self.delete_values(time, value, index)

        # If time is not stores, raise an error
        elif isinstance(
            times,
            (
                int,
                float,
                np.float16,
                np.float32,
                np.float64,
                np.int16,
                np.int32,
                np.int64,
            ),
        ):
            raise ValueError(f"{times} is not in the times of the function")

        elif values in self.values:
            # If value and index are provided, delete this if they match
            if isinstance(
                indices,
                (
                    int,
                    float,
                    np.float16,
                    np.float32,
                    np.float64,
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
        elif isinstance(values, (list, np.ndarray)):
            if indices is None:
                for value in values:
                    self.delete_values(values=value)
            else:
                for value, index in zip(values, indices):
                    self.delete_values(values=value, indices=index)

        # If value is provided, that is not stored, raise an error
        elif values is not None:
            raise ValueError("Value is stored as a value of the function")

        # If an index is provided, delete the corresponding time-value pair
        elif isinstance(indices, (int, np.int16, np.int32, np.int64)):
            if indices > len(self.times):
                raise ValueError(
                    f"Index {indices} is greater than number of\
                                 stored time-value pairs ({len(self.times)})"
                )
            del self.times[indices]
            del self.values[indices]

        # If multiple indices are provided, delete the corresponding time-value
        # pairs
        elif isinstance(indices, (list, np.ndarray)):
            for index in indices:
                self.delete_values(indices=index)

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
            return -np.infty

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
            return np.infty

    def jump_times(
        self, return_index: bool = False
    ) -> list[float] | tuple[list[float], list[int]]:
        """
        Returns the times at which jumps occur in the function.

        Parameters
        ----------
        return_index : bool, optional
            If True, also return the indices of the jump times.

        Returns
        -------
        jump_times : list
            The times at which jumps occur.
        indices : list, optional
            The indices of the jump times, if requested.
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
            If True, also return the indices of the dimension jump times.

        Returns
        -------
        dimension_jump_times : list
            The times at which dimension jumps occur.
        indices : list, optional
            The indices of the dimension jump times, if requested.
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
            If True, also return the indices of the non-dimension jump
            times.

        Returns
        -------
        non_dimension_jump_times : list
            The jump times at which there are no dimension jumps.
        non_dimension_jump_indices : list, optional
            The indices of the non-dimension jump times, if requested.
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
                jump_indices[~np.isin(jump_indices, dimension_jump_indices)]
            )
            return non_dimension_jump_times, non_dimension_jump_indices
        else:
            return non_dimension_jump_times

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
    ) -> plt.Figure:
        """
        Plots the coordinate values of the function against time.

        Parameters
        ----------
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        labels : list, optional
            Custom labels for the plot lines.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        separate_plots : bool, optional
            If True, create separate plots for each dimension.
        sharex : bool, optional
            If True, share the x-axis among separate plots.
        sharey : bool, optional
            If True, share the y-axis among separate plots.
        grid : bool, optional
            If True, show grid lines on the plot.
        show_jump_times : bool, optional
            If True, indicate jump times on the plot.
        show_dimension_jump_times : bool, optional
            If True, indicate dimension jump times on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.
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

        return fig

    def plot_norm(
        self,
        order: int = 2,
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
        order : int or list, optional
            The order of the norm to plot. Default is the 2-norm.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        labels : list, optional
            Custom labels for the plot lines.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.
        show_jump_times : bool, optional
            If True, indicate jump times on the plot.
        show_dimension_jump_times : bool, optional
            If True, indicate dimension jump times on the plot.

        Returns
        -------
        fig: FigureObject
            The figure object created.
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

    def norm(
        self,
        t: float | ArrayLike | None = None,
        order: int | None = None,
        return_times: bool = False,
    ) -> list[float] | tuple[list[float], list[float]]:
        """
        Computes the norm of the function values at given time(s).

        Computes the norm of the function values at given time(s) or at all
        times, if no time is provided.

        Parameters
        ----------
            t : float, array-like, or None, optional
                The time point(s) at which to evaluate the norm. If None is
                given, norm is evaluated at all times in self.times. The
                default is None.
            order : int or None, optional
                The order of the norm to compute.
            return_times : bool, optional
                If True, return the corresponding times with the norms.

        Returns
        -------
            value_norms : list
                The computed norms.
            times : list, optional
                The corresponding times, if requested.
        """
        if t is None:
            values = self.values
        else:
            values = self(t)

        value_norms = [float(la.norm(value, ord=order)) for value in values]

        if return_times:
            return value_norms, self.times

        return value_norms

    def dimension(self, t: float | ArrayLike | None = None) -> list[int]:
        """
        Returns the dimension of the function at given time(s).

        Returns the dimension of the function at the given time(s) or the
        dimensions at all times, if no time is provided.

        Parameters
        ----------
        t : float, array-like, or None, optional
            The time point(s) at which to evaluate the dimension.

        Returns
        -------
        dimension_values : list
            The dimensions at the requested times.

        Raises
        ------
        TypeError
            If t is not array-like nor a number.
        """
        # If t is not provided, return dimension for all times
        if t is None:
            dimension_values = [len(value) for value in self.values]
            # dimension_values = np.fromiter(
            #     (len(v) for v in self.values), dtype=int
            # )
            return dimension_values

        # If t is a number, return the dimension at that time
        elif isinstance(
            t,
            (
                int,
                float,
                np.float16,
                np.float32,
                np.float64,
                np.int16,
                np.int32,
                np.int64,
            ),
        ):
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

    def left_limit(self, t: float | ArrayLike):
        """
        Returns the left limit(s) of the function at given time(s).

        Parameters
        ----------
        t : float or array-like
            The time(s) at which to evaluate the left limit(s).

        Returns
        -------
        value : list
            The value(s) of the left limit(s).
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
                    "Dimensions of function do not coincide (only\
                                 possible when last time for first function\
                                 and first time for second function are\
                                 the same)"
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
                    "Dimensions of function do not coincide (only\
                                 possible when last time for first function\
                                 and first time for second function are\
                                 the same)"
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
                                "Functions do not coincide at \
                                             overlapping times"
                            )
                    else:
                        if self.dimension(t) != other.dimension(t):
                            raise ValueError(
                                "Functions do not coincide at \
                                             overlapping times"
                            )

            func = FunctionOfVariableDimension(
                self.times + other.times, self.values + other.values
            )

        return func

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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
            func = FunctionOfVariableDimension(self.times, self.values)
            for i, value in enumerate(func.values):
                func.values[i] = list(np.array(value) + other)
            return func
        # If array is added, add every component to the respective coordinate
        elif isinstance(other, (list, np.ndarray)):
            array_length = len(other)
            for value in other:
                if not isinstance(
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only add list/array of numbers")
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
        Adds another value or function to the values of this function.

        Parameters
        ----------
        other : float, or array-like, or FunctionOfVariableDimension
            The value or function to add to the values of this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            A new instance of the function with the added values.
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
            If dimensions do not match when subtracting another function or
            an array.
        """
        # If number is subtracted, subtract it from every value
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        # If number is divide, divide it by every value
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        Computes the modulus of this function values with other ones.

        Computes the modulus of the values of this function with another value
        or function.

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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        # If modulus to another funtion is compute, compute modulus of function
        # values at every time, if possible
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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        Raises the function values to the power of other values.

        Raises the values of this function to the power of another value or
        function.

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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        if isinstance(
            other,
            (
                int,
                float,
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ):
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
                    value,
                    (
                        int,
                        float,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                ):
                    raise ValueError("Can only subtract list/array of numbers")
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
        Compare this function with another one for equality.

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
        Compare this function with another one for inequality.

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
        Compare this function with another one for being greater.

        Compare this function with another one for being greater at all times
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
        Compare this function with another one for being less.

        Compare this function with another one for being less at all times and
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
        Compare this function with another one for being greater or equal.

        Compare this function with another one for being greater or equal at
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
        Compare this function with another one for being less or equal.

        Compare this function with another one for being less or equal at all
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


if __name__ == "__main__":
    times = [0, 1, 1, 2, 2, 4, 4, 5]
    values = [[0], [1], [0.5, 0.5], [1.5, 2.5], [4], [6], [8], [9]]
    func = FunctionOfVariableDimension(times, values)
    jt, index = func.jump_times(return_index=True)
    djt = func.dimension_jump_times()
    ndjt = func.non_dimension_jump_times()
    print(jt, index)
    print(djt)
    print(ndjt)
    # func.delete_values(4)
    y = func(1)
    print(y)
    y = func.left_limit(1)
    print(y)
    func.plot_coordinates()
    norm = func.norm()
    print(norm)
    print(func.dimension(1.5))
    func.plot_norm(order=1)
    t_values = np.arange(0, 1.01, 0.05)
    x_values = t_values**2
    func.add_values(t_values, x_values)
    func.delete_values(4, [6])
    func.plot_coordinates()
    func = -func
    func.plot_coordinates()
    func = func + 10
    func.plot_coordinates()
    func = func * 2
    func.plot_coordinates()
    func = 1 / func
    func.plot_coordinates()
    times = [0, 1, 1, 1, 2, 2, 4]
    values = [[0], [1], [1], [0.5, 0.5], [1.5, 2.5], [4], [6]]
    func = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(2, 3)

    times = [0, 1, 1, 2, 2, 4]
    values = [[0], [2], [1, 1], [3, 5], [8], [12]]
    times_2 = [4, 5, 6]
    values_2 = [[2, 2, 2], [4, 3, 1], [6, 4, 0]]

    func_new = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(times_2, values_2)

    func_conc = func.concatenate(func_2)

    func_test = func + func_new

    func.plot_coordinates()
    func_new.plot_coordinates()
    func_test.plot_coordinates()
    func_2.plot_coordinates()
    func_conc.plot_coordinates()

    test = func([1.5, 3])
    print(test)

    stats = StatsFunctionsOfVariableDimension(
        (0, 4), 0.25, order=[1, 2], collect_jump_times=True
    )
    stats.add_function(func)
    stats.add_function(func_new)
    print(stats.times)
    print(stats.norm_means)
    print(stats.norm_variances)
    stats.plot_norm([1, 2], legend=True)
    stats.plot_dimension(legend=True)
    stats.plot_jump_times(number=[1, 2], legend=True)

    times = [0, 1, 1, 2, 2, 3]
    values = [
        [1 - 0],
        [1 - 0.666],
        [1 - 0.666, 1 - 0],
        [1 - 1, 1 - 0.333],
        [1 - 0.333],
        [1 - 1],
    ]

    func = FunctionOfVariableDimension(times, values)
    fig = func.plot_coordinates(show_jump_times=True)
    plt.figure(fig)
    plt.axhline(0, color="black")
    plt.axhline(1, color="black")
    plt.xlim(0, 2.5)
    plt.savefig("cadlag_function.png")
    plt.show()
    print(func.left_limit([1, 2]))
    func2 = FunctionOfVariableDimension(times, values)
    print(func == func2)
    func3 = func2 + 1
    print(func3 == func)
