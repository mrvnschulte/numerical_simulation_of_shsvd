"""
Provides a representation of functions whose dimension may change over time.

The module supports interpolation, jump detection, arithmetic,
comparison, and plotting of functions represented by time-value pairs.
"""

from __future__ import annotations
import bisect

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as la

from ._utils import _convert_to_list


class FunctionOfVariableDimension:
    """
    Represents a function whose output dimension may change over time.

    The function is represented by time-value pairs. Two consecutive
    entries with the same time represent a jump: the first value is the
    left limit and the second value is the value at the jump time. Between
    stored times, values are interpolated linearly.

    Attributes
    ----------
    times : list of float or np.ndarray
        Stored time points.
    values : list of array-like or np.ndarray
        Function values corresponding to times.
    """

    def __init__(
        self,
        times: float | ArrayLike | None = None,
        values: float | ArrayLike | None = None,
        checks: bool = True,
    ) -> None:
        """
        Initializes the function from time-value pairs.

        Parameters
        ----------
        times : float, list, np.ndarray, or None, optional
            Stored time points. If both times and values are None, an empty
            function is created. The default is None.
        values : float, list, np.ndarray, or None, optional
            Values corresponding to times. Scalar values are converted to
            one-dimensional values. The default is None.
        checks : bool, optional
            If True, remove duplicate pairs, sort the data, and check jump
            multiplicity and dimensions. The default is True.

        Raises
        ------
        TypeError
            If times or values have an unsupported type or contain nonnumeric
            entries.
        ValueError
            If times and values have different lengths, multiple jumps occur at
            one time, or a dimension change occurs without a jump.
        """
        if times is None and values is None:
            self.times = []
            self.values = []

        else:
            if checks or np.isscalar(times) or np.isscalar(values):
                times = _convert_to_list(times)
                values = _convert_to_list(values)

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
                    values[index] = _convert_to_list(value)
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
        Evaluates the function at given time points.

        Values are interpolated linearly between stored points. At a jump time,
        the value after the jump is returned.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Time point or time points at which to evaluate the function.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated value for a scalar time, or one value per requested time.

        Raises
        ------
        ValueError
            If a requested time lies outside the stored time range.
        """
        if isinstance(t, (list, np.ndarray)):
            value = [self._interpolate_value(time) for time in t]
        else:
            value = self._interpolate_value(t)

        return value

    def add_values(
        self,
        times: float | ArrayLike,
        values: float | ArrayLike,
        checks: bool = True,
    ) -> None:
        """
        Adds time-value pairs to the function.

        Parameters
        ----------
        times : float, list, or np.ndarray
            Time point or time points to add.
        values : float, list, or np.ndarray
            Corresponding value or values. For a scalar time, values represents
            one state; otherwise it contains one state per time point.
        checks : bool, optional
            If True, remove duplicate pairs, sort the data, and check jump
            multiplicity and dimensions. The default is True.

        Raises
        ------
        TypeError
            If an input cannot be converted to a list.
        ValueError
            If validation detects multiple jumps at one time or a dimension
            change without a jump.

        Notes
        -----
        With checks=False, a first new pair approximately equal to the last
        stored pair is skipped using np.allclose. No sorting or further
        validation is performed.

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
        times = _convert_to_list(times)
        values = _convert_to_list(values)

        if is_scalar:
            values = [values]
        else:
            values = [_convert_to_list(v) for v in values]

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
        Deletes stored time-value pairs.

        If no arguments are provided, the last pair is deleted. Otherwise,
        pairs are selected by time, value, index, or a combination of these.

        Parameters
        ----------
        times : float, list of float, or None, optional
            Time or times to delete. The default is None.
        values : array-like, list of array-like, or None, optional
            Value or values to delete. The default is None.
        indices : int, list of int, or None, optional
            Index or indices to delete. The default is None.

        Raises
        ------
        ValueError
            If a requested time, value, or index cannot be matched, or if a
            dimension jump time is selected by time.
        TypeError
            If the input formats cannot be combined.

        Notes
        -----
        Deleting a non-dimension jump by time alone removes both entries.
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
        Returns the last stored time.

        Returns
        -------
        float
            Last stored time, or negative infinity if the function is empty.
        """
        if len(self.times) > 0:
            return self.times[-1]
        else:
            return -np.inf

    def minimal_time(self) -> float:
        """
        Returns the first stored time.

        Returns
        -------
        float
            First stored time, or positive infinity if the function is empty.
        """
        if len(self.times) > 0:
            return self.times[0]
        else:
            return np.inf

    def dimension(self, t: float | ArrayLike | None = None) -> int | list[int]:
        """
        Returns the dimension at given time points.

        Parameters
        ----------
        t : float, list, tuple, np.ndarray, or None, optional
            Time point or points at which to determine the dimension. If None,
            return the dimension at every stored entry. The default is None.

        Returns
        -------
        dimension : int or list of int
            Dimension at a scalar time, or a list of dimensions for multiple
            times or for all stored entries.

        Raises
        ------
        TypeError
            If t is neither a supported scalar nor a supported sequence.
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
        Returns the stored jump times.

        A jump is identified by a time occurring exactly twice in self.times.

        Parameters
        ----------
        return_index : bool, optional
            If True, also return the index of the first entry for each jump.
            The default is False.

        Returns
        -------
        jump_times : list of float
            Jump times in ascending order.
        indices : list of int, optional
            Indices of the left-limit entries. Returned only if return_index is
            True.
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
        Returns jump times at which the dimension changes.

        Parameters
        ----------
        return_index : bool, optional
            If True, also return the index of the first entry for each jump.
            The default is False.

        Returns
        -------
        times : list of float
            Dimension jump times.
        indices : list of int, optional
            Indices of the left-limit entries. Returned only if return_index is
            True.
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
        Returns jump times at which the dimension does not change.

        Parameters
        ----------
        return_index : bool, optional
            If True, also return the index of the first entry for each jump.
            The default is False.

        Returns
        -------
        times : list of float
            Jump times without a dimension change.
        indices : list of int, optional
            Indices of the left-limit entries. Returned only if return_index is
            True.
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
        Computes norms of the function values.

        Parameters
        ----------
        t : float, list, np.ndarray, or None, optional
            Evaluation time or times. If None, the stored values are used,
            including both values at jumps. The default is None.
        order : int, float, or None, optional
            Norm order passed to scipy.linalg.norm. If None, the default
            Euclidean norm is used. The default is None.
        return_times : bool, optional
            If True, also return self.times. The default is False.

        Returns
        -------
        value_norms : list of float
            Norms of the selected values.
        times : list of float or np.ndarray, optional
            Stored times. Returned only if return_times is True.

        Notes
        -----
        When t is provided, the returned times are still self.times rather than
        the requested evaluation times.
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
        Returns left-limit values at given time points.

        At a jump time, the first stored value is returned. At other times, the
        function is evaluated normally.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Time point or time points at which to evaluate the left limit.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Left-limit value for a scalar time, or one value per requested time.

        Raises
        ------
        ValueError
            If a requested non-jump time lies outside the stored time range.
        """
        if isinstance(t, (list, np.ndarray)):
            value = [self.left_limit(time) for time in t]
        elif t in self.jump_times():
            value = np.array(self.values[self.times.index(t)])
        else:
            value = self(t)

        return value

    def concatenate(
        self, other: FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Concatenates this function with another function of variable dimension.

        The functions may meet at a common endpoint, have a gap with compatible
        dimensions, or overlap subject to the implementation's consistency
        checks.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            Function to concatenate with this function.

        Returns
        -------
        func : FunctionOfVariableDimension
            Concatenated function. If either function is empty, the other object
            is returned directly.

        Raises
        ------
        TypeError
            If other is not a FunctionOfVariableDimension.
        ValueError
            If the checked values or dimensions are incompatible.
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
        labels_by_dimension: dict[int, list[str]] | None = None,
        colors_by_dimension: dict[int, list[str]] | None = None,
    ) -> plt.Figure | list[plt.Figure]:
        """
        Plot the coordinates of the function over time.

        Coordinate labels and colors can be specified separately for each
        state dimension. Use the same label and color in different dimensions
        to identify a coordinate whose index changes after a dimension jump.

        Parameters
        ----------
        title : str, optional
            Title of the combined plot. The default is
            "Plot of the coordinates".
        legend : bool, optional
            If True, show a legend. The default is True.
        labels : list of str or None, optional
            Default coordinate labels, ordered by coordinate index. The list
            must contain at least as many entries as the maximum state dimension.
            If None or empty, labels are generated automatically. These labels
            are used for dimensions not present in labels_by_dimension.
        xlabel : str, optional
            Label of the horizontal axis. The default is "Time".
        ylabel : str, optional
            Label of the vertical axis. The default is "Value".
        separate_plots : bool, optional
            If True, also create one figure per distinct coordinate label and
            return those figures instead of the combined figure. The default
            is False.
        sharex : bool, optional
            If True, apply the combined plot's horizontal limits to the
            separate plots. The default is True.
        sharey : bool, optional
            If True, apply the combined plot's vertical limits to the
            separate plots. The default is True.
        grid : bool, optional
            If True, display grid lines. The default is False.
        show_jump_times : bool, optional
            If True, mark all jump times with red dashed lines. The default
            is False.
        show_dimension_jump_times : bool, optional
            If True, mark dimension jump times with green dashed lines.
            A dimension jump is also marked in red when show_jump_times is
            True. The default is False.
        labels_by_dimension : dict of int to list of str or None, optional
            Coordinate labels for selected state dimensions. Each dictionary
            key is a state dimension, and the corresponding list must contain
            exactly that many labels in state-vector order. Equal labels across
            dimensions share one legend entry and one separate plot.
        colors_by_dimension : dict of int to list of str or None, optional
            Coordinate colors for selected state dimensions. Each dictionary
            key is a state dimension, and the corresponding list must contain
            exactly that many Matplotlib-compatible colors in state-vector
            order. Dimensions not present in the dictionary use the default
            Matplotlib color cycle.

        Returns
        -------
        fig : plt.Figure or list of plt.Figure
            The combined figure, or the list of separate coordinate figures
            if separate_plots is True. The combined figure is also created
            when separate plots are requested.

        Raises
        ------
        ValueError
            If the function is empty, the default labels list is too short,
            or a dimension-specific labels or colors list has the wrong length.

        Notes
        -----
        Coordinates are drawn separately on either side of a dimension jump;
        no line is drawn across the jump. Labels are used to identify the
        same coordinate across dimensions. For consistent visual appearance,
        assign the same color to equal labels in every dimension.

        The method calls matplotlib.pyplot.show().
        """
        if len(self.values) == 0:
            raise ValueError("Cannot plot an empty function.")

        dimensions = self.dimension()
        max_dimension = max(dimensions)

        # Prepare default labels for dimensions without explicit labels.
        if labels is None or len(labels) == 0:
            labels = [f"Coordinate {i + 1}" for i in range(max_dimension)]
        elif len(labels) < max_dimension:
            raise ValueError(
                "labels must contain at least as many entries as the "
                "maximum state dimension."
            )

        # Validate the dimension-specific lists.
        for mapping, name in (
            (labels_by_dimension, "labels_by_dimension"),
            (colors_by_dimension, "colors_by_dimension"),
        ):
            if mapping is not None:
                for dimension, entries in mapping.items():
                    if len(entries) != dimension:
                        raise ValueError(
                            f"{name}[{dimension}] must contain exactly "
                            f"{dimension} entries."
                        )

        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Create the combined plot and keep one legend handle per label.
        fig, ax = plt.subplots()
        legend_lines = {}

        # Separate figures are grouped by label rather than coordinate index.
        figures = {}
        separate_axes = {}

        _, dimension_jump_indices = self.dimension_jump_times(
            return_index=True
        )

        # Each segment contains states of one dimension. Both sides of a
        # dimension jump are plotted in their respective segments.
        segment_starts = [0] + [index + 1 for index in dimension_jump_indices]
        segment_ends = [index + 1 for index in dimension_jump_indices]
        segment_ends.append(len(self.values))

        for start, end in zip(segment_starts, segment_ends):
            current_dimension = dimensions[start]

            if (
                labels_by_dimension is not None
                and current_dimension in labels_by_dimension
            ):
                current_labels = labels_by_dimension[current_dimension]
            else:
                current_labels = labels[:current_dimension]

            if (
                colors_by_dimension is not None
                and current_dimension in colors_by_dimension
            ):
                current_colors = colors_by_dimension[current_dimension]
            else:
                current_colors = [
                    default_colors[i % len(default_colors)]
                    for i in range(current_dimension)
                ]

            times = self.times[start:end]
            values = np.asarray(self.values[start:end])

            for i in range(current_dimension):
                label = current_labels[i]
                color = current_colors[i]

                # Draw one line per coordinate to assign its color explicitly.
                (line,) = ax.plot(times, values[:, i], color=color)

                if label not in legend_lines:
                    legend_lines[label] = line

                if separate_plots:
                    if label not in figures:
                        figures[label], separate_axes[label] = plt.subplots()

                    separate_axes[label].plot(times, values[:, i], color=color)

        # Add titles, axis labels, and grid to the combined plot.
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid()

        # Add markers without changing the existing vertical axis limits.
        def add_jump_markers(plot_ax):
            marker_handles = []
            marker_labels = []
            ylim = plot_ax.get_ylim()

            if show_jump_times:
                jump_times = self.jump_times()
                if len(jump_times) > 0:
                    lines = plot_ax.vlines(
                        jump_times,
                        ylim[0],
                        ylim[1],
                        colors="red",
                        linestyles="--",
                    )
                    marker_handles.append(lines)
                    marker_labels.append("Jump times")

            if show_dimension_jump_times:
                jump_times = self.dimension_jump_times()
                if len(jump_times) > 0:
                    lines = plot_ax.vlines(
                        jump_times,
                        ylim[0],
                        ylim[1],
                        colors="green",
                        linestyles="--",
                    )
                    marker_handles.append(lines)
                    marker_labels.append("Dimension jump times")

            plot_ax.set_ylim(ylim)
            return marker_handles, marker_labels

        marker_handles, marker_labels = add_jump_markers(ax)

        if legend:
            ax.legend(
                list(legend_lines.values()) + marker_handles,
                list(legend_lines.keys()) + marker_labels,
            )

        # Configure each separate figure after all its segments have been drawn.
        if separate_plots:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for label, separate_ax in separate_axes.items():
                if title:
                    separate_ax.set_title(f"{label} plot")
                if xlabel:
                    separate_ax.set_xlabel(xlabel)
                if ylabel:
                    separate_ax.set_ylabel(ylabel)
                if grid:
                    separate_ax.grid()
                if sharex:
                    separate_ax.set_xlim(xlim)
                if sharey:
                    separate_ax.set_ylim(ylim)

                marker_handles, marker_labels = add_jump_markers(separate_ax)
                if legend and marker_handles:
                    separate_ax.legend(marker_handles, marker_labels)

        plt.show()

        if separate_plots:
            return list(figures.values())
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
        Plots norms of the stored function values over time.

        Parameters
        ----------
        order : int, float, or list of int or float, optional
            Norm order or orders passed to scipy.linalg.norm. The default is 2.
        title : str, optional
            Plot title. The default is 'Plot of the norm'.
        legend : bool, optional
            If True, show a legend. The default is True.
        labels : list of str or None, optional
            Labels for the norms. If None or empty, labels are generated
            automatically. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. The default is 'Value'.
        grid : bool, optional
            If True, show grid lines. The default is False.
        show_jump_times : bool, optional
            If True, mark jump times. The default is False.
        show_dimension_jump_times : bool, optional
            If True, mark dimension jump times. The default is False.

        Returns
        -------
        fig : plt.Figure
            Created figure.
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
        Negates the stored function values.

        Returns
        -------
        func : FunctionOfVariableDimension
            New function with negated values.
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
        Adds a scalar, array, or function to this function.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Operand to add. Arrays are applied coordinate-wise and must be at
            least as long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Adds this function to a scalar or array.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Left-hand operand.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.
        """
        return self + other

    def __sub__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Subtracts a scalar, array, or function from this function.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Operand to subtract. Arrays are applied coordinate-wise and must be
            at least as long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Subtracts this function from a scalar or array.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Left-hand operand.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.
        """
        return -self + other

    def __mul__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Multiplies this function by a scalar, array, or function.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Operand to multiply by. Arrays are applied coordinate-wise and must
            be at least as long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Multiplies a scalar or array by this function.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Left-hand operand.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.
        """
        return self * other

    def __truediv__(
        self, other: float | ArrayLike | FunctionOfVariableDimension
    ) -> FunctionOfVariableDimension:
        """
        Divides this function by a scalar, array, or function.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Divisor. Arrays are applied coordinate-wise and must be at least as
            long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Divides a scalar or array by this function coordinate-wise.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Numerator.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting values.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is shorter than a state.
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
        Computes the coordinate-wise remainder of this function.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Divisor. Arrays must be at least as long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting remainders.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Computes the coordinate-wise remainder of a scalar or array modulo this
        function.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Left-hand operand.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting remainders.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is shorter than a state.
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
        Raises this function to a scalar, array, or function coordinate-wise.

        Parameters
        ----------
        other : float, list, np.ndarray, or FunctionOfVariableDimension
            Exponent. Arrays must be at least as long as every state.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting powers.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is too short or the functions have incompatible time
            ranges or dimensions.
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
        Raises a scalar or array to the values of this function coordinate-wise.

        Parameters
        ----------
        other : float, list, or np.ndarray
            Base of the exponentiation.

        Returns
        -------
        func : FunctionOfVariableDimension
            Function with the resulting powers.

        Raises
        ------
        TypeError
            If an array contains nonnumeric entries.
        ValueError
            If an array is shorter than a state.
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

    def __eq__(self, other: object) -> bool:
        """
        Compares this function with another object for equality.

        For another FunctionOfVariableDimension, the time ranges, dimensions,
        values at stored times, and left limits at jump times are compared.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            True if the checked values agree; False if they differ or other is
            not a FunctionOfVariableDimension.
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

    def __ne__(self, other: object) -> bool:
        """
        Compares this function with another object for inequality.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            Negation of the equality comparison.
        """
        return not self == other

    def __gt__(self, other: FunctionOfVariableDimension) -> bool:
        """
        Checks whether this function is greater than another function.

        Values and left limits are compared coordinate-wise at the stored times
        and jump times of both functions.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            Function to compare with.

        Returns
        -------
        bool
            True if all checked coordinates are strictly greater.

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
        Checks whether this function is less than another function.

        Values and left limits are compared coordinate-wise at the stored times
        and jump times of both functions.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            Function to compare with.

        Returns
        -------
        bool
            True if all checked coordinates are strictly less.

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
        Checks whether this function is greater than or equal to another function.

        Values and left limits are compared coordinate-wise at the stored times
        and jump times of both functions.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            Function to compare with.

        Returns
        -------
        bool
            True if all checked coordinates are greater than or equal to the
            corresponding coordinates of other.

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
        Checks whether this function is less than or equal to another function.

        Values and left limits are compared coordinate-wise at the stored times
        and jump times of both functions.

        Parameters
        ----------
        other : FunctionOfVariableDimension
            Function to compare with.

        Returns
        -------
        bool
            True if all checked coordinates are less than or equal to the
            corresponding coordinates of other.

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
        Sorts the stored time-value pairs by time.

        The relative order of entries with equal times is preserved. The stored
        times and values are replaced in-place.
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

        Pairs are considered duplicates if their times are equal, their values
        have the same length, and their values are close according to
        np.allclose.

        Parameters
        ----------
        times : list of float
            Candidate time points.
        values : list of array-like
            Candidate values corresponding to times.
        initialisation : bool, optional
            If True, compare candidate pairs with one another. Otherwise,
            compare them with the stored pairs. The default is False.

        Returns
        -------
        times : list of float
            Candidate times after duplicate removal.
        values : list of array-like
            Candidate values after duplicate removal.
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
        Checks whether more than two entries share a time point.

        Raises
        ------
        ValueError
            If a time occurs more than twice.
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
        Checks that dimension changes occur only at stored jump times.

        Raises
        ------
        ValueError
            If the dimension changes within a segment without a jump.
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
        Linearly interpolates the function at a scalar time.

        At a jump time, the value after the jump is used.

        Parameters
        ----------
        t : float
            Time at which to evaluate the function.

        Returns
        -------
        value : np.ndarray
            Interpolated value.

        Raises
        ------
        ValueError
            If t lies outside the stored time range.
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
