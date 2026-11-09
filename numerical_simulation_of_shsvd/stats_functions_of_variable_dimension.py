"""
Provides statistics for functions of variable dimension.

The module collects and evaluates statistics of norms, dimensions, custom
evaluations, and jump times across multiple functions.
"""

from __future__ import annotations
import bisect
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .function_of_variable_dimension import FunctionOfVariableDimension
from ._utils import _is_sublist, _list_difference, _partial_t_eval


class StatsFunctionsOfVariableDimension:
    """
    Collects statistics of functions of variable dimension.

    Means and sample variances are updated incrementally as functions are
    added. Norm and custom evaluation statistics are stored at fixed
    evaluation times. Dimension and jump-time collection are optional.

    Attributes
    ----------
    evaluated_functions : int
        Number of functions added to the collector.
    times : np.ndarray
        Evaluation times.
    order : list
        Norm orders for which statistics are collected.
    norm_means : np.ndarray
        Mean norms, with shape (len(order), len(times)).
    norm_variances : np.ndarray
        Sample variances of the norms, with the same shape as norm_means.
    norm_m2 : np.ndarray
        Sums of squared deviations used to update norm variances.
    collect_dimension : bool
        Whether dimension statistics are collected.
    dimension_means : np.ndarray
        Mean dimensions. Available if collect_dimension is True.
    dimension_variances : np.ndarray
        Sample variances of the dimensions. Available if collect_dimension
        is True.
    dimension_m2 : np.ndarray
        Sums of squared deviations for the dimensions. Available if
        collect_dimension is True.
    custom_evaluation_functions : list of callable
        Registered custom evaluation functions.
    cef_means : list or np.ndarray
        Mean custom evaluations. Converted from a list to an array when
        the first function is added.
    cef_variances : list or np.ndarray
        Sample variances of custom evaluations.
    cef_m2 : list or np.ndarray
        Sums of squared deviations for custom evaluations.
    collect_jump_times : bool
        Whether jump times are collected.
    jump_times : list of lists
        Jump times for each added function. Available if
        collect_jump_times is True.
    collect_dimension_jump_times : bool
        Whether dimension jump times are collected.
    dimension_jump_times : list of lists
        Dimension jump times for each added function. Available if
        collect_dimension_jump_times is True.

    Notes
    -----
    For a single added function, the stored variances are set to zero.
    """

    def __init__(
        self,
        t_span: tuple[float, float],
        stepsize: float = 1,
        t_eval: ArrayLike | None = None,
        order: int | ArrayLike = 2,
        collect_dimension: bool = True,
        collect_jump_times: bool = False,
        collect_dimension_jump_times: bool = False,
    ) -> None:
        """
        Initializes the statistics collector.

        Parameters
        ----------
        t_span : tuple of float
            Start and end times of the evaluation interval.
        stepsize : float, optional
            Step size used to construct evaluation times when t_eval is None.
            The default is 1.
        t_eval : array-like or None, optional
            Evaluation times. Times outside t_span are removed and interval
            endpoints are added if missing. The default is None.
        order : int, float, or array-like, optional
            Norm order or orders accepted by numpy.linalg.norm. The default is 2.
        collect_dimension : bool, optional
            If True, collect dimension statistics. The default is True.
        collect_jump_times : bool, optional
            If True, store jump times for each function. The default is False.
        collect_dimension_jump_times : bool, optional
            If True, store dimension jump times for each function. The default
            is False.

        Raises
        ------
        TypeError
            If order is neither a supported number nor a list, tuple, or
            NumPy array.
        """
        # Create self.times according to stepsize or t_eval
        t0, tf = t_span
        decimal_places = (
            len(str(stepsize).split(".")[1]) if "." in str(stepsize) else 0
        )
        if t_eval is not None:
            self.times = _partial_t_eval(t_eval, t0, tf)
        else:
            self.times = np.round(np.arange(t0, tf, stepsize), decimal_places)
        if not np.isclose(self.times[-1], tf):
            self.times = np.append(self.times, tf)

        # Initialize all other attributes
        self.evaluated_functions = 0

        if isinstance(order, (list, np.ndarray, tuple)):
            self.order = list(order)
        elif isinstance(order, (int, float, np.integer, np.floating)):
            self.order = [order]
        else:
            raise TypeError("Order is not array-like nor a number")

        self.norm_means = np.zeros((len(self.order), len(self.times)))
        self.norm_variances = np.zeros((len(self.order), len(self.times)))
        self.norm_m2 = np.zeros((len(self.order), len(self.times)))
        self.collect_dimension = collect_dimension
        if collect_dimension:
            self.dimension_means = np.zeros(len(self.times))
            self.dimension_variances = np.zeros(len(self.times))
            self.dimension_m2 = np.zeros(len(self.times))
        self.cef_means = []
        self.cef_variances = []
        self.cef_m2 = []

        self.collect_jump_times = collect_jump_times
        if self.collect_jump_times:
            self.jump_times = []
        self.collect_dimension_jump_times = collect_dimension_jump_times
        if self.collect_dimension_jump_times:
            self.dimension_jump_times = []

        self.custom_evaluation_functions = []

    def add_function(self, func: FunctionOfVariableDimension) -> None:
        """
        Adds a function and updates the collected statistics.

        The function is evaluated at self.times. Norms, dimensions, and custom
        evaluations are used to update the corresponding means and variances.

        Parameters
        ----------
        func : FunctionOfVariableDimension
            Function to add. Its time range must cover self.times.

        Raises
        ------
        TypeError
            If func is not a FunctionOfVariableDimension.
        ValueError
            If the time range of func does not cover the evaluation times.
        """
        if not isinstance(func, FunctionOfVariableDimension):
            raise TypeError(f"{func} is not a function of variable dimension")

        if func.times[0] > self.times[0] or func.times[-1] < self.times[-1]:
            raise ValueError(
                f"{func} has values between {func.times[0]} and\
                             {func.times[-1]} but they need to be at least \
                                 between {self.times[0]} and {self.times[-1]}"
            )

        if self.evaluated_functions == 0:
            self.cef_means = np.array(self.cef_means)
            self.cef_variances = np.array(self.cef_variances)
            self.cef_m2 = np.array(self.cef_m2)
        self.evaluated_functions += 1

        values = func(self.times)

        # Update norm means and variances
        for index, order in enumerate(self.order):

            norm_values = np.array(
                [np.linalg.norm(value, ord=order) for value in values]
            )
            old_means = self.norm_means[index]
            old_m2 = self.norm_m2[index]

            new_means, new_variances, new_m2 = (
                self._update_means_and_variances(
                    norm_values, old_means, old_m2
                )
            )

            self.norm_means[index] = new_means
            self.norm_m2[index] = new_m2
            self.norm_variances[index] = new_variances

        # Update dimension means and variances
        if self.collect_dimension:
            dimension_values = np.array(func.dimension(self.times))
            old_means = self.dimension_means
            old_m2 = self.dimension_m2

            new_means, new_variances, new_m2 = (
                self._update_means_and_variances(
                    dimension_values, old_means, old_m2
                )
            )

            self.dimension_means = new_means
            self.dimension_m2 = new_m2
            self.dimension_variances = new_variances

        # Update means and variances of custom evaluation functions
        for index, evaluation_func in enumerate(
            self.custom_evaluation_functions
        ):

            cef_values = np.array(evaluation_func(values))
            old_means = self.cef_means[index]
            old_m2 = self.cef_m2[index]

            new_means, new_variances, new_m2 = (
                self._update_means_and_variances(cef_values, old_means, old_m2)
            )

            self.cef_means[index] = new_means
            self.cef_m2[index] = new_m2
            self.cef_variances[index] = new_variances

        # Collect jump times, if desired
        if self.collect_jump_times:
            self.jump_times.append(func.jump_times())
        if self.collect_dimension_jump_times:
            self.dimension_jump_times.append(func.dimension_jump_times())

    def add_custom_evaluation_function(
        self, func: Callable[[list[np.ndarray]], np.ndarray]
    ) -> None:
        """
        Registers a custom evaluation function.

        Parameters
        ----------
        func : callable
            Function receiving a list of state values evaluated at self.times.
            It should return one scalar value per evaluation time.

        Raises
        ------
        ValueError
            If a function has already been added to the collector.
        """
        if self.evaluated_functions > 0:
            raise ValueError(
                "Can only add custom evaluation function if no functions have "
                "been added yet"
            )

        self.custom_evaluation_functions.append(func)

        self.cef_means.append(np.zeros(len(self.times)))
        self.cef_variances.append(np.zeros(len(self.times)))
        self.cef_m2.append(np.zeros(len(self.times)))

    def plot_norm(
        self,
        order: int | float | list[int | float] = 2,
        sigma: float = 2,
        labels: list[str] | None = None,
        title: str = "Plot of mean norm",
        legend: bool = True,
        xlabel: str = "Time",
        ylabel: str = "Norm",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the mean norms with bands of standard deviations.

        Parameters
        ----------
        order : int, float, or list of int or float, optional
            Norm order or orders to plot. They must be stored in self.order.
            The default is 2.
        sigma : float, optional
            Number of standard deviations used for the shaded band. The default
            is 2.
        labels : list of str or None, optional
            Legend labels. If None or empty, labels are generated automatically.
            The default is None.
        title : str, optional
            Plot title. The default is 'Plot of mean norm'.
        legend : bool, optional
            If True, show a legend. The default is True.
        xlabel : str, optional
            Label for the x-axis. The default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. The default is 'Norm'.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            Created figure.

        Raises
        ------
        ValueError
            If a requested norm order is not stored.
        """
        fig = plt.figure()

        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order

        if not labels:
            labels = [f"{order} norm" for order in order_list]

        lines = []

        for order in order_list:
            if order not in self.order:
                raise ValueError(
                    f"{order} is not in self.order and hence was not computed"
                )
            index = self.order.index(order)
            (line,) = plt.plot(self.times, self.norm_means[index])
            plt.fill_between(
                self.times,
                self.norm_means[index]
                - sigma * np.sqrt(self.norm_variances[index]),
                self.norm_means[index]
                + sigma * np.sqrt(self.norm_variances[index]),
                alpha=0.3,
            )
            lines.append(line)

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

        return fig

    def plot_dimension(
        self,
        sigma: float = 2,
        labels: list[str] | None = None,
        title: str = "Plot of mean dimension",
        legend: bool = False,
        xlabel: str = "Time",
        ylabel: str = "Dimension",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the mean dimension with a band of standard deviations.

        Parameters
        ----------
        sigma : float, optional
            Number of standard deviations used for the shaded band. The default
            is 2.
        labels : list of str or None, optional
            Legend labels. If None, the label 'Dimension' is used. The default
            is None.
        title : str, optional
            Plot title. The default is 'Plot of mean dimension'.
        legend : bool, optional
            If True, show a legend. The default is False.
        xlabel : str, optional
            Label for the x-axis. The default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. The default is 'Dimension'.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            Created figure.

        Raises
        ------
        ValueError
            If dimension statistics are not collected.
        """
        if not self.collect_dimension:
            raise ValueError("Cannot plot dimension if it was not collected.")
        fig = plt.figure()

        lines = []

        if labels is None:
            labels = ["Dimension"]

        (line,) = plt.plot(self.times, self.dimension_means)
        plt.fill_between(
            self.times,
            self.dimension_means - sigma * np.sqrt(self.dimension_variances),
            self.dimension_means + sigma * np.sqrt(self.dimension_variances),
            alpha=0.3,
        )
        lines.append(line)

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

        return fig

    def plot_cefs(
        self,
        sigma: float = 2,
        labels: list[str] | None = None,
        title: str = "Plot of cefs",
        legend: bool = False,
        xlabel: str = "Time",
        ylabel: str = "Value",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the means of the custom evaluation functions.

        For each custom evaluation function, a band of standard deviations is
        shown around the mean.

        Parameters
        ----------
        sigma : float, optional
            Number of standard deviations used for the shaded bands. The default
            is 2.
        labels : list of str or None, optional
            Legend labels. If None or empty, labels are generated automatically.
            The default is None.
        title : str, optional
            Plot title. The default is 'Plot of cefs'.
        legend : bool, optional
            If True, show a legend. The default is False.
        xlabel : str, optional
            Label for the x-axis. The default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. The default is 'Value'.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            Created figure.
        """
        fig = plt.figure()

        lines = []

        labels = labels or [
            f"Cef {i+1}" for i in range(len(self.custom_evaluation_functions))
        ]

        for index in range(len(self.custom_evaluation_functions)):
            (line,) = plt.plot(self.times, self.cef_means[index])
            plt.fill_between(
                self.times,
                self.cef_means[index]
                - sigma * np.sqrt(self.cef_variances[index]),
                self.cef_means[index]
                + sigma * np.sqrt(self.cef_variances[index]),
                alpha=0.3,
            )
            lines.append(line)

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

        return fig

    def plot_jump_times(
        self,
        number: int | list[int] | None = None,
        density: bool = False,
        separate: bool = False,
        separate_plots: bool = False,
        title: str | None = None,
        legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots a histogram of collected jump times.

        Parameters
        ----------
        number : int, list of int, or None, optional
            One-based jump index or indices to include. If None or empty, all
            observed indices are included. The default is None.
        density : bool, optional
            If True, normalize the histogram to a probability density. The
            default is False.
        separate : bool, optional
            If True, plot a histogram for each selected index in the current
            figure. The default is False.
        separate_plots : bool, optional
            If True, create a new figure after plotting each selected index.
            The default is False.
        title : str or None, optional
            Plot title. The default is None.
        legend : bool, optional
            If True, show a legend. The default is False.
        xlabel : str or None, optional
            Label for the x-axis. The default is None.
        ylabel : str or None, optional
            Label for the y-axis. The default is None.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            First created figure.

        Raises
        ------
        ValueError
            If jump times are not collected.
        """
        if not self.collect_jump_times:
            raise ValueError(
                f"Jump times have not been in collected in \
                             {self}"
            )

        fig = plt.figure()

        if not number:
            max_jumps = max(len(jump_times) for jump_times in self.jump_times)
            number = np.arange(1, max_jumps + 1)
        if isinstance(number, (list, np.ndarray, tuple)):
            if separate_plots or separate:
                for i, index in enumerate(number):
                    index -= 1
                    jump_times = [
                        item[index]
                        for item in self.jump_times
                        if index < len(item)
                    ]
                    plt.hist(
                        jump_times, density=density, label=f"Jump {index+1}"
                    )
                    if separate_plots and not i + 1 == len(number):
                        plt.figure()
            else:
                jump_times = []
                for index in number:
                    index -= 1
                    jump_times.extend(
                        [
                            item[index]
                            for item in self.jump_times
                            if index < len(item)
                        ]
                    )
                plt.hist(jump_times, density=density, label="Jump times")
        elif isinstance(number, (int, float, np.integer, np.floating)):
            number -= 1
            jump_times = [
                item[int(number)]
                for item in self.jump_times
                if int(number) < len(item)
            ]
            plt.hist(jump_times, density=density, label="Jump times")

        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()

        return fig

    def plot_inter_jump_times(
        self,
        number: list[int] | None = None,
        density: bool = False,
        separate: bool = False,
        separate_plots: bool = False,
        title: str | None = None,
        legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots a histogram of collected inter-jump times.

        For the first jump, the stored jump time itself is used. For subsequent
        jumps, the difference between consecutive jump times is used.

        Parameters
        ----------
        number : int, list of int, or None, optional
            One-based jump index or indices to include. If None or empty, all
            observed indices are included. The default is None.
        density : bool, optional
            If True, normalize the histogram to a probability density. The
            default is False.
        separate : bool, optional
            If True, plot a histogram for each selected index in the current
            figure. The default is False.
        separate_plots : bool, optional
            If True, create a new figure after plotting each selected index.
            The default is False.
        title : str or None, optional
            Plot title. The default is None.
        legend : bool, optional
            If True, show a legend. The default is False.
        xlabel : str or None, optional
            Label for the x-axis. The default is None.
        ylabel : str or None, optional
            Label for the y-axis. The default is None.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            First created figure.

        Raises
        ------
        ValueError
            If jump times are not collected.
        """
        if not self.collect_jump_times:
            raise ValueError(
                f"Jump times have not been in collected in \
                             {self}"
            )

        fig = plt.figure()

        inter_jump_times = []

        for i, jump_times in enumerate(self.jump_times):
            inter_jump_times.append(np.zeros(len(jump_times)))
            if len(jump_times) > 0:
                inter_jump_times[i][0] = jump_times[0]
                if len(jump_times) > 1:
                    inter_jump_times[i][1:] = np.array(
                        jump_times[1:]
                    ) - np.array(jump_times[:-1])

        if not number:
            max_jumps = max(len(jump_times) for jump_times in self.jump_times)
            number = np.arange(1, max_jumps + 1)
        if isinstance(number, (list, np.ndarray, tuple)):
            if separate_plots or separate:
                for i, index in enumerate(number):
                    index -= 1
                    jump_times = [
                        item[index]
                        for item in inter_jump_times
                        if index < len(item)
                    ]
                    plt.hist(
                        jump_times, density=density, label=f"Jump {index+1}"
                    )
                    if separate_plots and not i + 1 == len(number):
                        plt.figure()
            else:
                jump_times = []
                for index in number:
                    index -= 1
                    jump_times.extend(
                        [
                            item[index]
                            for item in inter_jump_times
                            if index < len(item)
                        ]
                    )
                plt.hist(jump_times, density=density, label="Inter-jump times")
        elif isinstance(number, (int, float, np.integer, np.floating)):
            number -= 1
            jump_times = [
                item[int(number)]
                for item in inter_jump_times
                if int(number) < len(item)
            ]
            plt.hist(jump_times, density=density, label="Inter-jump times")

        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()

        return fig

    def plot_number_of_jumps(
        self,
        density: bool = False,
        title: str | None = None,
        legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots a histogram of the number of jumps per function.

        Parameters
        ----------
        density : bool, optional
            If True, normalize the histogram to a probability density. The
            default is False.
        title : str or None, optional
            Plot title. The default is None.
        legend : bool, optional
            If True, show a legend. The default is False.
        xlabel : str or None, optional
            Label for the x-axis. The default is None.
        ylabel : str or None, optional
            Label for the y-axis. The default is None.
        grid : bool, optional
            If True, show grid lines. The default is False.

        Returns
        -------
        fig : plt.Figure
            Created figure.

        Raises
        ------
        ValueError
            If jump times are not collected.
        """
        if not self.collect_jump_times:
            raise ValueError(
                f"Jump times have not been in collected in \
                             {self}"
            )

        number_of_jumps = [len(jump_times) for jump_times in self.jump_times]

        fig = plt.figure()

        plt.hist(number_of_jumps, density=density, label="Number of jumps")

        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()

        return fig

    def evaluate_norm(self, t: float | ArrayLike) -> np.ndarray | list:
        """
        Evaluates the mean norms at given time points by linear interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Mean norms for each stored order, or one such array per requested
            time.

        Raises
        ------
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.norm_means)

        return value

    def evaluate_dimension(
        self, t: float | ArrayLike
    ) -> np.ndarray | list[np.ndarray]:
        """
        Evaluates the mean dimension at given time points by linear interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.number or list of np.number
            Mean dimension at a scalar time, or one value per requested time.

        Raises
        ------
        AttributeError
            If dimension statistics are not collected.
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.dimension_means)

        return value

    def evaluate_cefs(self, t: float | ArrayLike) -> np.ndarray | list:
        """
        Evaluates the means of the custom evaluation functions by linear
        interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Mean values for the registered custom evaluations, or one such array
            per requested time.

        Raises
        ------
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.cef_means)

        return value

    def evaluate_norm_variance(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variances of the norms by linear interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Norm variances for each stored order, or one such array per requested
            time.

        Raises
        ------
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.norm_variances)

        return value

    def evaluate_dimension_variances(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variance of the dimension by linear interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.number or list of np.number
            Dimension variance at a scalar time, or one value per requested
            time.

        Raises
        ------
        AttributeError
            If dimension statistics are not collected.
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.dimension_variances)

        return value

    def evaluate_cef_variances(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variances of the custom evaluations by linear
        interpolation.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times within the range of self.times.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Variances for the registered custom evaluations, or one such array
            per requested time.

        Raises
        ------
        ValueError
            If a requested time lies outside the range of self.times.
        """
        value = self._evaluate(t, self.cef_variances)

        return value

    def _update_means_and_variances(
        self, values: np.ndarray, old_means: np.ndarray, old_m2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates means and sample variances using Welford's online algorithm.

        Parameters
        ----------
        values : np.ndarray
            Values of the newly added function.
        old_means : np.ndarray
            Means before adding the new values.
        old_m2 : np.ndarray
            Sums of squared deviations before adding the new values.

        Returns
        -------
        new_means : np.ndarray
            Updated means.
        new_variances : np.ndarray
            Updated sample variances. For the first function, an array of zeros
            is returned.
        new_m2 : np.ndarray
            Updated sums of squared deviations.

        Notes
        -----
        self.evaluated_functions must already include the newly added function.
        """
        delta = values - old_means
        new_means = old_means + delta / self.evaluated_functions
        delta2 = values - new_means
        new_m2 = old_m2 + delta * delta2
        if self.evaluated_functions > 1:
            new_variances = new_m2 / (self.evaluated_functions - 1)
        else:
            new_variances = np.zeros_like(new_m2)

        return new_means, new_variances, new_m2

    def _evaluate(
        self, t: float | ArrayLike, values: ArrayLike
    ) -> np.ndarray | list[np.ndarray]:
        """
        Evaluates stored statistics at one or more time points.

        Parameters
        ----------
        t : float, list, or np.ndarray
            Evaluation time or times.
        values : array-like
            Stored statistics, with time along the last axis.

        Returns
        -------
        value : np.ndarray, np.number, or list
            Interpolated value for a scalar time, or a list of values for
            multiple times.

        Raises
        ------
        ValueError
            If a requested time lies outside the range of self.times.
        """
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self._interpolate_value(time, values))
        else:
            value = self._interpolate_value(t, values)

        return value

    def _interpolate_value(self, t: float, values: ArrayLike) -> np.ndarray:
        """
        Linearly interpolates stored statistics at a scalar time.

        Parameters
        ----------
        t : float
            Evaluation time.
        values : array-like
            Stored statistics, with time along the last axis.

        Returns
        -------
        value : np.ndarray or np.number
            Interpolated value. One-dimensional input produces a scalar.

        Raises
        ------
        ValueError
            If t lies outside the range of self.times.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(
                f"t is not between {self.times[0]} and \
                             {self.times[-1]} (first and last value of times)"
            )
        elif t == self.times[-1]:
            return np.asarray(values)[..., -1]

        index_t = bisect.bisect_right(self.times, t)

        interp_times = np.array(self.times[index_t - 1 : index_t + 1])
        interp_values = np.array(values).T[index_t - 1 : index_t + 1]

        value = interp_values[0] + (
            ((t - interp_times[0]) / (interp_times[1] - interp_times[0]))
            * (interp_values[1] - interp_values[0])
        )

        return value

    def __add__(
        self, other: StatsFunctionsOfVariableDimension
    ) -> StatsFunctionsOfVariableDimension:
        """
        Combines the statistics of two collectors.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            Collector whose statistics are added.

        Returns
        -------
        stats : StatsFunctionsOfVariableDimension
            Collector containing the combined statistics.

        Raises
        ------
        ValueError
            If evaluation times, norm orders, numbers of custom functions, or
            collection settings do not match.

        Notes
        -----
        The identity and order of custom evaluation functions are not checked.
        A warning is printed to remind the user to verify them.
        """
        if not isinstance(other, StatsFunctionsOfVariableDimension):
            return NotImplemented
        if not np.all(self.times == other.times):
            raise ValueError("Can only add statistics with same times")
        if not (sorted(self.order) == sorted(other.order)):
            raise ValueError(
                "Can only add statistics with same stored " "norm orders"
            )
        if not (
            len(self.custom_evaluation_functions)
            == len(other.custom_evaluation_functions)
        ):
            raise ValueError(
                "Can only add statistics with same custom "
                "evaluation functions"
            )
        if not self.collect_dimension == other.collect_dimension:
            raise ValueError(
                "Can only add statistics with both collecting"
                "dimensions or none"
            )
        if not (self.collect_jump_times == other.collect_jump_times):
            raise ValueError(
                "Can only add statistics with both collecting"
                "jump times or none"
            )
        if not (
            self.collect_dimension_jump_times
            == other.collect_dimension_jump_times
        ):
            raise ValueError(
                "Can only add statistics with both collecting"
                "dimension jump times or none"
            )
        print(
            "Warning: Please ensure that custom evaluation functions are the"
            " same and in the same order. This cannot be tested."
        )
        # Initialize new stats object
        stats = StatsFunctionsOfVariableDimension(
            (self.times[0], self.times[-1])
        )
        stats.evaluated_functions = (
            self.evaluated_functions + other.evaluated_functions
        )
        stats.times = self.times
        stats.order = self.order
        stats.custom_evaluation_functions = self.custom_evaluation_functions
        stats.collect_dimension = self.collect_dimension
        stats.collect_jump_times = self.collect_jump_times
        stats.collect_dimension_jump_times = self.collect_dimension_jump_times

        # If norm orders are the same, but not in the same order, resort
        if self.order == other.order:
            other_norm_means = other.norm_means
            other_norm_m2 = other.norm_m2
        else:
            other_order = other.order
            other_norm_means = np.empty_like(other.norm_means)
            other_norm_m2 = np.empty_like(other.norm_m2)

            for i, order_value in enumerate(self.order):
                index = other_order.index(order_value)
                other_norm_means[i] = other.norm_means[index]
                other_norm_m2[i] = other.norm_m2[index]

        # Compute combined means and variances
        stats.norm_means, stats.norm_m2 = self._add_means_and_m2(
            self.norm_means,
            other_norm_means,
            self.norm_m2,
            other_norm_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.norm_variances = stats.norm_m2 / (stats.evaluated_functions - 1)

        if stats.collect_dimension:
            stats.dimension_means, stats.dimension_m2 = self._add_means_and_m2(
                self.dimension_means,
                other.dimension_means,
                self.dimension_m2,
                other.dimension_m2,
                self.evaluated_functions,
                other.evaluated_functions,
            )
            stats.dimension_variances = stats.dimension_m2 / (
                stats.evaluated_functions - 1
            )

        stats.cef_means, stats.cef_m2 = self._add_means_and_m2(
            self.cef_means,
            other.cef_means,
            self.cef_m2,
            other.cef_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.cef_variances = stats.cef_m2 / (stats.evaluated_functions - 1)

        # Combine jump times, if collected
        if stats.collect_jump_times:
            stats.jump_times = self.jump_times + other.jump_times
        if stats.collect_dimension_jump_times:
            stats.dimension_jump_times = (
                self.dimension_jump_times + other.dimension_jump_times
            )

        return stats

    def __sub__(
        self, other: StatsFunctionsOfVariableDimension
    ) -> StatsFunctionsOfVariableDimension:
        """
        Removes the statistics of one collector from another.

        The functions represented by other are assumed to be a subset of the
        functions represented by this collector.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            Collector whose statistics are removed.

        Returns
        -------
        stats : StatsFunctionsOfVariableDimension
            Collector containing the resulting statistics.

        Raises
        ------
        ValueError
            If collector settings are incompatible, other contains more
            functions, or collected jump-time lists cannot be removed.

        Notes
        -----
        The function subset and the identity and order of custom evaluation
        functions cannot be verified in general. Warnings are printed to remind
        the user to check them.
        """
        if not isinstance(other, StatsFunctionsOfVariableDimension):
            return NotImplemented
        if not np.all(self.times == other.times):
            raise ValueError(
                "Can only subtract statistics with the same " "times"
            )
        if not (sorted(self.order) == sorted(other.order)):
            raise ValueError(
                "Can only subtract statistics with the same "
                "stored norm orders"
            )
        if not (
            len(self.custom_evaluation_functions)
            == len(other.custom_evaluation_functions)
        ):
            raise ValueError(
                "Can only subtract statistics with the same "
                "custom evaluation functions"
            )
        if not self.collect_dimension == other.collect_dimension:
            raise ValueError(
                "Can only add statistics with both collecting"
                "dimensions or none"
            )
        if not (self.collect_jump_times == other.collect_jump_times):
            raise ValueError(
                "Can only subtract statistics with both "
                "collecting jump times or none"
            )
        if not (
            self.collect_dimension_jump_times
            == other.collect_dimension_jump_times
        ):
            raise ValueError(
                "Can only subtract statistics with both "
                "collecting dimension jump times or none"
            )
        if not (self.evaluated_functions >= other.evaluated_functions):
            raise ValueError(
                "Can only subtract statistics when functions "
                "added to the subtrahend are a subset of those "
                "added to the minuend. In particular, the number "
                "of evaluated functions must be less or equal "
                "for the subtrahend"
            )
        print(
            "Warning: Please ensure that the functions added to the "
            "subtrahend are a subset of those added to the minuend. This "
            "cannot be tested."
        )
        print(
            "Warning: Please ensure that custom evaluation functions are the"
            " same and in the same order. This cannot be tested."
        )
        # If self and other are of the same length, return empty stats
        if self.evaluated_functions == other.evaluated_functions:
            stats = StatsFunctionsOfVariableDimension(
                t_span=(self.times[0], self.times[-1]),
                t_eval=self.times.copy(),
                order=self.order.copy(),
                collect_dimension=self.collect_dimension,
                collect_jump_times=self.collect_jump_times,
                collect_dimension_jump_times=self.collect_dimension_jump_times,
            )

            for func in self.custom_evaluation_functions:
                stats.add_custom_evaluation_function(func)

            return stats

        # Initialize new stats object
        stats = StatsFunctionsOfVariableDimension(
            (self.times[0], self.times[-1])
        )
        stats.evaluated_functions = (
            self.evaluated_functions - other.evaluated_functions
        )
        stats.times = self.times
        stats.order = self.order
        stats.custom_evaluation_functions = self.custom_evaluation_functions
        stats.collect_dimension = self.collect_dimension
        stats.collect_jump_times = self.collect_jump_times
        stats.collect_dimension_jump_times = self.collect_dimension_jump_times

        # If norm orders are the same, but not in the same order, resort
        if self.order == other.order:
            other_norm_means = other.norm_means
            other_norm_m2 = other.norm_m2
        else:
            other_order = other.order
            other_norm_means = np.empty_like(other.norm_means)
            other_norm_m2 = np.empty_like(other.norm_m2)

            for i, order_value in enumerate(self.order):
                index = other_order.index(order_value)
                other_norm_means[i] = other.norm_means[index]
                other_norm_m2[i] = other.norm_m2[index]

        # Compute subtracted means and variances
        stats.norm_means, stats.norm_m2 = self._subtract_means_and_m2(
            self.norm_means,
            other_norm_means,
            self.norm_m2,
            other_norm_m2,
            stats.evaluated_functions,
            other.evaluated_functions,
        )
        if stats.evaluated_functions > 1:
            stats.norm_variances = stats.norm_m2 / (
                stats.evaluated_functions - 1
            )
        else:
            stats.norm_variances = np.zeros_like(stats.norm_m2)

        if stats.collect_dimension:
            stats.dimension_means, stats.dimension_m2 = (
                self._subtract_means_and_m2(
                    self.dimension_means,
                    other.dimension_means,
                    self.dimension_m2,
                    other.dimension_m2,
                    stats.evaluated_functions,
                    other.evaluated_functions,
                )
            )
            if stats.evaluated_functions > 1:
                stats.dimension_variances = stats.dimension_m2 / (
                    stats.evaluated_functions - 1
                )
            else:
                stats.dimension_variances = np.zeros_like(stats.dimension_m2)

        stats.cef_means, stats.cef_m2 = self._subtract_means_and_m2(
            self.cef_means,
            other.cef_means,
            self.cef_m2,
            other.cef_m2,
            stats.evaluated_functions,
            other.evaluated_functions,
        )
        if stats.evaluated_functions > 1:
            stats.cef_variances = stats.cef_m2 / (
                stats.evaluated_functions - 1
            )
        else:
            stats.cef_variances = np.zeros_like(stats.cef_m2)

        # Reduce jump times, if collected
        if stats.collect_jump_times:
            if not _is_sublist(other.jump_times, self.jump_times):
                raise ValueError(
                    "Jump times of subtrahend are not part of "
                    "the jump times of the minuend. Please "
                    "ensure that the functions added to the "
                    "subtrahend are a subset of those added to "
                    "the minuend."
                )
            stats.jump_times = _list_difference(
                other.jump_times, self.jump_times
            )
        if stats.collect_dimension_jump_times:
            if not _is_sublist(
                other.dimension_jump_times, self.dimension_jump_times
            ):
                raise ValueError(
                    "Dimension jump times of subtrahend are not "
                    "part of the jump times of the minuend. "
                    "Please ensure that the functions added to "
                    "the subtrahend are a subset of those added "
                    "to the minuend."
                )
            stats.dimension_jump_times = _list_difference(
                other.dimension_jump_times, self.dimension_jump_times
            )

        return stats

    def __eq__(self, other: StatsFunctionsOfVariableDimension) -> bool:
        """
        Compares this collector with another collector for equality.

        Stored attributes are compared except for custom_evaluation_functions.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            Collector to compare with.

        Returns
        -------
        bool
            True if the checked attributes are equal, False otherwise.
        """
        if not isinstance(other, StatsFunctionsOfVariableDimension):
            return NotImplemented

        self_attributes = vars(self)
        other_attributes = vars(other)

        return all(
            np.all(self_attributes[key] == other_attributes[key])
            for key in self_attributes
            if key != "custom_evaluation_functions"
        )

    def __ne__(self, other: StatsFunctionsOfVariableDimension) -> bool:
        """
        Compares this collector with another collector for inequality.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            Collector to compare with.

        Returns
        -------
        bool
            Negation of the equality comparison.
        """
        return not (self == other)

    @staticmethod
    def _add_means_and_m2(
        means_x: np.ndarray,
        means_y: np.ndarray,
        m2_x: np.ndarray,
        m2_y: np.ndarray,
        n: int,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combines means and sums of squared deviations from two samples.

        Parameters
        ----------
        means_x : np.ndarray
            Means of the first sample.
        means_y : np.ndarray
            Means of the second sample.
        m2_x : np.ndarray
            Sums of squared deviations of the first sample.
        m2_y : np.ndarray
            Sums of squared deviations of the second sample.
        n : int
            Number of observations in the first sample.
        m : int
            Number of observations in the second sample.

        Returns
        -------
        means : np.ndarray
            Combined means.
        m2 : np.ndarray
            Combined sums of squared deviations.
        """
        means = (n * means_x + m * means_y) / (n + m)
        m2 = m2_x + m2_y + n * m * (means_x - means_y) ** 2 / (n + m)

        return means, m2

    @staticmethod
    def _subtract_means_and_m2(
        means_xy: np.ndarray,
        means_y: np.ndarray,
        m2_xy: np.ndarray,
        m2_y: np.ndarray,
        n: int,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes means and sums of squared deviations after removing a sample.

        Parameters
        ----------
        means_xy : np.ndarray
            Means of the combined sample.
        means_y : np.ndarray
            Means of the sample to remove.
        m2_xy : np.ndarray
            Sums of squared deviations of the combined sample.
        m2_y : np.ndarray
            Sums of squared deviations of the sample to remove.
        n : int
            Number of observations in the remaining sample.
        m : int
            Number of observations in the sample to remove.

        Returns
        -------
        means : np.ndarray
            Means of the remaining sample.
        m2 : np.ndarray
            Sums of squared deviations of the remaining sample.
        """
        means = ((n + m) * means_xy - m * means_y) / n
        m2 = m2_xy - m2_y - n * m * (means - means_y) ** 2 / (n + m)

        return means, m2
