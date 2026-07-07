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
    A class to compute and store statistics of functions of variable
    dimension.

    Attributes
    ----------
    evaluated_functions : int
        Count of the functions evaluated.
    times : list
        A list of time points for evaluations.
    order : list
        A list of orders for norms to be computed.
    norm_means : np.ndarray
        Means of the norms for each order over time. The array has the
        shape (len(order), len(times)).
    norm_variances : np.ndarray
        Sample variances of the norms for each order over time. The array
        has the shape (len(order), len(times)).
    norm_m2 : np.ndarray
        Sum of squared differences of the norms for each order over time.
        The array has the shape (len(order), len(times)).
    dimension_means : np.ndarray
        Means of the dimension over time. The array has the length
        len(times).
    dimension_variances : np.ndarray
        Sample variances of the dimension over time. The array has the
        length len(times).
    dimension_m2 : np.ndarray
        Sum of squared differences of the dimension for each order over
        time. The array has the length len(times).
    custom_evaluation_functions : list of callables
        List of user-provided functions to evaluate.
    cef_means : np.ndarray
        Means of the custom evaluation functions over time. The array has
        the shape (len(custom_evaluation_functions), len(times)).
    cef_variances : np.ndarray
        Sample variances of the custom evaluation functions over time. The
        array has the shape (len(custom_evaluation_functions), len(times)).
    cef_m2 : np.ndarray
        Sum of squared differences of the custom evaluation functions over
        time. The array has the shape
        (len(custom_evaluation_functions), len(times)).
    collect_jump_times : bool
        Indicates whether jump times are stored.
    jump_times : list of lists
        List of all jump times of evaluated functions. Exists only, when
        collect_jump_times is True.
    collect_dimension_jump_times : bool
        Indicates whether dimension jump times are stored.
    dimension_jump_times : list of lists
        List of all dimension jump times of evaluated functions. Exists
        only, when collect_dimension_jump_times is True.
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
            The start and end times ``(t0, tf)`` for evaluations.
        stepsize : float, optional
            The step size for time evaluations. Ignored if ``t_eval`` is
            provided. The default is ``1``.
        t_eval : array-like or None, optional
            Specific times at which to evaluate the functions. If ``None``,
            times are generated from ``t_span`` using ``stepsize``. The
            default is ``None``.
        order : int, float, or array-like, optional
            The order(s) of the norm(s) to compute as accepted by
            ``numpy.linalg.norm`` (e.g. ``1``, ``2``, ``np.inf``). The
            default is ``2`` (Euclidean norm).
        collect_dimension : bool, optional
            If ``True``, dimension statistics are collected during evaluation.
            The default is ``True``.
        collect_jump_times : bool, optional
            If ``True``, jump times of each evaluated function are stored in
            ``self.jump_times``. The default is ``False``.
        collect_dimension_jump_times : bool, optional
            If ``True``, dimension jump times of each evaluated function are
            stored in ``self.dimension_jump_times``. The default is ``False``.

        Raises
        ------
        TypeError
            If ``order`` is neither a number nor array-like.
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
        Adds a function to the statistics collector for evaluation.

        Parameters
        ----------
        func : FunctionOfVariableDimension
            The function to add and evaluate.

        Raises
        ------
        TypeError
            If func is not a function of variable dimension.
        ValueError
            If the function's time range does not match the collector's.
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

    def _update_means_and_variances(
        self, values: np.ndarray, old_means: np.ndarray, old_m2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates the means and variances based on new values.

        Updates the means and variances based on new values using Welford's
        online algorithm.

        Parameters
        ----------
        values : np.ndarray
            The new values for which to update statistics.
        old_means : np.ndarray
            The current means before the update.
        old_m2 : np.ndarray
            The current M2 values before the update.

        Returns
        -------
        new_means : np.ndarray
            Updated means with the new values.
        new_variances : np.ndarray
            Updated variances with the new values.
        new_m2 : np.ndarray
            Updated M2 values with the new values.
        """
        delta = values - old_means
        new_means = old_means + delta / self.evaluated_functions
        delta2 = values - new_means
        new_m2 = old_m2 + delta * delta2
        if self.evaluated_functions > 1:
            new_variances = new_m2 / (self.evaluated_functions - 1)
        else:
            new_variances = 0

        return new_means, new_variances, new_m2

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
        Plots the mean norms of the collected functions.

        For each requested norm order, the mean is plotted as a line and the
        uncertainty band of ±``sigma`` standard deviations is shaded.

        Parameters
        ----------
        order : int, float, or list of int or float, optional
            The norm order(s) to plot as accepted by ``numpy.linalg.norm``
            (e.g. ``1``, ``2``, ``np.inf``). Must be a subset of the orders
            stored in ``self.order``. The default is ``2``.
        sigma : float, optional
            Scaling factor for the variance shading (the band spans
            ±``sigma`` · standard deviation). The default is ``2``.
        labels : list of str or None, optional
            Custom legend labels for each norm order. If ``None``, labels are
            generated automatically (e.g. ``'2 norm'``). The default is
            ``None``.
        title : str, optional
            The title of the plot. The default is ``'Plot of mean norm'``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``True``.
        xlabel : str, optional
            Label for the x-axis. The default is ``'Time'``.
        ylabel : str, optional
            Label for the y-axis. The default is ``'Norm'``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.

        Raises
        ------
        ValueError
            If a requested ``order`` has not been computed, i.e. is not
            contained in ``self.order``.
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
        Plots the mean dimension of the collected functions over time.

        The mean dimension is plotted as a line and the uncertainty band of
        ±``sigma`` standard deviations is shaded.

        Parameters
        ----------
        sigma : float, optional
            Scaling factor for the variance shading (the band spans
            ±``sigma`` · standard deviation). The default is ``2``.
        labels : list of str or None, optional
            Custom legend labels. If ``None``, the label ``'Dimension'`` is
            used. The default is ``None``.
        title : str, optional
            The title of the plot. The default is
            ``'Plot of mean dimension'``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``False``.
        xlabel : str, optional
            Label for the x-axis. The default is ``'Time'``.
        ylabel : str, optional
            Label for the y-axis. The default is ``'Dimension'``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.

        Raises
        ------
        ValueError
            If dimension statistics have not been collected
            (``collect_dimension=False``).
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
        Plots the means of the custom evaluation functions (CEFs) over time.

        For each CEF, the mean is plotted as a line and the uncertainty band
        of ±``sigma`` standard deviations is shaded.

        Parameters
        ----------
        sigma : float, optional
            Scaling factor for the variance shading (the band spans
            ±``sigma`` · standard deviation). The default is ``2``.
        labels : list of str or None, optional
            Custom legend labels for each CEF. If ``None``, labels are
            generated automatically (e.g. ``'Cef 1'``). The default is
            ``None``.
        title : str, optional
            The title of the plot. The default is ``'Plot of cefs'``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``False``.
        xlabel : str, optional
            Label for the x-axis. The default is ``'Time'``.
        ylabel : str, optional
            Label for the y-axis. The default is ``'Value'``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.
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
        Plots a histogram of the jump times of the collected functions.

        Parameters
        ----------
        number : list of int or None, optional
            The jump indices to include (1-based). For example, ``[1, 2]``
            plots the first and second jump times. If ``None``, all jump
            indices up to the maximum observed number of jumps are included.
            The default is ``None``.
        density : bool, optional
            If ``True``, the histogram is normalized to form a probability
            density. The default is ``False``.
        separate : bool, optional
            If ``True``, each jump index is plotted as a separate bar group
            within the same figure. The default is ``False``.
        separate_plots : bool, optional
            If ``True``, each jump index is plotted in a separate figure.
            The default is ``False``.
        title : str or None, optional
            The title of the plot. If ``None``, no title is set. The default
            is ``None``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``False``.
        xlabel : str or None, optional
            Label for the x-axis. If ``None``, no label is set. The default
            is ``None``.
        ylabel : str or None, optional
            Label for the y-axis. If ``None``, no label is set. The default
            is ``None``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.

        Raises
        ------
        ValueError
            If jump times have not been collected
            (``collect_jump_times=False``).
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
                for index in number:
                    index -= 1
                    jump_times = [
                        item[index]
                        for item in self.jump_times
                        if index < len(item)
                    ]
                    plt.hist(
                        jump_times, density=density, label=f"Jump {index+1}"
                    )
                    if separate_plots:
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
                plt.hist(jump_times, density=density)
        elif isinstance(number, (int, float, np.integer, np.floating)):
            number -= 1
            jump_times = [
                item[int(number)]
                for item in self.jump_times
                if int(number) < len(item)
            ]
            plt.hist(jump_times, density=density)

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
        Plots a histogram of the inter-jump times of the collected functions.

        The inter-jump time for the first jump equals the jump time itself;
        for subsequent jumps it is the difference between consecutive jump
        times.

        Parameters
        ----------
        number : list of int or None, optional
            The jump indices to include (1-based). For example, ``[1, 2]``
            plots the inter-jump times preceding the first and second jump.
            If ``None``, all indices up to the maximum observed number of
            jumps are included. The default is ``None``.
        density : bool, optional
            If ``True``, the histogram is normalized to form a probability
            density. The default is ``False``.
        separate : bool, optional
            If ``True``, each jump index is plotted as a separate bar group
            within the same figure. The default is ``False``.
        separate_plots : bool, optional
            If ``True``, each jump index is plotted in a separate figure.
            The default is ``False``.
        title : str or None, optional
            The title of the plot. If ``None``, no title is set. The default
            is ``None``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``False``.
        xlabel : str or None, optional
            Label for the x-axis. If ``None``, no label is set. The default
            is ``None``.
        ylabel : str or None, optional
            Label for the y-axis. If ``None``, no label is set. The default
            is ``None``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.

        Raises
        ------
        ValueError
            If jump times have not been collected
            (``collect_jump_times=False``).
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
                for index in number:
                    index -= 1
                    jump_times = [
                        item[index]
                        for item in inter_jump_times
                        if index < len(item)
                    ]
                    plt.hist(
                        jump_times, density=density, label=f"Jump {index+1}"
                    )
                    if separate_plots:
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
                plt.hist(jump_times, density=density)
        elif isinstance(number, (int, float, np.integer, np.floating)):
            number -= 1
            jump_times = [
                item[int(number)]
                for item in inter_jump_times
                if int(number) < len(item)
            ]
            plt.hist(jump_times, density=density)

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
        Plots a histogram of the number of jumps per collected function.

        Parameters
        ----------
        density : bool, optional
            If ``True``, the histogram is normalized to form a probability
            density. The default is ``False``.
        title : str or None, optional
            The title of the plot. If ``None``, no title is set. The default
            is ``None``.
        legend : bool, optional
            If ``True``, a legend is shown. The default is ``False``.
        xlabel : str or None, optional
            Label for the x-axis. If ``None``, no label is set. The default
            is ``None``.
        ylabel : str or None, optional
            Label for the y-axis. If ``None``, no label is set. The default
            is ``None``.
        grid : bool, optional
            If ``True``, grid lines are shown. The default is ``False``.

        Returns
        -------
        fig : plt.Figure
            The created figure object.

        Raises
        ------
        ValueError
            If jump times have not been collected
            (``collect_jump_times=False``).
        """
        if not self.collect_jump_times:
            raise ValueError(
                f"Jump times have not been in collected in \
                             {self}"
            )

        number_of_jumps = [len(jump_times) for jump_times in self.jump_times]

        fig = plt.figure()

        plt.hist(number_of_jumps, density=density)

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

    def add_custom_evaluation_function(
        self, func: Callable[[list[np.ndarray]], np.ndarray]
    ) -> None:
        """
        Adds a custom evaluation function to be evaluated later.

        Parameters
        ----------
        func : callable
            A function that receives a list of np.ndarray (the function
            values evaluated at all times in ``self.times``) and returns
            a scalar or np.ndarray per time point.

        Raises
        ------
        ValueError
            If functions have already been added.
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

    def evaluate_norm(self, t: float | ArrayLike) -> np.ndarray | list:
        """
        Evaluates the mean norm at given time(s) by linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the mean norm. Must lie
            within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated norm means. A single ``np.ndarray`` of shape
            ``(len(self.order),)`` if ``t`` is a scalar; a list of such
            arrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If ``t`` is outside the range of ``self.times``.
        """
        value = self._evaluate(t, self.norm_means)

        return value

    def evaluate_dimension(
        self, t: float | ArrayLike
    ) -> np.ndarray | list[np.ndarray]:
        """
        Evaluates the mean dimension at given time(s) by linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the mean dimension. Must
            lie within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated dimension mean(s). A np.ndarray if ``t`` is a scalar;
            a list of np.ndarrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If dimension statistics have not been collected
            (``collect_dimension=False``) or if ``t`` is outside the range
            of ``self.times``.
        """
        value = self._evaluate(t, self.dimension_means)

        return value

    def evaluate_cefs(self, t: float | ArrayLike) -> np.ndarray | list:
        """
        Evaluates the means of the custom evaluation functions at given
        time(s) by linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the CEF means. Must lie
            within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated CEF means. A single ``np.ndarray`` of shape
            ``(len(custom_evaluation_functions),)`` if ``t`` is a scalar;
            a list of such arrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If ``t`` is outside the range of ``self.times``.
        """
        value = self._evaluate(t, self.cef_means)

        return value

    def evaluate_norm_variance(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variances of the norms at given time(s) by
        linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the norm variances. Must
            lie within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated norm variances. A single ``np.ndarray`` of shape
            ``(len(self.order),)`` if ``t`` is a scalar; a list of such
            arrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If ``t`` is outside the range of ``self.times``.
        """
        value = self._evaluate(t, self.norm_variances)

        return value

    def evaluate_dimension_variances(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variances of the dimensions at given time(s) by
        linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the dimension variances.
            Must lie within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : float or list of float
            Evaluated dimension variance(s). A np.ndarray if ``t`` is a scalar;
            a list of np.ndarrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If dimension statistics have not been collected
            (``collect_dimension=False``) or if ``t`` is outside the range
            of ``self.times``.
        """
        value = self._evaluate(t, self.dimension_variances)

        return value

    def evaluate_cef_variances(
        self, t: float | ArrayLike
    ) -> np.ndarray | list:
        """
        Evaluates the sample variances of the custom evaluation functions at
        given time(s) by linear interpolation.

        Parameters
        ----------
        t : float or array-like
            The time point(s) at which to evaluate the CEF variances. Must
            lie within ``[self.times[0], self.times[-1]]``.

        Returns
        -------
        value : np.ndarray or list of np.ndarray
            Evaluated CEF variances. A single ``np.ndarray`` of shape
            ``(len(custom_evaluation_functions),)`` if ``t`` is a scalar;
            a list of such arrays if ``t`` is array-like.

        Raises
        ------
        ValueError
            If ``t`` is outside the range of ``self.times``.
        """
        value = self._evaluate(t, self.cef_variances)

        return value

    def _evaluate(
        self, t: float | ArrayLike, values: ArrayLike
    ) -> np.ndarray | list[np.ndarray]:
        """
        Evaluates the specified values at given time points by interpolation.

        Parameters
        ----------
        t : float, list, or array-like
            The time point(s) at which to evaluate the values.
        values : array-like
            The values for interpolation.

        Returns
        -------
        value : np.ndarray or list
            Evaluated value(s) at the specified time point(s).
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
        Linearly interpolates the values at a given time(s).

        Parameters
        ----------
        t : float
            The time at which to interpolate the value.
        values : array-like
            The values to interpolate.

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
                f"t is not between {self.times[0]} and \
                             {self.times[-1]} (first and last value of times)"
            )
        elif t == self.times[-1]:
            return np.array(values[:, -1])

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
        Adds two instances of StatsFunctionsOfVariableDimension together.

        This method combines the statistics of two
        `StatsFunctionsOfVariableDimension` objects, ensuring that the time
        points, norm orders, and other attributes are compatible. It
        computes the combined means and variances for norms and dimensions,
        as well as for custom evaluation functions.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            The statistics object to add to this instance.

        Returns
        -------
        stats : StatsFunctionsOfVariableDimension
            A new instance containing the combined statistics.

        Raises
        ------
        ValueError
            If the time points, norm orders, custom evaluation functions,
            or jump time collection settings do not match between the two
            instances being added.

        Notes
        -----
        Prints a warning to stdout that equality of custom evaluation functions
        cannot be verified programmatically.
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
                index = np.where(other_order == order_value)[0][0]
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
        Subtracts another instance of StatsFunctionsOfVariableDimension.

        This method computes the difference between two
        `StatsFunctionsOfVariableDimension` objects in the sense of removing
        the functions of the subtracted `StatsFunctionsOfVariableDimension`
        object, ensuring that the time points, norm orders, and other
        attributes are compatible. It calculates the resulting means and
        variances for norms and dimensions, as well as for custom evaluation
        functions.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            The statistics object to subtract from this instance.

        Returns
        -------
        stats : StatsFunctionsOfVariableDimension
            A new instance containing the resulting statistics after
            subtraction.

        Raises
        ------
        ValueError
            If the time points, norm orders, custom evaluation functions,
            or jump time collection settings do not match between the two
            instances being subtracted.

        Notes
        -----
        Prints a warning to stdout that functions added to the subtrahend
        should be a subset of functions added to the minuend which cannot be
        verified programmatically.
        Prints a warning to stdout that equality of custom evaluation functions
        cannot be verified programmatically.
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
                index = np.where(other_order == order_value)[0][0]
                other_norm_means[i] = other.norm_means[index]
                other_norm_m2[i] = other.norm_m2[index]

        # Compute subtracted means and variances
        stats.norm_means, stats.norm_m2 = self._subtract_means_and_m2(
            self.norm_means,
            other_norm_means,
            self.norm_m2,
            other_norm_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.norm_variances = stats.norm_m2 / (stats.evaluated_functions - 1)

        if stats.collect_dimension:
            stats.dimension_means, stats.dimension_m2 = (
                self._subtract_means_and_m2(
                    self.dimension_means,
                    other.dimension_means,
                    self.dimension_m2,
                    other.dimension_m2,
                    self.evaluated_functions,
                    other.evaluated_functions,
                )
            )
            stats.dimension_variances = stats.dimension_m2 / (
                stats.evaluated_functions - 1
            )

        stats.cef_means, stats.cef_m2 = self._subtract_means_and_m2(
            self.cef_means,
            other.cef_means,
            self.cef_m2,
            other.cef_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.cef_variances = stats.cef_m2 / (stats.evaluated_functions - 1)

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
        Compare this statistics object with another for equality.

        Compares the attributes of this instance with those of another
        `StatsFunctionsOfVariableDimension` instance, excluding the
        `custom_evaluation_functions`.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            The statistics object to compare with.

        Returns
        -------
        bool
            True if the statistics objects are equal, False otherwise.
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
        Compare this statistics object with another for inequality.

        Compares the attributes of this instance with those of another
        `StatsFunctionsOfVariableDimension` instance, excluding the
        `custom_evaluation_functions`.

        Parameters
        ----------
        other : StatsFunctionsOfVariableDimension
            The statistics object to compare with.

        Returns
        -------
        bool
            True if the statistics objects are not equal, False otherwise.
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
        Add means and second moments of variable with n and m realizations.

        Parameters
        ----------
        means_x : np.ndarray
            The means of the first variable.
        means_y : np.ndarray
            The means of the second variable.
        m2_x : np.ndarray
            The second moments of the first variable.
        m2_y : np.ndarray
            The second moments of the second variable.
        n : int
            The number of realizations of the first variable.
        m : int
            The number of realizations of the second variable.

        Returns
        -------
        means : np.ndarray
            The added means.
        m2 : np.ndarray
            The added second moments.
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
        Subtract means and 2nd moments of variable with n and m realizations.

        Parameters
        ----------
        means_xy : np.ndarray
            The means of the minuend variable.
        means_y : np.ndarray
            The means of the subtrahend variable.
        m2_xy : np.ndarray
            The second moments of the minuend variable.
        m2_y : np.ndarray
            The second moments of the subtrahend variable.
        n : int
            The number of realizations of the minuend variable.
        m : int
            The number of realizations of the subtrahend variable.

        Returns
        -------
        means : np.ndarray
            The subtracted means.
        m2 : np.ndarray
            The subtracted second moments.
        """
        means = ((n + m) * means_xy - m * means_y) / n
        m2 = m2_xy - m2_y - n * m * (means - means_y) ** 2 / (n + m)

        return means, m2

    # @staticmethod
    # def _partial_t_eval(t_eval: ArrayLike, t0: float, tf: float) -> np.ndarray:
    #     """
    #     Reduces t_eval to values between t0 and tf.

    #     Prepares the evaluation times by ensuring they are within the specified
    #     range.

    #     Parameters
    #     ----------
    #         t_eval : list or array-like
    #             The specified evaluation times.
    #         t0 : float
    #             The start of the time range.
    #         tf : float
    #             The end of the time range.

    #     Returns
    #     -------
    #         t_eval_partial : np.ndarray
    #             The filtered evaluation times within the range.
    #     """
    #     if t_eval is None:
    #         return None

    #     t_eval = np.array(t_eval)
    #     t_eval_partial = t_eval[(t_eval >= t0) & (t_eval <= tf)].tolist()

    #     if t0 not in t_eval_partial:
    #         t_eval_partial.insert(0, t0)

    #     if tf not in t_eval_partial:
    #         t_eval_partial.append(tf)

    #     t_eval_partial = np.array(t_eval_partial)

    #     return t_eval_partial
