import bisect
from collections import Counter
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.linalg as la

from .function_of_variable_dimension import FunctionOfVariableDimension


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
        t_span : tuple
            The start and end times for evaluations.
        stepsize : float, optional
            The step size for time evaluations.
        t_eval : list or array-like, optional
            Specific times at which to evaluate the functions.
        order : int, list or array-like, optional
            The order(s) of the norm(s) to compute.
        collect_jump_times : bool, optional
            If True, collect jump times during evaluation.
        collect_dimension_jump_times : bool, optional
            If True, collect dimension jump times during evaluation.
        """
        # Create self.times according to stepsize or t_eval
        t0, tf = t_span
        decimal_places = (
            len(str(stepsize).split(".")[1]) if "." in str(stepsize) else 0
        )
        if t_eval is not None:
            self.times = self._partial_t_eval(t_eval, t0, tf)
        else:
            self.times = np.round(np.arange(t0, tf, stepsize), decimal_places)
        if not np.isclose(self.times[-1], tf):
            self.times = np.append(self.times, tf)

        # Initialize all other attributes
        self.evaluated_functions = 0

        if isinstance(order, (list, np.ndarray, tuple)):
            self.order = list(order)
        elif isinstance(
            order,
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
        order: int | ArrayLike = 2,
        sigma: float = 2,
        labels: list[str] = None,
        title: str = "Plot of mean norm",
        legend: bool = True,
        xlabel: str = "Time",
        ylabel: str = "Norm",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the mean norms of the collected functions.

        Parameters
        ----------
        order : int or array-like, optional
            The order(s) of the norm to plot.
        sigma : float, optional
            The factor for the variance shading.
        labels : list, optional
            Custom labels for the plot lines.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.

        Raises
        ------
        ValueError
            If order to plot is not in order, i.e. has not been stored and
            computed.
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
                    "{order} is not in self.order and hence was \
                                 not computed"
                )
            index = self.order.index(order)
            (line,) = plt.plot(self.times, self.norm_means[index])
            plt.fill_between(
                self.times,
                self.norm_means[index] - sigma * self.norm_variances[index],
                self.norm_means[index] + sigma * self.norm_variances[index],
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
        labels: list[str] = None,
        title: str = "Plot of mean dimension",
        legend: bool = False,
        xlabel: str = "Time",
        ylabel: str = "Dimension",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the mean dimensions of the collected functions.

        Parameters
        ----------
        sigma : float, optional
            The factor for the variance shading.
        labels : list, optional
            Custom labels for the plot lines.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.
        """
        if not self.collect_dimension:
            raise ValueError("Cannot plot dimension if it was not collected.")
        fig = plt.figure()

        lines = []

        labels = ["Dimension"]

        (line,) = plt.plot(self.times, self.dimension_means)
        plt.fill_between(
            self.times,
            self.dimension_means - sigma * self.dimension_variances,
            self.dimension_means + sigma * self.dimension_variances,
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
        labels: list[str] = None,
        title: str = "Plot of cefs",
        legend: bool = False,
        xlabel: str = "Time",
        ylabel: str = "Value",
        grid: bool = False,
    ) -> plt.Figure:
        """
        Plots the means of the custom evaluation functions (CEFs).

        Parameters
        ----------
        sigma : float, optional
            The factor for the variance shading.
        labels : list, optional
            Custom labels for the plot lines.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.
        """
        fig = plt.figure()

        lines = []

        labels = labels or [
            "Cef {i+1}" for i in range(len(self.custom_evaluation_functions))
        ]

        for index in range(len(self.custom_evaluation_functions)):
            (line,) = plt.plot(self.times, self.cef_means[index])
            plt.fill_between(
                self.times,
                self.cef_means[index] - sigma * self.cef_variances[index],
                self.cef_means[index] + sigma * self.cef_variances[index],
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
        Plots a histogram of the jump times.

        Parameters
        ----------
        number : list, optional
            Specific jump numbers to plot.
        density : bool, optional
            If True, normalize the histogram.
        separate : bool, optional
            If True, plot jumps separately.
        separate_plots : bool, optional
            If True, create separate plots for each jump time.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.

        Raises
        ------
        ValueError
            If jump times have not been collected.
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
        elif isinstance(
            number,
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
        Plots a histogram of the inter-jump times.

        Parameters
        ----------
        number : list, optional
            Specific jump numbers to plot.
        density : bool, optional
            If True, normalize the histogram.
        separate : bool, optional
            If True, create separate plots for each jump.
        separate_plots : bool, optional
            Create separate plots for each inter-jump time.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.

        Raises
        ------
        ValueError
            If jump times have not been collected.
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
        elif isinstance(
            number,
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
        Plots a histogram of the number of jumps.

        Parameters
        ----------
        density : bool, optional
            If True, normalize the histogram.
        title : str, optional
            The title of the plot.
        legend : bool, optional
            If True, show the legend.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        grid : bool, optional
            If True, show grid lines on the plot.

        Returns
        -------
        fig : FigureObject
            The figure object created.

        Raises
        ------
        ValueError
            If jump times have not been collected.
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

    def add_custom_evaluation_function(self, func: callable) -> None:
        """
        Adds a custom evaluation function to be evaluated later.

        Parameters
        ----------
        func : callable
            The custom evaluation function to add.

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

    def evaluate_norm(self, t: float | ArrayLike):
        """
        Evaluates the mean norm at given time(s).

        Parameters
        ----------
            t : float, or array-like
                The time point(s) at which to evaluate the mean norm.

        Returns
        -------
            value : np.ndarray, or list
                Evaluated norm means.
        """
        value = self._evaluate(t, self.norm_means)

        return value

    def evaluate_dimension(self, t: float | ArrayLike):
        """
        Evaluates the mean dimension at a given time.

        Parameters
        ----------
        t : float, or array-like
            The time point(s) at which to evaluate the mean dimension.

        Returns
        -------
        value : float, or list
            Evaluated dimension means.
        """
        value = self._evaluate(t, self.dimension_means)

        return value

    def evaluate_cefs(self, t: float | ArrayLike):
        """
        Evaluates the mean of the custom evaluation functions at given time(s).

        Parameters
        ----------
        t : float, list, or array-like
            The time point(s) at which to evaluate the mean of the CEFs.

        Returns
        -------
        value : np.ndarray, or list
            Evaluated custom evaluation function means.
        """
        value = self._evaluate(t, self.cef_means)

        return value

    def evaluate_norm_variance(self, t: float | ArrayLike):
        """
        Evaluates the variances of the norms at given time(s).

        Parameters
        ----------
        t : float, list, or array-like
            The time point(s) at which to evaluate the variances.

        Returns
        -------
        value : np.ndarray, or list
            Evaluated norm variances.
        """
        value = self._evaluate(t, self.norm_variances)

        return value

    def evaluate_dimension_variances(self, t: float | ArrayLike):
        """
        Evaluates the variances of the dimensions at given time(s).

        Parameters
        ----------
        t : float, list, or array-like
            The time point(s) at which to evaluate the variances.

        Returns
        -------
        value : float, or list
            Evaluated dimension variances.
        """
        value = self._evaluate(t, self.dimension_variances)

        return value

    def evaluate_cef_variances(self, t: float | ArrayLike):
        """
        Evaluates the variances of the cefs at given time(s).

        Evaluates the variances of the custom evaluation functions at given
        time(s).

        Parameters
        ----------
        t : float, list, or array-like
            The time point(s) at which to evaluate the variances of the CEFs.

        Returns
        -------
        value : np.ndarray, or list
            Evaluated variances of the CEFs.
        """
        value = self._evaluate(t, self.cef_variances)

        return value

    def _evaluate(self, t: float | ArrayLike, values: ArrayLike):
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
            value : float, np.ndarray or list
                Evaluated value(s) at the specified time point(s).
        """
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self._interpolate_value(time, values))
        else:
            value = self._interpolate_value(t, values)

        return value

    def _interpolate_value(self, t: float, values: ArrayLike):
        """
        Interpolates the values at a given time(s).

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
            return np.array(values[:, -1])

        index_t = bisect.bisect_right(self.times, t)

        interp_times = np.array(self.times[index_t - 1 : index_t + 1])
        interp_values = np.array(values).T[index_t - 1 : index_t + 1]

        value = interp_values[0] + (
            ((t - interp_times[0]) / (interp_times[1] - interp_times[0]))
            * (interp_values[1] - interp_values[0])
        )

        return value

    def __add__(self, other: StatsFunctionsOfVariableDimension):
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
        stats.norm_variances = stats.norm_m2 / stats.evaluated_functions

        if stats.collect_dimension:
            stats.dimension_means, stats.dimension_m2 = self._add_means_and_m2(
                self.dimension_means,
                other.dimension_means,
                self.dimension_m2,
                other.dimension_m2,
                self.evaluated_functions,
                other.evaluated_functions,
            )
            stats.dimension_variances = (
                stats.dimension_m2 / stats.evaluated_functions
            )

        stats.cef_means, stats.cef_m2 = self._add_means_and_m2(
            self.cef_means,
            other.cef_means,
            self.cef_m2,
            other.cef_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.cef_variances = stats.cef_m2 / stats.evaluated_functions

        # Combine jump times, if collected
        if stats.collect_jump_times:
            stats.jump_times = self.jump_times + other.jump_times
        if stats.collect_dimension_jump_times:
            stats.dimension_jump_times = (
                self.dimension_jump_times + other.dimension_jump_times
            )

        return stats

    def __sub__(self, other):
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
        stats.norm_variances = stats.norm_m2 / stats.evaluated_functions

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
            stats.dimension_variances = (
                stats.dimension_m2 / stats.evaluated_functions
            )

        stats.cef_means, stats.cef_m2 = self._subtract_means_and_m2(
            self.cef_means,
            other.cef_means,
            self.cef_m2,
            other.cef_m2,
            self.evaluated_functions,
            other.evaluated_functions,
        )
        stats.cef_variances = stats.cef_m2 / stats.evaluated_functions

        # Reduce jump times, if collected
        if stats.collect_jump_times:
            if not self._is_sublist(other.jump_times, self.jump_times):
                raise ValueError(
                    "Jump times of subtrahend are not part of "
                    "the jump times of the minuend. Please "
                    "ensure that the functions added to the "
                    "subtrahend are a subset of those added to "
                    "the minuend."
                )
            stats.jump_times = self.list_difference(
                other.jump_times, self.jump_times
            )
        if stats.collect_dimension_jump_times:
            if not self._is_sublist(
                other.dimension_jump_times, self.dimension_jump_times
            ):
                raise ValueError(
                    "Dimension jump times of subtrahend are not "
                    "part of the jump times of the minuend. "
                    "Please ensure that the functions added to "
                    "the subtrahend are a subset of those added "
                    "to the minuend."
                )
            stats.dimension_jump_times = self.list_difference(
                other.dimension_jump_times, self.dimension_jump_times
            )

        return stats

    def __eq__(self, other):
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

    def __ne__(self, other):
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
    def _add_means_and_m2(means_x, means_y, m2_x, m2_y, n, m):
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
    def _subtract_means_and_m2(means_xy, means_y, m2_xy, m2_y, n, m):
        """
        Subtract means and 2nd moments of variable with n and m realizations.

        Parameters
        ----------
        means_xy : np.ndarray
            The means of the minuend variable.
        means_y : np.ndarray
            The means of the substrahend variable.
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

    @staticmethod
    def _is_sublist(list1, list2):
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
    def list_difference(list1, list2):
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
        if not StatsFunctionsOfVariableDimension._is_sublist(list1, list2):
            raise ValueError(f"{list1} is not a sublist of {list2}")

        counter1 = Counter(map(tuple, list1))
        counter2 = Counter(map(tuple, list2))

        result = []
        for item in counter2:
            count_difference = counter2[item] - counter1[item]
            result.extend([list(item)] * count_difference)

        return result

    @staticmethod
    def _partial_t_eval(t_eval, t0, tf):
        """
        Reduces t_eval to values between t0 and tf.

        Prepares the evaluation times by ensuring they are within the specified
        range.

        Parameters
        ----------
            t_eval : list or array-like
                The specified evaluation times.
            t0 : float
                The start of the time range.
            tf : float
                The end of the time range.

        Returns
        -------
            t_eval_partial : np.ndarray
                The filtered evaluation times within the range.
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
