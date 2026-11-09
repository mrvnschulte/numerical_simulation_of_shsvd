"""
Tests for stats_functions_of_variable_dimension.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    StatsFunctionsOfVariableDimension,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def constant_function(value):
    """Return a constant one-dimensional function on [0, 2]."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 2.0],
        [[value], [value], [value]],
    )


def dimension_jump_function():
    """Return a function with a dimension jump at t=1."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 1.0, 2.0],
        [[1.0], [1.0], [1.0, 2.0], [1.0, 2.0]],
    )


def value_jump_function():
    """Return a function with a value jump at t=1."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 1.0, 2.0],
        [[1.0], [1.0], [2.0], [2.0]],
    )


def first_coordinate(values):
    """Return the first coordinate at every evaluation time."""
    return np.array([value[0] for value in values])


def make_stats(values, **kwargs):
    """Create a collector and add constant functions."""
    stats = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)
    for value in values:
        stats.add_function(constant_function(value))
    return stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stats():
    """Empty collector with norm, dimension, and jump collection."""
    return StatsFunctionsOfVariableDimension(
        (0.0, 2.0),
        stepsize=1.0,
        order=2,
        collect_dimension=True,
        collect_jump_times=True,
        collect_dimension_jump_times=True,
    )


@pytest.fixture
def stats_with_functions(stats):
    """Collector with two constant functions."""
    stats.add_function(constant_function(1.0))
    stats.add_function(constant_function(3.0))
    return stats

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:

    def test_default_grid(self, stats):
        npt.assert_array_equal(stats.times, [0.0, 1.0, 2.0])

    def test_initial_count_is_zero(self, stats):
        assert stats.evaluated_functions == 0

    def test_initial_norm_arrays(self, stats):
        assert stats.norm_means.shape == (1, 3)
        npt.assert_array_equal(stats.norm_means, np.zeros((1, 3)))
        npt.assert_array_equal(stats.norm_variances, np.zeros((1, 3)))
        npt.assert_array_equal(stats.norm_m2, np.zeros((1, 3)))

    def test_initial_dimension_arrays(self, stats):
        npt.assert_array_equal(stats.dimension_means, np.zeros(3))
        npt.assert_array_equal(stats.dimension_variances, np.zeros(3))
        npt.assert_array_equal(stats.dimension_m2, np.zeros(3))

    def test_initial_jump_lists_are_empty(self, stats):
        assert stats.jump_times == []
        assert stats.dimension_jump_times == []

    def test_custom_grid_adds_endpoints(self):
        result = StatsFunctionsOfVariableDimension(
            (0.0, 2.0),
            t_eval=[0.5, 1.5],
        )
        npt.assert_array_equal(result.times, [0.0, 0.5, 1.5, 2.0])

    def test_custom_grid_filters_outside_times(self):
        result = StatsFunctionsOfVariableDimension(
            (0.0, 2.0),
            t_eval=[-1.0, 0.5, 1.5, 3.0],
        )
        npt.assert_array_equal(result.times, [0.0, 0.5, 1.5, 2.0])

    @pytest.mark.parametrize("order", [1, 2.0, np.inf])
    def test_scalar_order(self, order):
        result = StatsFunctionsOfVariableDimension((0.0, 2.0), order=order)
        assert result.order == [order]

    def test_multiple_orders(self):
        result = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), order=[1, 2, np.inf]
        )
        assert result.order == [1, 2, np.inf]
        assert result.norm_means.shape == (3, 3)

    def test_invalid_order_raises(self):
        with pytest.raises(TypeError, match="Order"):
            StatsFunctionsOfVariableDimension((0.0, 2.0), order="invalid")

    def test_dimension_not_collected(self):
        result = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), collect_dimension=False
        )
        assert result.collect_dimension is False
        assert not hasattr(result, "dimension_means")

    def test_jump_times_not_collected(self):
        result = StatsFunctionsOfVariableDimension((0.0, 2.0))
        assert not hasattr(result, "jump_times")
        assert not hasattr(result, "dimension_jump_times")

# ---------------------------------------------------------------------------
# Add functions
# ---------------------------------------------------------------------------


class TestAddFunction:

    def test_count_increases(self, stats):
        stats.add_function(constant_function(2.0))
        assert stats.evaluated_functions == 1

    def test_first_sample_mean(self, stats):
        stats.add_function(constant_function(2.0))
        npt.assert_allclose(stats.norm_means[0], [2.0, 2.0, 2.0])

    def test_first_sample_variance_is_zero_array(self, stats):
        stats.add_function(constant_function(2.0))

        assert isinstance(stats.norm_variances, np.ndarray)
        npt.assert_array_equal(stats.norm_variances[0], [0.0, 0.0, 0.0])
        npt.assert_array_equal(stats.dimension_variances, [0.0, 0.0, 0.0])

    def test_mean_and_sample_variance(self, stats_with_functions):
        stats = stats_with_functions

        npt.assert_allclose(stats.norm_means[0], [2.0, 2.0, 2.0])
        npt.assert_allclose(stats.norm_variances[0], [2.0, 2.0, 2.0])
        npt.assert_allclose(stats.norm_m2[0], [2.0, 2.0, 2.0])

    def test_dimension_statistics(self, stats_with_functions):
        stats = stats_with_functions

        npt.assert_allclose(stats.dimension_means, [1.0, 1.0, 1.0])
        npt.assert_allclose(stats.dimension_variances, [0.0, 0.0, 0.0])

    def test_dimension_jump_is_collected(self, stats):
        stats.add_function(dimension_jump_function())

        assert stats.jump_times == [[1.0]]
        assert stats.dimension_jump_times == [[1.0]]

    def test_value_jump_is_not_dimension_jump(self, stats):
        stats.add_function(value_jump_function())

        assert stats.jump_times == [[1.0]]
        assert stats.dimension_jump_times == [[]]

    def test_wrong_type_raises(self, stats):
        with pytest.raises(TypeError):
            stats.add_function("not a function")

    def test_short_time_range_raises(self, stats):
        func = FunctionOfVariableDimension([0.0, 1.0], [[1.0], [1.0]])

        with pytest.raises(ValueError):
            stats.add_function(func)

    def test_collection_disabled(self):
        stats = StatsFunctionsOfVariableDimension(
            (0.0, 2.0),
            collect_dimension=False,
            collect_jump_times=False,
            collect_dimension_jump_times=False,
        )
        stats.add_function(constant_function(1.0))

        assert stats.evaluated_functions == 1
        assert not hasattr(stats, "dimension_means")
        assert not hasattr(stats, "jump_times")

# ---------------------------------------------------------------------------
# Custom evaluation functions
# ---------------------------------------------------------------------------


class TestAddCustomEvaluationFunction:

    def test_registers_function(self, stats):
        stats.add_custom_evaluation_function(first_coordinate)

        assert stats.custom_evaluation_functions == [first_coordinate]
        npt.assert_array_equal(stats.cef_means[0], np.zeros(3))

    def test_custom_mean_and_variance(self, stats):
        stats.add_custom_evaluation_function(first_coordinate)
        stats.add_function(constant_function(1.0))
        stats.add_function(constant_function(3.0))

        npt.assert_allclose(stats.cef_means[0], [2.0, 2.0, 2.0])
        npt.assert_allclose(stats.cef_variances[0], [2.0, 2.0, 2.0])

    def test_cannot_register_after_adding_function(self, stats):
        stats.add_function(constant_function(1.0))

        with pytest.raises(ValueError):
            stats.add_custom_evaluation_function(first_coordinate)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlots:

    def teardown_method(self):
        plt.close("all")

    def test_plot_norm(self, stats_with_functions):
        fig = stats_with_functions.plot_norm()

        assert isinstance(fig, plt.Figure)
        npt.assert_allclose(fig.axes[0].lines[0].get_ydata(), [2.0, 2.0, 2.0])

    def test_plot_dimension(self, stats_with_functions):
        fig = stats_with_functions.plot_dimension()

        assert isinstance(fig, plt.Figure)
        npt.assert_allclose(fig.axes[0].lines[0].get_ydata(), [1.0, 1.0, 1.0])

    def test_plot_cefs(self, stats):
        stats.add_custom_evaluation_function(first_coordinate)
        stats.add_function(constant_function(2.0))

        fig = stats.plot_cefs()

        assert isinstance(fig, plt.Figure)
        npt.assert_allclose(fig.axes[0].lines[0].get_ydata(), [2.0, 2.0, 2.0])

    def test_jump_plots(self, stats):
        stats.add_function(value_jump_function())

        assert isinstance(stats.plot_jump_times(number=1), plt.Figure)
        assert isinstance(stats.plot_inter_jump_times(number=1), plt.Figure)
        assert isinstance(stats.plot_number_of_jumps(), plt.Figure)

    def test_unknown_norm_order_raises(self, stats):
        with pytest.raises(ValueError):
            stats.plot_norm(order=1)

    def test_missing_collection_raises(self):
        stats = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), collect_dimension=False
        )

        with pytest.raises(ValueError):
            stats.plot_dimension()

        with pytest.raises(ValueError):
            stats.plot_jump_times(number=1)

        with pytest.raises(ValueError):
            stats.plot_inter_jump_times(number=1)

        with pytest.raises(ValueError):
            stats.plot_number_of_jumps()

    @pytest.mark.parametrize(
        "method_name",
        ["plot_jump_times", "plot_inter_jump_times"],
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"number": [1, 2]},
            {"number": [1, 2], "separate": True},
            {"number": [1, 2], "separate_plots": True},
            {"number": None},
        ],
    )
    def test_jump_histogram_options(self, method_name, kwargs):
        stats = StatsFunctionsOfVariableDimension(
            (0.0, 3.0), collect_jump_times=True
        )
        func = FunctionOfVariableDimension(
            [0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            [[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]],
        )
        stats.add_function(func)

        fig = getattr(stats, method_name)(
            title="Jumps",
            legend=True,
            xlabel="Time",
            ylabel="Count",
            grid=True,
            **kwargs,
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_number_of_jumps_options(self, stats):
        stats.add_function(value_jump_function())

        fig = stats.plot_number_of_jumps(
            title="Jumps",
            legend=True,
            xlabel="Count",
            ylabel="Frequency",
            grid=True,
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_dimension_options(self, stats_with_functions):
        fig = stats_with_functions.plot_dimension(
            legend=True,
            xlabel="Time",
            ylabel="Dimension",
            grid=True,
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_cef_options(self, stats):
        stats.add_custom_evaluation_function(first_coordinate)
        stats.add_function(constant_function(1.0))

        fig = stats.plot_cefs(
            legend=True,
            xlabel="Time",
            ylabel="Value",
            grid=True,
        )

        assert isinstance(fig, plt.Figure)

    def test_plot_norm_multiple_orders_with_grid(self):
        stats = StatsFunctionsOfVariableDimension((0.0, 2.0), order=[1, 2])
        func = FunctionOfVariableDimension(
            [0.0, 2.0],
            [[3.0, 4.0], [3.0, 4.0]],
        )
        stats.add_function(func)

        fig = stats.plot_norm(order=[1, 2], grid=True)

        assert len(fig.axes[0].get_lines()) == 2
        npt.assert_allclose(fig.axes[0].lines[0].get_ydata(), [7.0] * 3)
        npt.assert_allclose(fig.axes[0].lines[1].get_ydata(), [5.0] * 3)
        assert fig.axes[0].xaxis.get_gridlines()[0].get_visible()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:

    def test_norm_at_scalar_time(self, stats):
        func = FunctionOfVariableDimension(
            [0.0, 1.0, 2.0], [[1.0], [3.0], [5.0]]
        )
        stats.add_function(func)

        npt.assert_allclose(stats.evaluate_norm(0.5), [2.0])

    def test_norm_at_multiple_times(self, stats):
        func = FunctionOfVariableDimension(
            [0.0, 1.0, 2.0], [[1.0], [3.0], [5.0]]
        )
        stats.add_function(func)

        result = stats.evaluate_norm([0.0, 2.0])
        npt.assert_allclose(result, [[1.0], [5.0]])

    def test_norm_variance(self, stats_with_functions):
        npt.assert_allclose(
            stats_with_functions.evaluate_norm_variance(1.0), [2.0]
        )

    def test_dimension_at_final_time(self, stats_with_functions):
        assert stats_with_functions.evaluate_dimension(2.0) == pytest.approx(
            1.0
        )

    def test_dimension_variance_at_final_time(self, stats_with_functions):
        result = stats_with_functions.evaluate_dimension_variances(2.0)
        assert result == pytest.approx(0.0)

    def test_dimension_at_multiple_times(self, stats_with_functions):
        result = stats_with_functions.evaluate_dimension([0.0, 2.0])
        npt.assert_allclose(result, [1.0, 1.0])

    def test_cef_mean_and_variance(self, stats):
        stats.add_custom_evaluation_function(first_coordinate)
        stats.add_function(constant_function(1.0))
        stats.add_function(constant_function(3.0))

        npt.assert_allclose(stats.evaluate_cefs(2.0), [2.0])
        npt.assert_allclose(stats.evaluate_cef_variances(2.0), [2.0])

    @pytest.mark.parametrize("time", [-1.0, 3.0])
    def test_outside_range_raises(self, stats, time):
        with pytest.raises(ValueError):
            stats.evaluate_norm(time)

    def test_dimension_not_collected_raises(self):
        stats = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), collect_dimension=False
        )

        with pytest.raises(AttributeError):
            stats.evaluate_dimension(1.0)

        with pytest.raises(AttributeError):
            stats.evaluate_dimension_variances(1.0)

# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


class TestAdd:

    def test_addition_matches_direct_collection(self):
        left = make_stats([1.0, 2.0])
        right = make_stats([3.0, 4.0])
        direct = make_stats([1.0, 2.0, 3.0, 4.0])

        result = left + right

        assert result.evaluated_functions == 4
        npt.assert_allclose(result.norm_means, direct.norm_means)
        npt.assert_allclose(result.norm_variances, direct.norm_variances)
        npt.assert_allclose(result.dimension_means, direct.dimension_means)

    def test_different_order_arrangement(self):
        left = make_stats([1.0, 2.0], order=[1, 2])
        right = make_stats([3.0, 4.0], order=[2, 1])
        direct = make_stats([1.0, 2.0, 3.0, 4.0], order=[1, 2])

        result = left + right

        assert result.order == [1, 2]
        npt.assert_allclose(result.norm_means, direct.norm_means)
        npt.assert_allclose(result.norm_variances, direct.norm_variances)

    def test_incompatible_orders_raise(self):
        left = make_stats([1.0], order=1)
        right = make_stats([2.0], order=2)

        with pytest.raises(ValueError):
            left + right

    def test_incompatible_collection_settings_raise(self):
        left = make_stats([1.0], collect_dimension=True)
        right = make_stats([2.0], collect_dimension=False)

        with pytest.raises(ValueError):
            left + right

    def test_unsupported_type_returns_not_implemented(self, stats):
        assert stats.__add__("invalid") is NotImplemented

    def test_addition_combines_collected_jump_times(self):
        kwargs = {
            "collect_jump_times": True,
            "collect_dimension_jump_times": True,
        }
        left = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)
        right = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)

        left.add_function(value_jump_function())
        right.add_function(dimension_jump_function())

        result = left + right

        assert result.jump_times == [[1.0], [1.0]]
        assert result.dimension_jump_times == [[], [1.0]]

# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


class TestSub:

    def test_subtraction_recovers_remaining_statistics(self):
        total = make_stats([1.0, 2.0, 3.0, 4.0])
        removed = make_stats([3.0, 4.0])
        expected = make_stats([1.0, 2.0])

        result = total - removed

        assert result.evaluated_functions == 2
        npt.assert_allclose(result.norm_means, expected.norm_means)
        npt.assert_allclose(result.norm_m2, expected.norm_m2)
        npt.assert_allclose(result.norm_variances, expected.norm_variances)
        npt.assert_allclose(result.dimension_means, expected.dimension_means)
        npt.assert_allclose(
            result.dimension_variances, expected.dimension_variances
        )

    def test_one_remaining_function_has_zero_variance(self):
        total = make_stats([1.0, 3.0])
        removed = make_stats([3.0])

        result = total - removed

        assert result.evaluated_functions == 1
        npt.assert_allclose(result.norm_means[0], [1.0, 1.0, 1.0])
        npt.assert_array_equal(result.norm_variances[0], [0.0, 0.0, 0.0])
        npt.assert_array_equal(result.dimension_variances, [0.0, 0.0, 0.0])

    def test_subtract_all_returns_empty_collector(self):
        total = make_stats([1.0, 3.0])
        removed = make_stats([1.0, 3.0])

        result = total - removed

        assert result.evaluated_functions == 0
        npt.assert_array_equal(result.times, total.times)
        assert result.order == total.order
        npt.assert_array_equal(
            result.norm_means, np.zeros_like(total.norm_means)
        )
        npt.assert_array_equal(
            result.norm_variances, np.zeros_like(total.norm_variances)
        )
        npt.assert_array_equal(
            result.dimension_means, np.zeros_like(total.dimension_means)
        )

    def test_subtract_all_preserves_settings_and_cefs(self):
        kwargs = {
            "t_eval": [0.0, 0.5, 2.0],
            "order": [1, 2],
            "collect_dimension": True,
            "collect_jump_times": True,
            "collect_dimension_jump_times": True,
        }
        total = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)
        removed = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)

        for collector in (total, removed):
            collector.add_custom_evaluation_function(first_coordinate)
            collector.add_function(constant_function(2.0))

        result = total - removed

        assert result.evaluated_functions == 0
        npt.assert_array_equal(result.times, total.times)
        assert result.order == [1, 2]
        assert result.collect_dimension is True
        assert result.collect_jump_times is True
        assert result.collect_dimension_jump_times is True
        assert result.jump_times == []
        assert result.dimension_jump_times == []
        assert result.custom_evaluation_functions == [first_coordinate]
        npt.assert_array_equal(result.cef_means[0], np.zeros(3))
        npt.assert_array_equal(result.cef_variances[0], np.zeros(3))

        # The returned collector can be used like a newly initialized one.
        result.add_function(constant_function(4.0))
        assert result.evaluated_functions == 1
        npt.assert_allclose(result.cef_means[0], [4.0, 4.0, 4.0])

    def test_subtract_empty_collectors(self):
        left = StatsFunctionsOfVariableDimension((0.0, 2.0))
        right = StatsFunctionsOfVariableDimension((0.0, 2.0))

        result = left - right

        assert result.evaluated_functions == 0
        npt.assert_array_equal(result.norm_means, np.zeros((1, 3)))

    def test_different_order_arrangement(self):
        total = make_stats([1.0, 2.0, 3.0, 4.0], order=[1, 2])
        removed = make_stats([3.0, 4.0], order=[2, 1])
        expected = make_stats([1.0, 2.0], order=[1, 2])

        result = total - removed

        npt.assert_allclose(result.norm_means, expected.norm_means)
        npt.assert_allclose(result.norm_variances, expected.norm_variances)

    def test_custom_evaluation_statistics(self):
        total = StatsFunctionsOfVariableDimension((0.0, 2.0))
        removed = StatsFunctionsOfVariableDimension((0.0, 2.0))
        expected = StatsFunctionsOfVariableDimension((0.0, 2.0))

        for collector in (total, removed, expected):
            collector.add_custom_evaluation_function(first_coordinate)

        for value in [1.0, 2.0, 3.0, 4.0]:
            total.add_function(constant_function(value))

        for value in [3.0, 4.0]:
            removed.add_function(constant_function(value))

        for value in [1.0, 2.0]:
            expected.add_function(constant_function(value))

        result = total - removed

        npt.assert_allclose(result.cef_means, expected.cef_means)
        npt.assert_allclose(result.cef_variances, expected.cef_variances)

    def test_more_functions_in_subtrahend_raises(self):
        total = make_stats([1.0])
        removed = make_stats([1.0, 2.0])

        with pytest.raises(ValueError):
            total - removed

    def test_incompatible_orders_raise(self):
        total = make_stats([1.0, 2.0], order=1)
        removed = make_stats([2.0], order=2)

        with pytest.raises(ValueError):
            total - removed

    def test_unsupported_type_returns_not_implemented(self, stats):
        assert stats.__sub__("invalid") is NotImplemented

    def test_subtraction_rejects_unmatched_dimension_jump_times(self):
        kwargs = {
            "collect_jump_times": True,
            "collect_dimension_jump_times": True,
        }
        total = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)
        removed = StatsFunctionsOfVariableDimension((0.0, 2.0), **kwargs)

        # Both collectors have a jump at t=1, so the jump-time check passes.
        # Only the subtrahend has a dimension jump there.
        total.add_function(value_jump_function())
        total.add_function(constant_function(1.0))
        removed.add_function(dimension_jump_function())

        with pytest.raises(ValueError, match="Dimension jump times"):
            total - removed

# ---------------------------------------------------------------------------
# Combination validation
# ---------------------------------------------------------------------------


class TestAdditionalCombinationValidation:

    @pytest.mark.parametrize("operation", ["add", "sub"])
    @pytest.mark.parametrize(
        "left_kwargs, right_kwargs",
        [
            ({"collect_jump_times": True}, {"collect_jump_times": False}),
            (
                {"collect_dimension_jump_times": True},
                {"collect_dimension_jump_times": False},
            ),
            ({"collect_dimension": True}, {"collect_dimension": False}),
        ],
    )
    def test_collection_mismatch(self, operation, left_kwargs, right_kwargs):
        left = make_stats([1.0, 2.0], **left_kwargs)
        right = make_stats([2.0], **right_kwargs)

        with pytest.raises(ValueError):
            if operation == "add":
                left + right
            else:
                left - right

    @pytest.mark.parametrize("operation", ["add", "sub"])
    def test_custom_function_count_mismatch(self, operation):
        left = StatsFunctionsOfVariableDimension((0.0, 2.0))
        right = StatsFunctionsOfVariableDimension((0.0, 2.0))

        left.add_custom_evaluation_function(first_coordinate)
        left.add_function(constant_function(1.0))
        right.add_function(constant_function(1.0))

        with pytest.raises(ValueError):
            if operation == "add":
                left + right
            else:
                left - right

    @pytest.mark.parametrize("operation", ["add", "sub"])
    def test_different_evaluation_times(self, operation):
        left = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), t_eval=[0.0, 0.5, 2.0]
        )
        right = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), t_eval=[0.0, 1.0, 2.0]
        )

        with pytest.raises(ValueError):
            if operation == "add":
                left + right
            else:
                left - right

    def test_subtraction_reduces_collected_jumps(self):
        total = StatsFunctionsOfVariableDimension(
            (0.0, 2.0),
            collect_jump_times=True,
            collect_dimension_jump_times=True,
        )
        removed = StatsFunctionsOfVariableDimension(
            (0.0, 2.0),
            collect_jump_times=True,
            collect_dimension_jump_times=True,
        )

        total.add_function(value_jump_function())
        total.add_function(dimension_jump_function())
        removed.add_function(value_jump_function())

        result = total - removed

        assert result.jump_times == [[1.0]]
        assert result.dimension_jump_times == [[1.0]]

    def test_subtraction_rejects_unmatched_jump_lists(self):
        total = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), collect_jump_times=True
        )
        removed = StatsFunctionsOfVariableDimension(
            (0.0, 2.0), collect_jump_times=True
        )

        total.add_function(constant_function(1.0))
        total.add_function(constant_function(2.0))
        removed.add_function(value_jump_function())

        with pytest.raises(ValueError, match="Jump times"):
            total - removed

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class TestAdditionalComparisons:

    def test_equal_collectors(self):
        first = make_stats([1.0, 2.0])
        second = make_stats([1.0, 2.0])

        assert first == second
        assert not (first != second)

    def test_different_collectors(self):
        first = make_stats([1.0, 2.0])
        second = make_stats([1.0, 3.0])

        assert first != second

    def test_equality_with_other_type(self, stats):
        assert stats.__eq__("invalid") is NotImplemented

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:

    def test_first_update_returns_array_variance(self, stats):
        stats.evaluated_functions = 1

        means, variances, m2 = stats._update_means_and_variances(
            np.array([1.0, 2.0]),
            np.zeros(2),
            np.zeros(2),
        )

        npt.assert_allclose(means, [1.0, 2.0])
        assert isinstance(variances, np.ndarray)
        npt.assert_array_equal(variances, [0.0, 0.0])
        npt.assert_array_equal(m2, [0.0, 0.0])

    def test_interpolate_one_dimensional_values_at_endpoint(self, stats):
        result = stats._interpolate_value(2.0, np.array([1.0, 2.0, 3.0]))
        assert result == pytest.approx(3.0)

    def test_interpolate_two_dimensional_values_at_endpoint(self, stats):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = stats._interpolate_value(2.0, values)

        npt.assert_array_equal(result, [3.0, 6.0])

    def test_add_and_subtract_moments(self):
        cls = StatsFunctionsOfVariableDimension

        combined_mean, combined_m2 = cls._add_means_and_m2(
            np.array([1.0]),
            np.array([3.0]),
            np.array([0.0]),
            np.array([0.0]),
            1,
            1,
        )

        npt.assert_allclose(combined_mean, [2.0])
        npt.assert_allclose(combined_m2, [2.0])

        remaining_mean, remaining_m2 = cls._subtract_means_and_m2(
            combined_mean,
            np.array([3.0]),
            combined_m2,
            np.array([0.0]),
            1,
            1,
        )

        npt.assert_allclose(remaining_mean, [1.0])
        npt.assert_allclose(remaining_m2, [0.0])
