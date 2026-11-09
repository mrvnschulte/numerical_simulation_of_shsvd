"""
Tests for solve_shsvd.py.
"""

import numpy as np
import numpy.testing as npt
import pytest

from numerical_simulation_of_shsvd import (
    StatsFunctionsOfVariableDimension,
    solve_shsvd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def zero_flow(t, y):
    return np.zeros_like(y)


def zero_rate(t, y):
    return 0.0


def unit_rate(t, y):
    return 1.0


def increment(t, y):
    return np.asarray(y) + 1.0


def first_coordinate(values):
    return np.array([value[0] for value in values])


# ---------------------------------------------------------------------------
# Fixed iterations and random generator
# ---------------------------------------------------------------------------


class TestSimulation:

    def test_fixed_iterations(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            iterations=3,
            rng=np.random.default_rng(0),
        )

        assert isinstance(stats, StatsFunctionsOfVariableDimension)
        assert stats.evaluated_functions == 3
        npt.assert_allclose(stats.norm_means[0], [2.0, 2.0])
        npt.assert_allclose(stats.norm_variances[0], [0.0, 0.0])

    def test_default_rng(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            iterations=1,
            rng=None,
        )

        assert stats.evaluated_functions == 1

    def test_same_seed_reproduces_statistics(self):
        def run():
            return solve_shsvd(
                zero_flow,
                unit_rate,
                increment,
                [0.0],
                (0.0, 3.0),
                jump_rate_bound=1.0,
                iterations=4,
                collect_jump_times=True,
                rng=np.random.default_rng(42),
            )

        first = run()
        second = run()

        npt.assert_allclose(first.norm_means, second.norm_means)
        for jumps1, jumps2 in zip(first.jump_times, second.jump_times):
            npt.assert_allclose(jumps1, jumps2)

    def test_verbose_fixed_iterations(self, capsys):
        solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            iterations=2,
            verbose=True,
            rng=np.random.default_rng(42),
        )

        assert "Given number" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Custom evaluation functions
# ---------------------------------------------------------------------------


class TestCustomEvaluationFunctions:

    def test_single_custom_evaluation_function(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            custom_evaluation_functions=first_coordinate,
            iterations=1,
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(stats.cef_means[0], [2.0, 2.0])

    def test_custom_evaluation_list_without_args(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            custom_evaluation_functions=[first_coordinate],
            iterations=1,
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(stats.cef_means[0], [2.0, 2.0])

    def test_custom_evaluation_list_with_none_args(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            custom_evaluation_functions=[first_coordinate],
            cef_args=(None,),
            iterations=1,
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(stats.cef_means[0], [2.0, 2.0])

    def test_custom_evaluation_with_args(self):
        def evaluate(values, factor):
            return np.array([factor * value[0] for value in values])

        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            custom_evaluation_functions=[evaluate],
            cef_args=((3.0,),),
            iterations=2,
            rng=np.random.default_rng(0),
        )

        npt.assert_allclose(stats.cef_means[0], [6.0, 6.0])

    def test_argument_count_mismatch(self):
        with pytest.raises(ValueError, match="number of cef arguments"):
            solve_shsvd(
                zero_flow,
                zero_rate,
                increment,
                [0.0],
                (0.0, 1.0),
                custom_evaluation_functions=[first_coordinate],
                cef_args=(),
                iterations=1,
            )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:

    def test_stops_after_minimum(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            convergence_estimate="dimension",
            min_iterations=2,
            max_iterations=5,
            rng=np.random.default_rng(0),
        )

        assert stats.evaluated_functions == 2

    @pytest.mark.parametrize("estimate_type", ["max", "mean", "int"])
    @pytest.mark.parametrize("mode", ["all", "any", "sum", "weighted_sum"])
    def test_convergence_types_and_modes(self, estimate_type, mode):
        kwargs = {}
        if mode == "weighted_sum":
            kwargs["estimation_weights"] = [[2.0, 1.0]]

        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            convergence_estimate=("norm", 2),
            convergence_estimate_type=estimate_type,
            estimation_mode=mode,
            min_iterations=2,
            max_iterations=4,
            verbose=True,
            rng=np.random.default_rng(42),
            **kwargs,
        )

        assert stats.evaluated_functions == 2

    def test_convergence_at_specific_time(self):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            convergence_estimate="dimension",
            convergence_estimate_type=[("time", 1.0)],
            min_iterations=2,
            max_iterations=4,
            rng=np.random.default_rng(42),
        )

        assert stats.evaluated_functions == 2

    def test_convergence_not_reached(self, capsys):
        stats = solve_shsvd(
            zero_flow,
            zero_rate,
            increment,
            [1.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            convergence_estimate="dimension",
            tol=0.0,
            min_iterations=2,
            max_iterations=3,
            verbose=True,
            rng=np.random.default_rng(42),
        )

        assert stats.evaluated_functions == 3
        assert "without" in capsys.readouterr().out

    def test_invalid_estimation_mode(self):
        with pytest.raises(ValueError, match="estimation"):
            solve_shsvd(
                zero_flow,
                zero_rate,
                increment,
                [1.0],
                (0.0, 1.0),
                convergence_estimate="dimension",
                estimation_mode="invalid",
            )

    @pytest.mark.parametrize(
        "estimate, estimate_type, expected",
        [
            ("invalid", "max", "Convergence estimate"),
            (("invalid", 0), "max", "Convergence estimate"),
            (("norm", 1), "max", "not stored"),
            ("dimension", [("invalid", 1.0)], "Possible options"),
            ("dimension", [("time", 0.5)], "must be in t_eval"),
            ("dimension", ["max", "mean"], "length equal"),
        ],
    )
    def test_invalid_convergence_parameters(
        self, estimate, estimate_type, expected
    ):
        with pytest.raises(ValueError, match=expected):
            solve_shsvd(
                zero_flow,
                zero_rate,
                increment,
                [1.0],
                (0.0, 1.0),
                convergence_estimate=estimate,
                convergence_estimate_type=estimate_type,
            )

    def test_invalid_weight_count(self):
        with pytest.raises(ValueError, match="weigths"):
            solve_shsvd(
                zero_flow,
                zero_rate,
                increment,
                [1.0],
                (0.0, 1.0),
                convergence_estimate="dimension",
                estimation_mode="weighted_sum",
                estimation_weights=[],
            )

    @pytest.mark.parametrize("index", [-1, 1, 2])
    def test_invalid_cef_index_raises(self, index):
        with pytest.raises(ValueError, match="cannot be accessed"):
            solve_shsvd(
                zero_flow,
                zero_rate,
                increment,
                [1.0],
                (0.0, 1.0),
                custom_evaluation_functions=[first_coordinate],
                convergence_estimate=("cef", index),
            )
