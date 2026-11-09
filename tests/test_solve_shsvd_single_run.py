"""
Tests for solve_shsvd_single_run.py.
"""

import importlib

import numpy as np
import numpy.testing as npt

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    solve_shsvd_single_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def zero_flow(t, y):
    return np.zeros_like(y)


def unit_rate(t, y):
    return 1.0


def zero_rate(t, y):
    return 0.0


def increment(t, y):
    return np.asarray(y) + 1.0


class SequenceRng:
    """Return predetermined uniform random numbers."""

    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


# ---------------------------------------------------------------------------
# Basic trajectory and random generator
# ---------------------------------------------------------------------------


class TestTrajectory:

    def test_zero_rate_has_no_jumps(self):
        result = solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            t_eval=[0.0, 0.5, 1.0],
            rng=np.random.default_rng(42),
        )

        assert isinstance(result, FunctionOfVariableDimension)
        npt.assert_allclose(result(1.0), [2.0])
        assert result.jump_times() == []

    def test_every_accepted_jump_increments_state(self):
        result = solve_shsvd_single_run(
            zero_flow,
            unit_rate,
            increment,
            [0.0],
            (0.0, 5.0),
            jump_rate_bound=1.0,
            rng=np.random.default_rng(42),
        )

        assert len(result.jump_times()) > 0
        npt.assert_allclose(result(5.0), [float(len(result.jump_times()))])

    def test_rejected_candidate_with_t_eval(self):
        def half_rate(t, y):
            return 0.5

        result = solve_shsvd_single_run(
            zero_flow,
            half_rate,
            increment,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            t_eval=[0.0, 0.5, 1.0],
            rng=SequenceRng([0.2, 0.9, 0.99]),
        )

        npt.assert_allclose(result(1.0), [0.0])
        assert result.jump_times() == []


class TestRandomGenerator:

    def test_default_rng(self):
        result = solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            rng=None,
        )

        npt.assert_allclose(result(1.0), [2.0])

    def test_same_seed_reproduces_trajectory(self):
        def run(seed):
            return solve_shsvd_single_run(
                zero_flow,
                unit_rate,
                increment,
                [0.0],
                (0.0, 5.0),
                jump_rate_bound=1.0,
                rng=np.random.default_rng(seed),
            )

        first = run(42)
        second = run(42)

        npt.assert_allclose(first.times, second.times)
        npt.assert_allclose(first.values, second.values)

    def test_generator_state_advances(self):
        rng = np.random.default_rng(42)
        before = rng.bit_generator.state

        solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            rng=rng,
        )

        assert rng.bit_generator.state != before


# ---------------------------------------------------------------------------
# Callback arguments
# ---------------------------------------------------------------------------


class TestCallbackArguments:

    def test_jump_rate_args(self):
        def rate(t, y, value):
            return value

        result = solve_shsvd_single_run(
            zero_flow,
            rate,
            increment,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            jump_rate_args=(0.0,),
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(result(1.0), [0.0])

    def test_jump_rate_wrapper_passes_arguments(self):
        calls = []

        def rate(t, y, base_rate, factor):
            calls.append((base_rate, factor))
            return base_rate * factor

        result = solve_shsvd_single_run(
            zero_flow,
            rate,
            increment,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            jump_rate_args=(0.2, 2.0),
            check_jump_rate=True,
            rng=np.random.default_rng(42),
        )

        assert calls
        assert all(call == (0.2, 2.0) for call in calls)
        assert result.maximal_time() == 1.0

    def test_transition_args_are_passed(self):
        def transition(t, y, amount):
            return np.asarray(y) + amount

        result = solve_shsvd_single_run(
            zero_flow,
            unit_rate,
            transition,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            transition_args=(3.0,),
            rng=SequenceRng([0.2, 0.1, 0.99]),
        )

        npt.assert_allclose(result(1.0), [3.0])


# ---------------------------------------------------------------------------
# Jump-rate bound
# ---------------------------------------------------------------------------


class TestJumpRateBound:

    def test_zero_bound_is_initialized(self):
        result = solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [2.0],
            (0.0, 1.0),
            jump_rate_bound=0.0,
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(result(1.0), [2.0])

    def test_bound_updated_when_rate_check_finds_larger_rate(self):
        def rate(t, y):
            return 2.0

        result = solve_shsvd_single_run(
            zero_flow,
            rate,
            increment,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1.0,
            check_jump_rate=True,
            rng=SequenceRng([0.2, 0.1, 0.99]),
        )

        npt.assert_allclose(result(1.0), [1.0])

    def test_updated_bound_recomputes_segment(self, monkeypatch):
        module = importlib.import_module(
            "numerical_simulation_of_shsvd.solve_shsvd_single_run"
        )
        received_bounds = []

        def fake_solve_hsvd(flow_function, initial_value, t_span, **kwargs):
            received_bounds.append(kwargs["jump_rate_bound"])

            partial = FunctionOfVariableDimension(
                [t_span[0], t_span[1]],
                [list(initial_value), list(initial_value)],
                checks=False,
            )

            return (
                partial,
                kwargs["flow_args"],
                kwargs["jump_rate_args"],
                kwargs["transition_args"],
                kwargs["jump_time_transition_args"],
                kwargs["jump_set_args"],
                kwargs["jump_set_transition_args"],
                2.0,
            )

        monkeypatch.setattr(module, "solve_hsvd", fake_solve_hsvd)

        result = solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [0.0],
            (0.0, 0.1),
            jump_rate_bound=1.0,
            rng=np.random.default_rng(42),
        )

        assert received_bounds == [1.0, 2.0]
        npt.assert_allclose(result(0.1), [0.0])


# ---------------------------------------------------------------------------
# Forced transitions
# ---------------------------------------------------------------------------


class TestForcedTransitions:

    def test_prescribed_jump_uses_default_transition(self):
        result = solve_shsvd_single_run(
            zero_flow,
            zero_rate,
            increment,
            [0.0],
            (0.0, 2.0),
            jump_rate_bound=1e-10,
            jump_times=[1.0],
            t_eval=[0.0, 1.0, 2.0],
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(result.left_limit(1.0), [0.0])
        npt.assert_allclose(result(2.0), [1.0])

    def test_jump_set_uses_default_transition(self):
        def flow(t, y):
            return np.ones_like(y)

        def event(t, y):
            return y[0] - 0.5

        def reset(t, y):
            return np.array([-10.0])

        result = solve_shsvd_single_run(
            flow,
            zero_rate,
            reset,
            [0.0],
            (0.0, 1.0),
            jump_rate_bound=1e-10,
            jump_set_function=event,
            rng=np.random.default_rng(42),
        )

        npt.assert_allclose(result(1.0), [-9.5], atol=1e-7)
