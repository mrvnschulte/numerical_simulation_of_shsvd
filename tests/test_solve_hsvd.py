"""
Tests for solve_hsvd.py.
"""

import numpy as np
import numpy.testing as npt
import pytest

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    solve_hsvd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def constant_flow(t, y, rate=1.0):
    return np.full_like(y, rate, dtype=float)


def add_two(t, y):
    return np.asarray(y) + 2.0


# ---------------------------------------------------------------------------
# Continuous dynamics
# ---------------------------------------------------------------------------


class TestContinuousDynamics:

    def test_constant_flow(self):
        result = solve_hsvd(
            constant_flow,
            [1.0],
            (0.0, 2.0),
            t_eval=[0.0, 1.0, 2.0],
        )

        assert isinstance(result, FunctionOfVariableDimension)
        npt.assert_allclose(result(2.0), [3.0], atol=1e-8)

    def test_flow_args(self):
        result = solve_hsvd(
            constant_flow,
            [1.0],
            (0.0, 2.0),
            flow_args=(2.0,),
            t_eval=[0.0, 2.0],
        )

        npt.assert_allclose(result(2.0), [5.0], atol=1e-8)

    def test_return_args(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 1.0),
            flow_args=(2.0,),
            jump_rate_bound=3.0,
            return_args=True,
        )

        assert len(result) == 8
        assert isinstance(result[0], FunctionOfVariableDimension)
        assert result[1] == (2.0,)
        assert result[-1] == 3.0


# ---------------------------------------------------------------------------
# Prescribed jump times
# ---------------------------------------------------------------------------


class TestPrescribedJumps:

    def test_prescribed_jump(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[1.0],
            jump_time_transition_function=add_two,
            t_eval=[0.0, 1.0, 2.0],
        )

        npt.assert_allclose(result.left_limit(1.0), [1.0], atol=1e-8)
        npt.assert_allclose(result(1.0), [3.0], atol=1e-8)
        npt.assert_allclose(result(2.0), [4.0], atol=1e-8)

    def test_dimension_change_at_jump(self):
        def transition(t, y):
            return [y[0], 4.0]

        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[1.0],
            jump_time_transition_function=transition,
            t_eval=[0.0, 1.0, 2.0],
        )

        assert result.dimension_jump_times() == [1.0]
        npt.assert_allclose(result(2.0), [2.0, 5.0], atol=1e-8)

    def test_jump_at_initial_time(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 1.0),
            jump_times=[0.0],
            jump_time_transition_function=add_two,
            t_eval=[0.0, 1.0],
        )

        npt.assert_allclose(result(0.0), [2.0])
        npt.assert_allclose(result(1.0), [3.0], atol=1e-8)

    def test_initial_jump_followed_by_another_jump(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[0.0, 1.0],
            jump_time_transition_function=add_two,
            t_eval=[0.0, 1.0, 2.0],
        )

        npt.assert_allclose(result(2.0), [6.0], atol=1e-8)

    def test_multiple_prescribed_jumps(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 3.0),
            jump_times=[-1.0, 1.0, 2.0],
            jump_time_transition_function=add_two,
            t_eval=[0.0, 1.0, 2.0, 3.0],
        )

        npt.assert_allclose(result(3.0), [7.0], atol=1e-8)
        npt.assert_allclose(result.jump_times(), [1.0, 2.0])

    def test_all_jumps_before_start_are_ignored(self):
        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 1.0),
            jump_times=[-2.0, -1.0],
            jump_time_transition_function=add_two,
        )

        npt.assert_allclose(result(1.0), [1.0], atol=1e-8)

    def test_transition_updates_flow_args(self):
        def transition(t, y, old_rate):
            return np.asarray(y), (3.0,)

        transition.returns_flow_args = True

        result, flow_args, *_ = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[1.0],
            jump_time_transition_function=transition,
            flow_args=(1.0,),
            return_args=True,
            t_eval=[0.0, 1.0, 2.0],
        )

        assert flow_args == (3.0,)
        npt.assert_allclose(result(2.0), [4.0], atol=1e-8)

    def test_missing_time_transition_raises(self):
        with pytest.raises(ValueError, match="transition function"):
            solve_hsvd(
                constant_flow,
                [0.0],
                (0.0, 2.0),
                jump_times=[1.0],
            )


# ---------------------------------------------------------------------------
# Jump-set events
# ---------------------------------------------------------------------------


class TestJumpSet:

    def test_jump_set(self):
        def event(t, y):
            return y[0] - 1.0

        def reset(t, y):
            return np.array([0.0])

        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 1.5),
            jump_set_function=event,
            jump_set_transition_function=reset,
            t_eval=[0.0, 1.0, 1.5],
        )

        npt.assert_allclose(result(1.5), [0.5], atol=1e-7)
        assert len(result.jump_times()) == 1

    def test_time_transition_used_as_fallback(self):
        def event(t, y):
            return y[0] - 1.0

        def reset(t, y):
            return np.array([0.0])

        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 1.5),
            jump_set_function=event,
            jump_time_transition_function=reset,
            jump_set_args=(),
            t_eval=[0.0, 1.5],
        )

        npt.assert_allclose(result(1.5), [0.5], atol=1e-7)

    def test_jump_set_before_prescribed_jump(self):
        def event(t, y, threshold):
            return y[0] - threshold

        def reset(t, y):
            return np.array([-10.0])

        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[1.5],
            jump_time_transition_function=add_two,
            jump_set_function=event,
            jump_set_transition_function=reset,
            jump_set_args=(1.0,),
            t_eval=[0.0, 1.5, 2.0],
        )

        npt.assert_allclose(result(2.0), [-7.0], atol=1e-7)

    def test_prescribed_jump_before_jump_set(self):
        def event(t, y):
            return y[0] - 10.0

        def reset(t, y):
            return np.array([0.0])

        result = solve_hsvd(
            constant_flow,
            [0.0],
            (0.0, 2.0),
            jump_times=[0.5, 1.5],
            jump_time_transition_function=add_two,
            jump_set_function=event,
            jump_set_transition_function=reset,
            t_eval=[0.0, 0.5, 1.5, 2.0],
        )

        npt.assert_allclose(result(2.0), [6.0], atol=1e-8)

    def test_missing_set_transition_raises(self):
        def event(t, y):
            return y[0] - 1.0

        with pytest.raises(ValueError, match="transition function"):
            solve_hsvd(
                constant_flow,
                [0.0],
                (0.0, 2.0),
                jump_set_function=event,
            )
