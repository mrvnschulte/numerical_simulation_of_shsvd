"""
Tests for the internal utility functions in _utils.py.
"""

import numpy as np
import numpy.testing as npt
import pytest

from numerical_simulation_of_shsvd._utils import (
    _add_attributes,
    _assign_args,
    _convert_to_list,
    _is_sublist,
    _list_difference,
    _partial_t_eval,
    _stack_args,
)


# ---------------------------------------------------------------------------
# _add_attributes
# ---------------------------------------------------------------------------


class TestAddAttributes:
    def test_adds_missing_flags(self):
        def transition(t, y):
            return y

        _add_attributes(transition)

        for name in (
            "returns_flow_args",
            "returns_jump_rate_args",
            "returns_transition_args",
            "returns_jump_time_transition_args",
            "returns_jump_set_args",
            "returns_jump_set_transition_args",
            "returns_jump_rate_bound",
        ):
            assert getattr(transition, name) is False

    def test_preserves_existing_flag(self):
        def transition(t, y):
            return y

        transition.returns_flow_args = True

        _add_attributes(transition)

        assert transition.returns_flow_args is True


# ---------------------------------------------------------------------------
# _assign_args
# ---------------------------------------------------------------------------


class TestAssignArgs:
    def test_no_flags_returns_state_unchanged(self):
        state = np.array([1.0, 2.0])

        result = _assign_args(
            state,
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            7.0,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

        assert result[0] is state
        assert result[1:] == (
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            7.0,
        )

    def test_all_flags(self):
        values = (
            [9.0],
            ("flow",),
            ("rate",),
            ("transition",),
            ("time",),
            ("set",),
            ("set_transition",),
            8.0,
        )

        result = _assign_args(
            values,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )

        assert result == values

    @pytest.mark.parametrize("index", range(7))
    def test_each_flag_individually(self, index):
        flags = [False] * 7
        flags[index] = True
        replacement = 2.0 if index == 6 else ("updated",)

        result = _assign_args(
            ([1.0], replacement),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *flags,
        )

        assert result[0] == [1.0]
        assert result[index + 1] == replacement

        for i in range(7):
            if i != index:
                assert result[i + 1] is None


# ---------------------------------------------------------------------------
# _convert_to_list
# ---------------------------------------------------------------------------


class TestConvertToList:
    @pytest.mark.parametrize("value", [1, 1.5, np.float64(2.0)])
    def test_scalar(self, value):
        assert _convert_to_list(value) == [value]

    def test_array(self):
        assert _convert_to_list(np.array([1, 2])) == [1, 2]

    def test_list_returned_unchanged(self):
        value = [1, 2]
        assert _convert_to_list(value) is value

    def test_tuple_raises(self):
        with pytest.raises(TypeError):
            _convert_to_list((1, 2))


# ---------------------------------------------------------------------------
# _is_sublist
# ---------------------------------------------------------------------------


class TestIsSublist:
    def test_counts_duplicates(self):
        assert _is_sublist([[1], [1]], [[1], [2], [1]])
        assert not _is_sublist([[1], [1]], [[1], [2]])

    def test_empty_sublist(self):
        assert _is_sublist([], [[1]])


# ---------------------------------------------------------------------------
# _list_difference
# ---------------------------------------------------------------------------


class TestListDifference:
    def test_counts_duplicates(self):
        assert _list_difference([[1]], [[1], [2], [1]]) == [[1], [2]]

    def test_missing_entry_raises(self):
        with pytest.raises(ValueError):
            _list_difference([[1], [1]], [[1]])


# ---------------------------------------------------------------------------
# _partial_t_eval
# ---------------------------------------------------------------------------


class TestPartialTEval:
    def test_none(self):
        assert _partial_t_eval(None, 0.0, 1.0) is None

    def test_filters_and_adds_endpoints(self):
        result = _partial_t_eval([-1.0, 0.5, 2.0], 0.0, 1.0)
        npt.assert_array_equal(result, [0.0, 0.5, 1.0])

    def test_existing_endpoints_not_duplicated(self):
        result = _partial_t_eval([0.0, 0.5, 1.0], 0.0, 1.0)
        npt.assert_array_equal(result, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# _stack_args
# ---------------------------------------------------------------------------


class TestStackArgs:
    @pytest.mark.parametrize(
        "jump_type, expected",
        [
            (
                "time",
                (
                    "time",
                    "flow",
                    "rate",
                    "transition",
                    "set",
                    "set_transition",
                    7.0,
                ),
            ),
            (
                "set",
                (
                    "set_transition",
                    "flow",
                    "rate",
                    "transition",
                    "time",
                    "set",
                    7.0,
                ),
            ),
            (
                "stochastic",
                (
                    "transition",
                    "flow",
                    "rate",
                    "time",
                    "set",
                    "set_transition",
                    7.0,
                ),
            ),
        ],
    )
    def test_all_flags(self, jump_type, expected):
        result = _stack_args(
            jump_type,
            ("flow",),
            ("rate",),
            ("transition",),
            ("time",),
            ("set",),
            ("set_transition",),
            7.0,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )

        assert result == expected

    def test_no_arguments(self):
        result = _stack_args(
            "time",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

        assert result == ()
