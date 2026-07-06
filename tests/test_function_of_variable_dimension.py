"""
Tests for FunctionOfVariableDimension.
"""

import pytest
import numpy as np
from function_of_variable_dimension import FunctionOfVariableDimension


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple():
    """Simple 2D function on [0, 2]."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 6]]
    )


@pytest.fixture
def with_jump():
    """Function with a value jump (no dimension change) at t=1."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 6], [7, 8]]
    )


@pytest.fixture
def with_dim_jump():
    """Function with a dimension jump at t=1: 2D -> 3D."""
    return FunctionOfVariableDimension(
        [0.0, 1.0, 1.0, 2.0],
        [[1, 2], [3, 4], [5, 6, 7], [8, 9, 10]],
    )


@pytest.fixture
def empty():
    return FunctionOfVariableDimension()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty(self, empty):
        assert empty.times == []
        assert empty.values == []

    def test_scalar_input(self):
        f = FunctionOfVariableDimension(0.0, 1.0)
        assert f.times == [0.0]
        assert f.values == [[1.0]]

    def test_list_input(self, simple):
        assert simple.times == [0.0, 1.0, 2.0]
        assert simple.values == [[1, 2], [3, 4], [5, 6]]

    def test_numpy_input(self):
        f = FunctionOfVariableDimension(
            np.array([0.0, 1.0]), [np.array([1, 2]), np.array([3, 4])]
        )
        assert f.times == [0.0, 1.0]
        assert f.values == [np.array([1, 2]), np.array([3, 4])]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            FunctionOfVariableDimension([0.0, 1.0], [[1, 2]])

    def test_non_array_like_raises(self):
        with pytest.raises(ValueError):
            FunctionOfVariableDimension("a", "b")

    def test_duplicate_removal_on_init(self):
        f = FunctionOfVariableDimension(
            [0.0, 1.0, 1.0], [[1, 2], [3, 4], [3, 4]]
        )
        assert f.times.count(1.0) == 1

    def test_sorting_on_init(self):
        f = FunctionOfVariableDimension(
            [2.0, 0.0, 1.0], [[5, 6], [1, 2], [3, 4]]
        )
        assert f.times == [0.0, 1.0, 2.0]

    def test_multiple_jumps_same_time_raises(self):
        with pytest.raises(ValueError):
            FunctionOfVariableDimension(
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [[1], [2], [3], [4], [5]],
            )

    def test_dimension_change_without_jump_raises(self):
        with pytest.raises(ValueError):
            FunctionOfVariableDimension(
                [0.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 6, 7]]
            )

    def test_checks_false_skips_validation(self):
        # No error even though unsorted — checks=False
        f = FunctionOfVariableDimension(
            [2.0, 0.0], [[5, 6], [1, 2]], checks=False
        )
        assert f.times == [2.0, 0.0]


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestCall:
    def test_at_defined_time(self, simple):
        np.testing.assert_array_equal(simple(0.0), [1, 2])

    def test_at_last_time(self, simple):
        np.testing.assert_array_equal(simple(2.0), [5, 6])

    def test_interpolation_midpoint(self, simple):
        result = simple(0.5)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_array_input(self, simple):
        results = simple([0.0, 1.0, 2.0])
        assert len(results) == 3

    def test_outside_range_raises(self, simple):
        with pytest.raises(ValueError):
            simple(-1.0)
        with pytest.raises(ValueError):
            simple(3.0)

    def test_at_jump_time_returns_right_limit(self, with_jump):
        np.testing.assert_array_equal(with_jump(1.0), [5, 6])

    def test_at_dim_jump_time_returns_right_limit(self, with_dim_jump):
        assert len(with_dim_jump(1.0)) == 3


# ---------------------------------------------------------------------------
# _convert_to_list
# ---------------------------------------------------------------------------


class TestConvertToList:
    def test_scalar(self):
        assert FunctionOfVariableDimension._convert_to_list(3.0) == [3.0]

    def test_numpy_array(self):
        result = FunctionOfVariableDimension._convert_to_list(
            np.array([1, 2, 3])
        )
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_list_unchanged(self):
        lst = [1, 2, 3]
        assert FunctionOfVariableDimension._convert_to_list(lst) is lst

    def test_integer(self):
        assert FunctionOfVariableDimension._convert_to_list(5) == [5]


# ---------------------------------------------------------------------------
# add_values
# ---------------------------------------------------------------------------


class TestAddValues:
    def test_add_single(self, simple):
        simple.add_values(3.0, [7, 8])
        assert simple.times[-1] == 3.0
        assert simple.values[-1] == [7, 8]

    def test_add_multiple(self, simple):
        simple.add_values([3.0, 4.0], [[7, 8], [9, 10]])
        assert 3.0 in simple.times
        assert 4.0 in simple.times

    def test_add_duplicate_ignored(self, simple):
        n = len(simple.times)
        simple.add_values(0.0, [1, 2])
        assert len(simple.times) == n

    def test_add_jump(self, simple):
        simple.add_values([2.0, 2.0], [[5, 6], [7, 8]])
        assert simple.times.count(2.0) == 2

    def test_add_checks_false_duplicate_at_end_skipped(self, simple):
        n = len(simple.times)
        simple.add_values(2.0, [5, 6], checks=False)
        assert len(simple.times) == n

    def test_add_out_of_order_sorted(self, simple):
        simple.add_values(0.5, [2, 3])
        idx = simple.times.index(0.5)
        assert simple.times[idx - 1] < 0.5 < simple.times[idx + 1]


# ---------------------------------------------------------------------------
# delete_values
# ---------------------------------------------------------------------------


class TestDeleteValues:
    def test_delete_last_when_nothing_provided(self, simple):
        simple.delete_values()
        assert simple.times[-1] == 1.0

    def test_delete_by_time(self, simple):
        simple.delete_values(1.0)
        assert 1.0 not in simple.times

    def test_delete_by_time_and_value(self, simple):
        simple.delete_values(1.0, [3, 4])
        assert 1.0 not in simple.times

    def test_delete_by_index(self, simple):
        simple.delete_values(indices=0)
        assert 0.0 not in simple.times

    def test_delete_nonexistent_time_raises(self, simple):
        with pytest.raises(ValueError):
            simple.delete_values(99.0)

    def test_delete_nonexistent_value_raises(self, simple):
        with pytest.raises(ValueError):
            simple.delete_values(values=[99, 99])

    def test_delete_dimension_jump_time_raises(self, with_dim_jump):
        with pytest.raises(ValueError):
            with_dim_jump.delete_values(1.0)

    def test_delete_jump_time_removes_both_entries(self, with_jump):
        with_jump.delete_values(1.0)
        assert 1.0 not in with_jump.times

    def test_delete_multiple_times(self, simple):
        simple.delete_values([0.0, 1.0])
        assert 0.0 not in simple.times
        assert 1.0 not in simple.times

    def test_delete_out_of_range_index_raises(self, simple):
        with pytest.raises(ValueError):
            simple.delete_values(indices=99)


# ---------------------------------------------------------------------------
# maximal_time / minimal_time
# ---------------------------------------------------------------------------


class TestMinMaxTime:
    def test_max(self, simple):
        assert simple.maximal_time() == 2.0

    def test_min(self, simple):
        assert simple.minimal_time() == 0.0

    def test_max_empty(self, empty):
        assert simple.maximal_time() == simple.times[-1]  # non-empty guard
        assert empty.maximal_time() == -np.inf

    def test_min_empty(self, empty):
        assert empty.minimal_time() == np.inf


# ---------------------------------------------------------------------------
# dimension
# ---------------------------------------------------------------------------


class TestDimension:
    def test_all_times(self, simple):
        assert simple.dimension() == [2, 2, 2]

    def test_scalar_time(self, simple):
        assert simple.dimension(0.5) == 2

    def test_at_last_time(self, simple):
        assert simple.dimension(2.0) == 2

    def test_array_time(self, simple):
        assert simple.dimension([0.0, 2.0]) == [2, 2]

    def test_after_dim_jump(self, with_dim_jump):
        assert with_dim_jump.dimension(1.5) == 3

    def test_invalid_type_raises(self, simple):
        with pytest.raises(TypeError):
            simple.dimension("a")


# ---------------------------------------------------------------------------
# jump_times
# ---------------------------------------------------------------------------


class TestJumpTimes:
    def test_no_jumps(self, simple):
        assert simple.jump_times() == []

    def test_with_jump(self, with_jump):
        assert with_jump.jump_times() == [1.0]

    def test_return_index(self, with_jump):
        times, indices = with_jump.jump_times(return_index=True)
        assert times == [1.0]
        assert with_jump.times[indices[0]] == 1.0

    def test_dim_jump_included(self, with_dim_jump):
        assert 1.0 in with_dim_jump.jump_times()


# ---------------------------------------------------------------------------
# dimension_jump_times
# ---------------------------------------------------------------------------


class TestDimensionJumpTimes:
    def test_no_dim_jump(self, with_jump):
        assert with_jump.dimension_jump_times() == []

    def test_with_dim_jump(self, with_dim_jump):
        assert with_dim_jump.dimension_jump_times() == [1.0]

    def test_return_index(self, with_dim_jump):
        times, indices = with_dim_jump.dimension_jump_times(return_index=True)
        assert times == [1.0]
        assert with_dim_jump.times[indices[0]] == 1.0

    def test_no_jumps_at_all(self, simple):
        assert simple.dimension_jump_times() == []


# ---------------------------------------------------------------------------
# non_dimension_jump_times
# ---------------------------------------------------------------------------


class TestNonDimensionJumpTimes:
    def test_only_value_jump(self, with_jump):
        assert with_jump.non_dimension_jump_times() == [1.0]

    def test_only_dim_jump(self, with_dim_jump):
        assert with_dim_jump.non_dimension_jump_times() == []

    def test_no_jumps(self, simple):
        assert simple.non_dimension_jump_times() == []

    def test_return_index(self, with_jump):
        times, indices = with_jump.non_dimension_jump_times(return_index=True)
        assert times == [1.0]
        assert with_jump.times[indices[0]] == 1.0


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


class TestNorm:
    def test_default_all_times(self, simple):
        norms = simple.norm()
        assert len(norms) == 3
        assert norms[0] == pytest.approx(np.sqrt(1**2 + 2**2))

    def test_scalar_t(self, simple):
        result = simple.norm(t=0.0)
        assert len(result) == 1

    def test_array_t(self, simple):
        result = simple.norm(t=[0.0, 2.0])
        assert len(result) == 2

    def test_order_1(self, simple):
        norms = simple.norm(order=1)
        assert norms[0] == pytest.approx(3.0)

    def test_return_times(self, simple):
        norms, times = simple.norm(return_times=True)
        assert times == simple.times


# ---------------------------------------------------------------------------
# left_limit
# ---------------------------------------------------------------------------


class TestLeftLimit:
    def test_non_jump_time(self, simple):
        np.testing.assert_array_equal(simple.left_limit(1.0), [3, 4])

    def test_at_jump_time(self, with_jump):
        np.testing.assert_array_equal(with_jump.left_limit(1.0), [3, 4])

    def test_array_input(self, with_jump):
        results = with_jump.left_limit([0.0, 1.0])
        assert len(results) == 2
        np.testing.assert_array_equal(results[1], [3, 4])

    def test_at_dim_jump_time(self, with_dim_jump):
        ll = with_dim_jump.left_limit(1.0)
        assert len(ll) == 2  # left limit is 2D


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------


class TestConcatenate:
    def test_basic(self, simple):
        other = FunctionOfVariableDimension([2.0, 3.0], [[5, 6], [7, 8]])
        result = simple.concatenate(other)
        assert result.maximal_time() == 3.0

    def test_self_empty(self, empty, simple):
        result = empty.concatenate(simple)
        assert result == simple

    def test_other_empty(self, simple, empty):
        result = simple.concatenate(empty)
        assert result == simple

    def test_shared_endpoint_same_value(self):
        f1 = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        f2 = FunctionOfVariableDimension([1.0, 2.0], [[3, 4], [5, 6]])
        result = f1.concatenate(f2)
        assert result.times.count(1.0) == 1  # duplicate removed

    def test_shared_endpoint_different_value(self):
        f1 = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        f2 = FunctionOfVariableDimension([1.0, 2.0], [[9, 9], [5, 6]])
        result = f1.concatenate(f2)
        assert result.times.count(1.0) == 2  # jump created

    def test_non_fovd_raises(self, simple):
        with pytest.raises(TypeError):
            simple.concatenate("not a function")

    def test_dimension_mismatch_raises(self):
        f1 = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        f2 = FunctionOfVariableDimension([2.0, 3.0], [[5, 6, 7], [8, 9, 10]])
        with pytest.raises(ValueError):
            f1.concatenate(f2)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


class TestArithmetic:

    # --- __neg__ ---
    def test_neg(self, simple):
        result = -simple
        np.testing.assert_array_equal(result.values[0], [-1, -2])

    # --- __add__ ---
    def test_add_scalar(self, simple):
        result = simple + 1
        np.testing.assert_array_equal(result.values[0], [2, 3])

    def test_radd_scalar(self, simple):
        result = 1 + simple
        np.testing.assert_array_equal(result.values[0], [2, 3])

    def test_add_array(self, simple):
        result = simple + [10, 20]
        np.testing.assert_array_equal(result.values[0], [11, 22])

    def test_add_function(self, simple):
        result = simple + simple
        np.testing.assert_array_equal(result.values[0], [2, 4])

    def test_add_function_different_range_raises(self, simple):
        other = FunctionOfVariableDimension([0.0, 3.0], [[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            simple + other

    def test_add_unsupported_returns_not_implemented(self, simple):
        assert simple.__add__("x") is NotImplemented

    # --- __sub__ ---
    def test_sub_scalar(self, simple):
        result = simple - 1
        np.testing.assert_array_equal(result.values[0], [0, 1])

    def test_rsub_scalar(self, simple):
        result = 10 - simple
        np.testing.assert_array_equal(result.values[0], [9, 8])

    def test_sub_function(self, simple):
        result = simple - simple
        np.testing.assert_array_equal(result.values[0], [0, 0])

    # --- __mul__ ---
    def test_mul_scalar(self, simple):
        result = simple * 2
        np.testing.assert_array_equal(result.values[0], [2, 4])

    def test_rmul_scalar(self, simple):
        result = 2 * simple
        np.testing.assert_array_equal(result.values[0], [2, 4])

    def test_mul_zero(self, simple):
        result = simple * 0
        np.testing.assert_array_equal(result.values[0], [0, 0])

    def test_mul_function(self, simple):
        result = simple * simple
        np.testing.assert_array_equal(result.values[0], [1, 4])

    # --- __truediv__ ---
    def test_div_scalar(self, simple):
        result = simple / 2
        np.testing.assert_allclose(result.values[0], [0.5, 1.0])

    def test_rtruediv_scalar(self, simple):
        result = 2 / simple
        np.testing.assert_allclose(result.values[0], [2.0, 1.0])

    def test_div_function(self, simple):
        result = simple / simple
        np.testing.assert_allclose(result.values[0], [1.0, 1.0])

    # --- __mod__ ---
    def test_mod_scalar(self, simple):
        result = simple % 3
        np.testing.assert_array_equal(result.values[0], [1, 2])

    def test_rmod_scalar(self, simple):
        result = 10 % simple
        np.testing.assert_array_equal(result.values[0], [0, 0])

    def test_mod_function(self, simple):
        result = simple % simple
        np.testing.assert_array_equal(result.values[0], [0, 0])

    # --- __pow__ ---
    def test_pow_scalar(self, simple):
        result = simple**2
        np.testing.assert_array_equal(result.values[0], [1, 4])

    def test_rpow_scalar(self, simple):
        result = 2**simple
        np.testing.assert_array_equal(result.values[0], [2, 4])

    def test_pow_function(self, simple):
        result = simple**simple
        np.testing.assert_array_equal(result.values[0], [1, 4])

    def test_array_too_short_raises(self, simple):
        with pytest.raises(ValueError):
            simple + [1]


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


class TestComparisons:

    def test_eq_same(self, simple):
        assert simple == FunctionOfVariableDimension(
            [0.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 6]]
        )

    def test_eq_different_values(self, simple):
        other = FunctionOfVariableDimension(
            [0.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 7]]
        )
        assert not (simple == other)

    def test_eq_non_fovd(self, simple):
        assert not (simple == 42)

    def test_ne(self, simple):
        other = FunctionOfVariableDimension(
            [0.0, 1.0, 2.0], [[1, 2], [3, 4], [5, 7]]
        )
        assert simple != other

    def test_gt_true(self):
        f1 = FunctionOfVariableDimension([0.0, 1.0], [[5, 6], [7, 8]])
        f2 = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        assert f1 > f2

    def test_gt_false(self, simple):
        assert not (simple > simple)

    def test_lt_true(self):
        f1 = FunctionOfVariableDimension([0.0, 1.0], [[1, 2], [3, 4]])
        f2 = FunctionOfVariableDimension([0.0, 1.0], [[5, 6], [7, 8]])
        assert f1 < f2

    def test_ge_equal(self, simple):
        assert simple >= simple

    def test_le_equal(self, simple):
        assert simple <= simple

    def test_gt_different_range_raises(self, simple):
        other = FunctionOfVariableDimension([0.0, 3.0], [[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            simple > other

    def test_gt_non_fovd_returns_not_implemented(self, simple):
        assert simple.__gt__(42) is NotImplemented


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternals:

    def test_sort(self):
        f = FunctionOfVariableDimension(
            [2.0, 0.0, 1.0], [[5, 6], [1, 2], [3, 4]], checks=False
        )
        f._sort()
        assert f.times == [0.0, 1.0, 2.0]

    def test_sort_preserves_jump_order(self):
        # Jump at t=1: left limit first, right limit second
        f = FunctionOfVariableDimension(
            [0.0, 1.0, 1.0, 2.0],
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            checks=False,
        )
        f._sort()
        assert f.values[f.times.index(1.0)] == [3, 4]

    def test_check_multiple_jumps_raises(self):
        f = FunctionOfVariableDimension.__new__(FunctionOfVariableDimension)
        f.times = [0.0, 1.0, 1.0, 1.0, 2.0]
        f.values = [[1], [2], [3], [4], [5]]
        with pytest.raises(ValueError):
            f._check_multiple_jumps()

    def test_check_dimension_raises(self):
        f = FunctionOfVariableDimension.__new__(FunctionOfVariableDimension)
        f.times = [0.0, 1.0, 2.0]
        f.values = [[1, 2], [3, 4], [5, 6, 7]]
        with pytest.raises((ValueError, IndexError)):
            f._check_dimension()

    def test_delete_double_values_initialisation(self):
        f = FunctionOfVariableDimension.__new__(FunctionOfVariableDimension)
        times = [0.0, 1.0, 1.0]
        values = [[1, 2], [3, 4], [3, 4]]
        t, v = f._delete_double_values(times, values, initialisation=True)
        assert len(t) == 2

    def test_interpolate_value_exact(self, simple):
        result = simple._interpolate_value(0.0)
        np.testing.assert_array_equal(result, [1, 2])

    def test_interpolate_value_midpoint(self, simple):
        result = simple._interpolate_value(0.5)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_interpolate_value_out_of_range_raises(self, simple):
        with pytest.raises(ValueError):
            simple._interpolate_value(-1.0)
        with pytest.raises(ValueError):
            simple._interpolate_value(5.0)
