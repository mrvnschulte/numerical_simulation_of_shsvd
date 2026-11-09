"""
Tests for the public package exports.
"""

import numerical_simulation_of_shsvd as shsvd


class TestPackageExports:
    def test_expected_exports(self):
        assert set(shsvd.__all__) == {
            "FunctionOfVariableDimension",
            "StatsFunctionsOfVariableDimension",
            "solve_hsvd",
            "solve_shsvd",
            "solve_shsvd_single_run",
        }

    def test_all_exports_exist(self):
        for name in shsvd.__all__:
            assert hasattr(shsvd, name)

    def test_solvers_are_callable(self):
        assert callable(shsvd.solve_hsvd)
        assert callable(shsvd.solve_shsvd)
        assert callable(shsvd.solve_shsvd_single_run)
