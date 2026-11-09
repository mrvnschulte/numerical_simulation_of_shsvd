"""
This module provides solvers for (stochastic) hybrid systems of variable
dimension and functionalities to store, compute and plot (statistics of)
functions of variable dimension.
"""

from .function_of_variable_dimension import FunctionOfVariableDimension
from .solve_hsvd import solve_hsvd
from .solve_shsvd import solve_shsvd
from .solve_shsvd_single_run import solve_shsvd_single_run
from .stats_functions_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)

__all__ = [
    "FunctionOfVariableDimension",
    "solve_hsvd",
    "solve_shsvd",
    "solve_shsvd_single_run",
    "StatsFunctionsOfVariableDimension",
]
